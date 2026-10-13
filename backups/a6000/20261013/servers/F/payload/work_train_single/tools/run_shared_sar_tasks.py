#!/usr/bin/env python3
"""Minimal shared VSS/DS-SSM runners for classification, segmentation, and change detection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.nn.modules.mamba_yolo import DS_VSSBlockSweep, SimpleStem, VisionClueMerge, VSSBlock


class ManifestDataset(Dataset):
    def __init__(self, records, task, split, size=256):
        self.records, self.task, self.size = [r for r in records if r.get("split") == split], task, size

    def __len__(self):
        return len(self.records)

    def _image(self, path, crop=None):
        image = Image.open(path).convert("RGB")
        if crop:
            image = image.crop(tuple(crop))
        image = image.resize((self.size, self.size))
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index):
        row = self.records[index]
        if self.task == "classification":
            return self._image(row["path"]), torch.tensor(row["label"], dtype=torch.long)
        if self.task == "segmentation":
            image = self._image(row["image"])
            mask = Image.open(row["mask"]).convert("RGB").resize((self.size, self.size), Image.NEAREST)
            pixels = np.asarray(mask, dtype=np.int16)
            # FUSAR-Map palette: background, red, green, blue, yellow.
            palette = np.asarray(
                [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                dtype=np.int16,
            )
            distances = ((pixels[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3)
            return image, torch.from_numpy(distances.argmin(axis=2).astype("int64"))
        image1, image2 = self._image(row["image1"], row.get("crop")), self._image(row["image2"], row.get("crop"))
        mask = Image.open(row["mask"]).convert("L")
        if row.get("crop"):
            mask = mask.crop(tuple(row["crop"]))
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        return (image2 - image1).abs(), torch.from_numpy((np.asarray(mask) > 0).astype("float32"))[None]


class SharedBackbone(nn.Module):
    def __init__(self, ours, state_dim):
        super().__init__()
        self.stem = SimpleStem(3, 32, 3)
        self.p2 = VSSBlock(32, 32)
        self.m2 = VisionClueMerge(32, 64)
        self.p3_pre = VSSBlock(64, 64)
        self.m3 = VisionClueMerge(64, 128)
        self.p3 = DS_VSSBlockSweep(128, 128, state_dim) if ours else VSSBlock(128, 128)
        self.m4 = VisionClueMerge(128, 256)
        self.p4 = VSSBlock(256, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.p2(x)
        x = self.m2(x)
        x = self.p3_pre(x)
        x = self.m3(x)
        x = self.p3(x)
        p3 = x
        x = self.m4(x)
        x = self.p4(x)
        return p3, x


class SharedTaskModel(nn.Module):
    def __init__(self, task, classes, ours, state_dim):
        super().__init__()
        self.task, self.backbone = task, SharedBackbone(ours, state_dim)
        if task == "classification":
            self.head = nn.Linear(256, classes)
        elif task == "segmentation":
            self.head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, classes, 1))
        else:
            self.head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        p3, deep = self.backbone(x)
        if self.task == "classification":
            return self.head(deep.mean(dim=(2, 3)))
        return nn.functional.interpolate(self.head(p3), size=x.shape[-2:], mode="bilinear", align_corners=False)


def metrics(task, logits, target):
    if task == "classification":
        return {"accuracy": float((logits.argmax(1) == target).float().mean())}
    if task == "segmentation":
        pred = logits.argmax(1)
        classes = logits.shape[1]
        values = []
        for cls in range(classes):
            p, t = pred == cls, target == cls
            tp = (p & t).sum().item(); fp = (p & ~t).sum().item(); fn = (~p & t).sum().item()
            union = tp + fp + fn
            values.append({"iou": tp / (union + 1e-9), "pa": tp / (t.sum().item() + 1e-9), "dice": 2 * tp / (2 * tp + fp + fn + 1e-9)})
        return {"miou": float(np.mean([v["iou"] for v in values])), "mpa": float(np.mean([v["pa"] for v in values])), "dice": float(np.mean([v["dice"] for v in values]))}
    pred = (logits.sigmoid() > 0.5).float()
    tp = (pred * target).sum().item(); fp = (pred * (1-target)).sum().item(); fn = ((1-pred)*target).sum().item(); tn=((1-pred)*(1-target)).sum().item()
    return {"precision": tp/(tp+fp+1e-9), "recall": tp/(tp+fn+1e-9), "f1": 2*tp/(2*tp+fp+fn+1e-9), "iou": tp/(tp+fp+fn+1e-9), "oa": (tp+tn)/(tp+tn+fp+fn+1e-9)}


def run(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    task = manifest["task"]
    classes = len(manifest["classes"]) if task == "classification" else int(manifest.get("classes", 1)) if task == "segmentation" else 1
    model = SharedTaskModel(task, classes, args.variant == "ours", args.state_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss() if task in {"classification", "segmentation"} else nn.BCEWithLogitsLoss()
    train = ManifestDataset(manifest["records"], task, "train", args.imgsz)
    val = ManifestDataset(manifest["records"], task, "val", args.imgsz)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=args.workers) if len(val) else None
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists() and not args.force:
        print(f"SKIP COMPLETE: {result_path}")
        return
    last = out / "last.pt"; best = out / "best.pt"; best_score = -1.0; start = 0; stale_epochs = 0
    if last.exists():
        state = torch.load(last, map_location=args.device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start = state["epoch"] + 1; best_score = state.get("best_score", -1.0); stale_epochs = state.get("stale_epochs", 0)
    if not best.exists() and last.exists():
        torch.save(state, best)
    for epoch in range(start, args.epochs):
        model.train()
        for image, target in loader:
            image, target = image.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(image), target); loss.backward(); optimizer.step()
        score = 0.0
        if val_loader:
            model.eval(); values=[]
            with torch.no_grad():
                for image, target in val_loader:
                    values.append(metrics(task, model(image.to(args.device)), target.to(args.device)))
            score = float(np.mean([v.get("accuracy", v.get("miou", v.get("f1", 0.0))) for v in values]))
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_score": max(best_score, score)}
        torch.save(state, last)
        if score > best_score:
            best_score = score; stale_epochs = 0; torch.save(state, best)
        else:
            stale_epochs += 1
        state["best_score"] = best_score
        state["stale_epochs"] = stale_epochs
        torch.save(state, last)
        print(f"{args.dataset} {args.variant} epoch={epoch+1}/{args.epochs} score={score:.5f}", flush=True)
        if stale_epochs >= args.patience:
            print(f"EARLY STOP: {args.dataset} {args.variant} after {args.patience} stale epochs", flush=True)
            break
    model.load_state_dict(torch.load(best, map_location=args.device)["model"]); model.eval()
    test = ManifestDataset(manifest["records"], task, "test", args.imgsz)
    test_loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    values=[]; begin=time.perf_counter()
    with torch.no_grad():
        for image, target in test_loader: values.append(metrics(task, model(image.to(args.device)), target.to(args.device)))
    elapsed=(time.perf_counter()-begin)/max(len(test),1)*1000
    if not values:
        raise RuntimeError(f"No test records found for {args.dataset}; cannot write a result")
    result={"dataset":args.dataset,"task":task,"variant":args.variant,"state_dim":args.state_dim if args.variant == "ours" else 0,"metric":{k:float(np.mean([v[k] for v in values])) for k in values[0]},"latency_ms":elapsed,"checkpoint":str(best),"epochs":args.epochs,"params":sum(p.numel() for p in model.parameters())}
    result_path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--dataset",required=True); ap.add_argument("--variant",choices=("baseline","ours"),required=True); ap.add_argument("--output",required=True); ap.add_argument("--epochs",type=int,default=100); ap.add_argument("--patience",type=int,default=20); ap.add_argument("--batch",type=int,default=16); ap.add_argument("--imgsz",type=int,default=256); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--state-dim",type=int,default=16); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--force",action="store_true"); run(ap.parse_args())


if __name__ == "__main__": main()
