#!/usr/bin/env python3
"""Measure forward stages and full train-step latency for the shared SAR segmenter."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from run_shared_sar_tasks_v2 import SharedTaskModel


def timed_cuda(callable_):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    value = callable_()
    end.record()
    torch.cuda.synchronize()
    return value, begin.elapsed_time(end)


def stage_profile(model: SharedTaskModel, image: torch.Tensor, amp: bool) -> dict[str, float]:
    backbone = model.backbone
    timings = {}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        x, timings["stem"] = timed_cuda(lambda: backbone.stem(image))
        p2, timings["p2_vss"] = timed_cuda(lambda: backbone.p2(x))
        x, timings["p3_entry"] = timed_cuda(lambda: backbone.p3_entry(backbone.merge2(p2)))
        p3, timings["p3_block"] = timed_cuda(lambda: backbone.p3(backbone.merge3(x)))
        p4, timings["p4_vss"] = timed_cuda(lambda: backbone.p4(backbone.merge4(p3)))
        _, timings["decoder"] = timed_cuda(lambda: model.decoder((p2, p3, p4), image.shape[-2:]))
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "ours"), required=True)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    model = SharedTaskModel("segmentation", 4, args.variant == "ours", args.state_dim).to(device).train()
    image = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    target = torch.randint(0, 4, (args.batch, args.imgsz, args.imgsz), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print({"variant": args.variant, "batch": args.batch, "imgsz": args.imgsz, "amp": args.amp})
    print("forward_stage_ms", stage_profile(model, image, args.amp))

    values = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = F.cross_entropy(model(image), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end.record()
        torch.cuda.synchronize()
        elapsed = begin.elapsed_time(end)
        if step:
            values.append(elapsed)
            print(f"train_step_{step}_ms={elapsed:.3f}", flush=True)
    print(f"train_step_median_ms={statistics.median(values):.3f}")

    optimizer.zero_grad(set_to_none=True)
    forward_begin = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    forward_begin.record()
    with torch.cuda.amp.autocast(enabled=args.amp):
        loss = F.cross_entropy(model(image), target)
    forward_end.record()
    scaler.scale(loss).backward()
    backward_end.record()
    scaler.step(optimizer)
    scaler.update()
    optimizer_end.record()
    torch.cuda.synchronize()
    print(
        "split_ms",
        {
            "forward": forward_begin.elapsed_time(forward_end),
            "backward": forward_end.elapsed_time(backward_end),
            "optimizer": backward_end.elapsed_time(optimizer_end),
        },
    )

    if args.profile:
        optimizer.zero_grad(set_to_none=True)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = F.cross_entropy(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
