"""Bound artifact checks and independent rendering of all 100 PDF pages.

Run only in the selected server's independent gallery directory, after the
build. The 50 individual PDFs and 50 booklet pages must have identical decoded
RGB pixels, in manifest order, at the same Poppler resolution. This is not a
visual review or a validation of the underlying research experiment.

Previous validation reports are retained under qa/history/. A RUNNING report
invalidates the old success before checks begin; the final report is published
atomically after all checks and QA contact sheets. No visual PASS is generated.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit
import uuid
import xml.etree.ElementTree as ET

from PIL import Image, ImageStat, __version__ as pillow_version

from package_gallery import (
    EXPECTED_FIGURES, EXPECTED_IDS, FORMATS, REQUIRED_CODE_FILES,
    check_destination, checked_path, check_generation_provenance, code_bindings,
    contact_sheet_bindings, expected_sha, file_digest,
    packaging_lock, read_json_bound, relative_parts, utc_now,
)

RENDER_DPI = 110
SCHEMA = "rgsp-figure-artifact-validation-v2"


def ensure_directory(root: Path, relative: str) -> Path:
    parts = relative_parts(relative)
    for count in range(1, len(parts) + 1):
        prefix = "/".join(parts[:count])
        (root / prefix).mkdir(exist_ok=True)
        checked_path(root, prefix, directory=True)
    return root / relative


def atomic_report(root: Path, report: dict) -> None:
    target = root / "qa/validation.json"
    checked_path(root, "qa", directory=True)
    check_destination(target, replace=True)
    descriptor, name = tempfile.mkstemp(prefix=".validation_", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        check_destination(target, replace=True)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def preserve_previous_report(root: Path, run_id: str) -> None:
    previous = root / "qa/validation.json"
    if not previous.exists():
        return
    previous = checked_path(root, "qa/validation.json")
    history = ensure_directory(root, "qa/history") / f"validation_{run_id}.json"
    # No-clobber preservation; an unsupported filesystem fails before any
    # artifact checks or renders. The prior published report is never deleted.
    os.link(previous, history)


def run_tool(*arguments: str) -> str:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    result = subprocess.run(arguments, capture_output=True, text=True, env=environment,
                            timeout=60, check=True)
    return result.stdout + ("\n" + result.stderr if result.stderr else "")


def pdf_pages(path: Path) -> int:
    match = re.search(r"^Pages:\s+(\d+)", run_tool("pdfinfo", str(path)), re.M)
    if match is None:
        raise ValueError(f"pdfinfo did not report a page count: {path.name}")
    return int(match.group(1))


def check_fonts(path: Path, label: str, failures: list[str]) -> str:
    report = run_tool("pdffonts", str(path))
    if "Type 3" in report:
        failures.append(label + ": Type 3 font")
    for line in report.splitlines()[2:]:
        # Columns from the right: embedded, subset, unicode, object, ID.
        columns = line.split()
        if len(columns) >= 5 and columns[-5] != "yes":
            failures.append(label + ": unembedded PDF font")
    return report


def render_page(root: Path, source: Path, page: int, relative: str) -> dict:
    target = root / relative
    # All render paths are in a newly created, run-specific directory.
    if target.exists():
        raise ValueError(f"Refusing to overwrite a PDF render: {relative}")
    run_tool("pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
             "-r", str(RENDER_DPI), "-png", str(source), str(target.with_suffix("")))
    target = checked_path(root, relative)
    with Image.open(target) as opened:
        rgb = opened.convert("RGB")
        size = list(rgb.size)
        digest = hashlib.sha256(f"RGB:{size[0]}x{size[1]}\0".encode("ascii"))
        digest.update(rgb.tobytes())
    return {"file": relative, "sha256": file_digest(root, relative)[1],
            "size": size, "pixels_sha256": digest.hexdigest()}


def bind_inputs_and_outputs(root: Path) -> tuple[dict, dict[str, str]]:
    manifest, manifest_sha = read_json_bound(root, "manifest.json")
    sources, sources_sha = read_json_bound(root, "inputs/source_manifest.json")
    figures = manifest.get("figures")
    if (manifest.get("figure_count") != EXPECTED_FIGURES or not isinstance(figures, list)
            or len(figures) != EXPECTED_FIGURES
            or not all(isinstance(item, dict) for item in figures)
            or {item.get("id") for item in figures} != EXPECTED_IDS):
        raise ValueError("The manifest must contain the documented 50 unique figure IDs")
    bindings = {"manifest.json": manifest_sha, "inputs/source_manifest.json": sources_sha}
    declared: dict[str, str] = {}
    for item in figures:
        files, digests = item.get("files"), item.get("sha256")
        if (not isinstance(files, dict) or set(files) != set(FORMATS)
                or not isinstance(digests, dict) or set(digests) != set(FORMATS)):
            raise ValueError(f"Three formats and hashes are required for {item['id']}")
        for fmt in FORMATS:
            relative = files[fmt]
            parts = relative_parts(relative)
            if parts[0] != fmt or Path(parts[-1]).suffix.lower() != "." + fmt:
                raise ValueError(f"Unexpected {fmt} path: {relative}")
            if relative in declared:
                raise ValueError(f"Duplicate artifact path: {relative}")
            declared[relative] = expected_sha(digests[fmt], relative)
    booklet = manifest.get("booklet")
    if not isinstance(booklet, dict) or booklet.get("file") != "gallery_booklet.pdf":
        raise ValueError("The manifest booklet must be gallery_booklet.pdf")
    declared["gallery_booklet.pdf"] = expected_sha(booklet.get("sha256"), "booklet")
    declared.update(contact_sheet_bindings(manifest))
    entries = sources.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The input source manifest is empty")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid input-source record")
        relative = entry.get("staged")
        if (relative_parts(relative)[0] != "inputs" or relative in declared
                or relative in bindings):
            raise ValueError(f"Duplicate or unsafe input path: {relative}")
        declared[relative] = expected_sha(entry.get("sha256"), relative)
    for relative, expected in declared.items():
        size, actual = file_digest(root, relative)
        if actual != expected:
            raise ValueError(f"Manifest SHA256 mismatch: {relative}")
        if relative_parts(relative)[0] in FORMATS and size < 1000:
            raise ValueError(f"Export is unexpectedly small: {relative}")
        bindings[relative] = actual
    bindings["index.html"] = file_digest(root, "index.html")[1]
    current_code = code_bindings(root)
    bindings.update(current_code)
    check_generation_provenance(
        manifest, {relative: digest for relative, digest in bindings.items()
                   if relative_parts(relative)[0] == "inputs"}, current_code)
    for relative in REQUIRED_CODE_FILES:
        if relative.endswith(".py"):
            ast.parse(checked_path(root, relative).read_text(encoding="utf-8-sig"), filename=relative)
    return manifest, bindings


class GalleryHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references: list[str] = []
        self.cards = 0
        self.figure_choices: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.references.extend(value for key, value in attrs if key in {"href", "src"} and value)
        if tag == "article" and "card" in attributes.get("class", "").split():
            self.cards += 1
        if tag == "input" and attributes.get("value") in EXPECTED_IDS:
            self.figure_choices.append(attributes["value"])


def check_html(root: Path) -> dict:
    parser = GalleryHTMLParser()
    parser.feed(checked_path(root, "index.html").read_text(encoding="utf-8"))
    if (parser.cards != EXPECTED_FIGURES or len(parser.figure_choices) != EXPECTED_FIGURES
            or set(parser.figure_choices) != EXPECTED_IDS):
        raise ValueError("Offline HTML must contain all 50 cards and figure-selection inputs")
    local = set()
    for reference in parser.references:
        if reference.startswith(("#", "data:")):
            continue
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc:
            raise ValueError(f"Unexpected external HTML reference: {reference}")
        relative = unquote(parsed.path)
        checked_path(root, relative)
        local.add(relative)
    return {"cards": parser.cards, "local_links_checked": sorted(local),
            "external_references": [], "browser_interaction_tested": False}


def validate(root: Path, result: dict, run_id: str) -> None:
    manifest, bindings = bind_inputs_and_outputs(root)
    result["bindings"] = bindings
    result["html"] = check_html(root)
    failures, rows = result["failures"], result["figures"]
    run_directory = f"qa/pdf_renders/{run_id}"
    ensure_directory(root, run_directory + "/individual")
    ensure_directory(root, run_directory + "/booklet")
    for index, item in enumerate(manifest["figures"], 1):
        ident = item["id"]
        row = {"id": ident, "files_sha256": item["sha256"]}
        with Image.open(checked_path(root, item["files"]["png"])) as opened:
            row["png_size"] = list(opened.size)
            if min(opened.size) < 700:
                failures.append(ident + ": PNG too small")
            if max(ImageStat.Stat(opened.convert("RGB")).stddev) < 4:
                failures.append(ident + ": PNG appears blank")
        svg = ET.parse(checked_path(root, item["files"]["svg"]))
        row["svg_editable_text_nodes"] = len(svg.findall(".//{http://www.w3.org/2000/svg}text"))
        if row["svg_editable_text_nodes"] < 4:
            failures.append(ident + ": SVG text not editable")
        pdf = checked_path(root, item["files"]["pdf"])
        row["pdf_pages"] = pdf_pages(pdf)
        if row["pdf_pages"] != 1:
            raise ValueError(ident + ": individual PDF is not one page")
        row["fonts_report"] = check_fonts(pdf, ident, failures)
        row["pdf_render"] = render_page(root, pdf, 1, f"{run_directory}/individual/{ident}.png")
        rows.append(row)
        if index % 10 == 0:
            print(json.dumps({"individual_pdfs_rendered": index}), flush=True)

    booklet = checked_path(root, "gallery_booklet.pdf")
    pages = pdf_pages(booklet)
    result["pdf_booklet_pages"] = pages
    if pages != EXPECTED_FIGURES:
        raise ValueError("Booklet is not 50 pages")
    book = {"file": "gallery_booklet.pdf", "sha256": bindings["gallery_booklet.pdf"],
            "render_dpi": RENDER_DPI, "pages": [], "all_pages_match": False,
            "fonts_report": check_fonts(booklet, "booklet", failures)}
    result["booklet"] = book
    for number, individual in enumerate(rows, 1):
        ident = individual["id"]
        rendered = render_page(root, booklet, number,
                               f"{run_directory}/booklet/page-{number:03d}-{ident}.png")
        matches = (rendered["size"] == individual["pdf_render"]["size"]
                   and rendered["pixels_sha256"] == individual["pdf_render"]["pixels_sha256"])
        book["pages"].append({"page": number, "id": ident, "render": rendered,
                              "matches_individual": matches})
        if not matches:
            failures.append(f"Booklet page {number} pixels differ from {ident}")
        if number % 10 == 0:
            print(json.dumps({"booklet_pages_compared": number}), flush=True)
    book["all_pages_match"] = all(page["matches_individual"] for page in book["pages"])

    # Contact sheets are part of validation, not work performed after success.
    # Their run-specific names preserve previous QA output on a repeated check.
    import build_gallery
    if manifest.get("categories") != build_gallery.CATEGORIES:
        raise ValueError("Manifest categories differ from the current gallery builder")
    ensure_directory(root, "contact_sheets")
    for category in build_gallery.CATEGORIES:
        selection = []
        for item in manifest["figures"]:
            if item["category"] == category:
                selected = dict(item)
                selected["files"] = {"png": f"{run_directory}/individual/{item['id']}.png"}
                selection.append(selected)
        if not selection:
            raise ValueError(f"Empty figure category: {category}")
        relative = build_gallery.make_sheet(
            root, selection, f"PDF_QA_{category}_{run_id}",
            columns=3 if len(selection) > 3 else 2, tile_width=680, tile_height=435)
        digest = file_digest(root, relative)[1]
        result["qa_contact_sheets"].append({"category": category, "file": relative, "sha256": digest})
        bindings[relative] = digest
    result["versions"] = {"python": platform.python_version(), "pillow": pillow_version,
                          "matplotlib": build_gallery.matplotlib.__version__,
                          "numpy": build_gallery.np.__version__,
                          "poppler": {name: run_tool(name, "-v").strip().splitlines()[0]
                                      for name in ("pdfinfo", "pdffonts", "pdftoppm")}}
    # Catch changes to any staged input, artifact or code during the whole run.
    for relative, expected in bindings.items():
        if file_digest(root, relative)[1] != expected:
            raise ValueError(f"File changed during validation: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if Path(__file__).resolve() != root / "code/verify_outputs.py":
        raise ValueError("Run the verifier copied into the target gallery's own code directory")
    ensure_directory(root, "qa")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
    with packaging_lock(root):
        preserve_previous_report(root, run_id)
        result = {"schema": SCHEMA, "status": "RUNNING", "run_id": run_id,
                  "started_at_utc": utc_now(), "render_dpi": RENDER_DPI,
                  "figure_count": 0, "pdf_booklet_pages": None,
                  "all_individual_pdf_pages_rendered": False,
                  "all_booklet_pages_rendered": False, "all_pdf_pages_rendered": False,
                  "failures": [], "figures": [], "qa_contact_sheets": [],
                  "visual_review": {"status": "not_performed_by_this_script",
                                    "required_record": "qa/visual_review.json"},
                  "research_validation": False, "new_inference_runs": 0}
        atomic_report(root, result)
        try:
            validate(root, result, run_id)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                detail += "\n" + (exc.stderr or "")[-4000:]
            result["failures"].append(detail)
        result["figure_count"] = len(result["figures"])
        result["all_individual_pdf_pages_rendered"] = result["figure_count"] == EXPECTED_FIGURES
        result["all_booklet_pages_rendered"] = len(result.get("booklet", {}).get("pages", [])) == EXPECTED_FIGURES
        result["all_pdf_pages_rendered"] = (result["all_individual_pdf_pages_rendered"]
                                             and result["all_booklet_pages_rendered"])
        result["finished_at_utc"] = utc_now()
        result["status"] = "FAILED" if result["failures"] else "STRUCTURAL_PASS"
        # This is the last write: no later rendering, contact sheet or check can
        # fail after a success report has been published.
        atomic_report(root, result)
        print(json.dumps({"status": result["status"], "figures": result["figure_count"],
                          "pages": result["pdf_booklet_pages"], "failures": result["failures"],
                          "visual_review": "NOT_PERFORMED"}), flush=True)
        return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
