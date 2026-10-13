"""Verify and package this gallery using only the Python standard library.

Run on an appropriate inventoried server, from any working directory:
    python code/package_gallery.py
    python code/package_gallery.py --replace

The gallery root is always the parent of this script's ``code`` directory.
No business-source paths from the provenance manifest are opened. Only staged
files inside the gallery are read. No model, renderer, or subprocess is run.
The structural report must bind the current artifacts and code. A separate,
explicit visual-review record is required; this script never performs or
invents visual review.

Without --replace, publication uses an atomic, no-clobber hard link. A filesystem
without hard-link support is rejected rather than falling back to an unsafe
check-then-overwrite. With --replace, each prepared file uses os.replace.
ZIP and receipt are two separate atomic publications, not one transaction.
Any partial publication is reported explicitly; an existing published target is
never deleted during error cleanup. A crash-left lock is not removed by guessing
whether its owner is alive.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import zipfile


ARCHIVE_NAME = "rgsp_visualization_gallery_20261010_complete.zip"
RECEIPT_NAME = "package_verification.json"
LOCK_NAME = ".package_gallery.lock"
EXPECTED_FIGURES = 50
FORMATS = ("png", "svg", "pdf")
CHUNK_SIZE = 1024 * 1024
ROOT_FILES = ("README.md", "index.html", "manifest.json", "gallery_booklet.pdf")
ROOT_DIRECTORIES = (
    "code", "inputs", "notes", "png", "svg", "pdf", "contact_sheets", "derived",
)
EXTRA_FILES = ("qa/validation.json", "qa/visual_review.json")
REQUIRED_CODE_FILES = tuple("code/" + name for name in (
    "build_gallery.py", "common.py", "method_diagrams.py", "plot_current.py",
    "plot_historical.py", "plot_protocol.py", "plot_qualitative.py",
    "verify_outputs.py", "package_gallery.py", "stage_inputs.ps1",
))
GENERATION_CODE_FILES = tuple(relative for relative in REQUIRED_CODE_FILES if relative not in {
    "code/verify_outputs.py", "code/package_gallery.py", "code/stage_inputs.ps1",
})
EXPECTED_IDS = frozenset(
    [f"C{i:02d}" for i in range(1, 17)] + [f"H{i:02d}" for i in range(1, 13)]
    + [f"Q{i:02d}" for i in range(1, 13)] + [f"M{i:02d}" for i in range(1, 7)]
    + [f"P{i:02d}" for i in range(1, 5)]
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class PackagingError(RuntimeError):
    """An integrity, scope, or publication requirement was not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise PackagingError(f"Invalid portable relative path: {value!r}")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise PackagingError(f"Absolute, empty, or traversal path is forbidden: {value!r}")
    return parts


def is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def checked_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    """Reject links/junctions at every component before reading a gallery path."""
    path = root
    for part in relative_parts(relative):
        path = path / part
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise PackagingError(f"Required gallery path is absent: {relative}") from exc
        if is_link_or_reparse(info):
            raise PackagingError(f"Symbolic links and reparse points are forbidden: {relative}")
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(info.st_mode):
        raise PackagingError(f"Expected a {'directory' if directory else 'regular file'}: {relative}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise PackagingError(f"Path escapes the gallery root: {relative}")
    return path


def hash_stream(stream) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
        digest.update(block)
        count += len(block)
    return count, digest.hexdigest()


def file_digest(root: Path, relative: str) -> tuple[int, str]:
    with checked_path(root, relative).open("rb") as stream:
        return hash_stream(stream)


def read_json_bound(root: Path, relative: str) -> tuple[dict, str]:
    raw = checked_path(root, relative).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackagingError(f"Invalid UTF-8 JSON: {relative}") from exc
    if not isinstance(value, dict):
        raise PackagingError(f"JSON object required: {relative}")
    return value, hashlib.sha256(raw).hexdigest()


def expected_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise PackagingError(f"Invalid SHA256 in {label}")
    return value.lower()


def code_bindings(root: Path) -> dict[str, str]:
    """Require every documented script, including the input-staging helper."""
    return {relative: file_digest(root, relative)[1] for relative in REQUIRED_CODE_FILES}


def contact_sheet_bindings(manifest: dict) -> dict[str, str]:
    sheets, digests = manifest.get("contact_sheets"), manifest.get("contact_sheet_sha256")
    categories = manifest.get("categories")
    if (not isinstance(categories, dict) or not isinstance(sheets, list)
            or not all(isinstance(relative, str) for relative in sheets)
            or len(sheets) != 1 + len(categories) or len(set(sheets)) != len(sheets)
            or "contact_sheets/START_HERE.png" not in sheets
            or not isinstance(digests, dict) or set(digests) != set(sheets)):
        raise PackagingError("All original contact sheets and their build hashes are required")
    result = {}
    for relative in sheets:
        parts = relative_parts(relative)
        if parts[0] != "contact_sheets" or Path(parts[-1]).suffix.lower() != ".png":
            raise PackagingError(f"Unexpected contact-sheet path: {relative}")
        result[relative] = expected_sha(digests[relative], relative)
    return result


def check_generation_provenance(manifest: dict, input_hashes: dict[str, str],
                                current_code: dict[str, str]) -> None:
    generation = manifest.get("generation")
    if not isinstance(generation, dict) or generation.get("schema") != "rgsp-gallery-generation-v1":
        raise PackagingError("The manifest must bind inputs and plotting code at generation time")
    for label, expected in (
        ("input_bindings", input_hashes),
        ("code_bindings", {relative: current_code[relative] for relative in GENERATION_CODE_FILES}),
    ):
        recorded = generation.get(label)
        if not isinstance(recorded, dict) or set(recorded) != set(expected):
            raise PackagingError(f"Incomplete generation {label}; rebuild the gallery")
        for relative, digest in expected.items():
            if expected_sha(recorded[relative], f"generation {relative}") != digest:
                raise PackagingError(f"Generation inputs/code changed; rebuild the gallery: {relative}")


def verify_visual_review(root: Path, manifest_sha: str, validation_sha: str,
                         booklet_sha: str, figure_order: list[str]) -> tuple[dict, str]:
    review, review_sha = read_json_bound(root, "qa/visual_review.json")
    if (review.get("schema") != "rgsp-figure-visual-review-v1"
            or review.get("status") != "VISUAL_PASS"):
        raise PackagingError("An explicit VISUAL_PASS in qa/visual_review.json is required")
    for key, expected in (
        ("gallery_manifest_sha256", manifest_sha),
        ("structural_report_sha256", validation_sha),
        ("booklet_sha256", booklet_sha),
    ):
        if expected_sha(review.get(key), f"visual review {key}") != expected:
            raise PackagingError(f"Visual review is stale: {key} does not match this build")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise PackagingError("The visual reviewer must be named")
    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        if reviewed_at.utcoffset() is None:
            raise ValueError("timezone is missing")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise PackagingError("Visual reviewed_at must be a timezone-aware ISO-8601 time") from exc
    rows = review.get("figures")
    if (not isinstance(rows, list) or len(rows) != EXPECTED_FIGURES
            or not all(isinstance(row, dict) for row in rows)
            or [row.get("id") for row in rows] != figure_order):
        raise PackagingError("Visual review must cover all 50 figures in manifest order")
    for row in rows:
        if (row.get("status") != "PASS" or not isinstance(row.get("notes"), str)
                or not row["notes"].strip()):
            raise PackagingError(f"Visual review is incomplete for {row.get('id')}")
    booklet = review.get("booklet_review")
    if (not isinstance(booklet, dict) or booklet.get("status") != "PASS"
            or booklet.get("pages_reviewed") != EXPECTED_FIGURES
            or not isinstance(booklet.get("method"), str) or not booklet["method"].strip()):
        raise PackagingError("Visual review must explicitly cover the 50-page booklet")
    return review, review_sha


def verify_metadata(root: Path) -> tuple[dict[str, str], dict]:
    manifest, manifest_sha = read_json_bound(root, "manifest.json")
    validation, validation_sha = read_json_bound(root, "qa/validation.json")
    sources, sources_sha = read_json_bound(root, "inputs/source_manifest.json")
    note = checked_path(root, "notes/EXECUTION_AND_QA.md")
    if note.stat().st_size == 0:
        raise PackagingError("notes/EXECUTION_AND_QA.md is empty")

    figures = manifest.get("figures")
    if (manifest.get("figure_count") != EXPECTED_FIGURES
            or not isinstance(figures, list) or len(figures) != EXPECTED_FIGURES):
        raise PackagingError("manifest.json must contain exactly 50 figures")
    if (validation.get("schema") != "rgsp-figure-artifact-validation-v2"
            or validation.get("status") != "STRUCTURAL_PASS"):
        raise PackagingError("qa/validation.json must be a version-2 STRUCTURAL_PASS report")
    if (validation.get("figure_count") != EXPECTED_FIGURES
            or validation.get("pdf_booklet_pages") != EXPECTED_FIGURES
            or validation.get("all_individual_pdf_pages_rendered") is not True
            or validation.get("all_booklet_pages_rendered") is not True
            or validation.get("failures") != []):
        raise PackagingError("The structural validation report is incomplete or inconsistent")

    hashes = {
        "manifest.json": manifest_sha,
        "qa/validation.json": validation_sha,
        "inputs/source_manifest.json": sources_sha,
    }
    figure_ids: set[str] = set()
    artifact_paths: set[str] = set()
    for figure in figures:
        if not isinstance(figure, dict):
            raise PackagingError("Invalid figure record")
        ident = figure.get("id")
        if not isinstance(ident, str) or not ident or ident in figure_ids:
            raise PackagingError(f"Missing or duplicate figure ID: {ident!r}")
        figure_ids.add(ident)
        files, digests = figure.get("files"), figure.get("sha256")
        if not isinstance(files, dict) or set(files) != set(FORMATS) or not isinstance(digests, dict):
            raise PackagingError(f"PNG, SVG and PDF records are required for {ident}")
        for fmt in FORMATS:
            relative = files[fmt]
            parts = relative_parts(relative)
            if parts[0] != fmt or Path(parts[-1]).suffix.lower() != "." + fmt:
                raise PackagingError(f"Unexpected {fmt} artifact path for {ident}: {relative}")
            if relative in artifact_paths:
                raise PackagingError(f"Multiple figures share an artifact path: {relative}")
            artifact_paths.add(relative)
            hashes[relative] = expected_sha(digests.get(fmt), f"{ident}/{fmt}")
    if figure_ids != EXPECTED_IDS:
        raise PackagingError("Manifest IDs differ from the documented 50-figure collection")

    figure_order = [figure["id"] for figure in figures]
    checked_figures = validation.get("figures")
    if (not isinstance(checked_figures, list)
            or len(checked_figures) != EXPECTED_FIGURES
            or not all(isinstance(item, dict) for item in checked_figures)
            or [item.get("id") for item in checked_figures] != figure_order):
        raise PackagingError("Structural report figure IDs do not match the gallery manifest")

    booklet = manifest.get("booklet")
    if not isinstance(booklet, dict) or booklet.get("file") != "gallery_booklet.pdf":
        raise PackagingError("Manifest booklet must be gallery_booklet.pdf")
    hashes["gallery_booklet.pdf"] = expected_sha(booklet.get("sha256"), "booklet")

    entries = sources.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PackagingError("Staged input manifest has no entries")
    staged_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PackagingError("Invalid staged input record")
        relative = entry.get("staged")
        if relative_parts(relative)[0] != "inputs":
            raise PackagingError(f"Input entry is not inside inputs/: {relative}")
        if relative in staged_paths or relative in hashes:
            raise PackagingError(f"Duplicate or conflicting staged input: {relative}")
        staged_paths.add(relative)
        hashes[relative] = expected_sha(entry.get("sha256"), f"input {relative}")
        # Deliberately ignore entry['source']: it may name a business directory.

    current_code = code_bindings(root)
    check_generation_provenance(
        manifest, {relative: digest for relative, digest in hashes.items()
                   if relative_parts(relative)[0] == "inputs"}, current_code)
    hashes.update(contact_sheet_bindings(manifest))
    bindings = validation.get("bindings")
    required_bindings = ((set(hashes) - {"qa/validation.json"})
                         | set(current_code) | {"index.html"})
    if not isinstance(bindings, dict) or not required_bindings.issubset(bindings):
        raise PackagingError("Structural report is missing artifact, input, HTML or code bindings")
    for relative, value in bindings.items():
        relative_parts(relative)
        digest = expected_sha(value, f"structural binding {relative}")
        if relative in hashes and digest != hashes[relative]:
            raise PackagingError(f"Structural report is stale: {relative}")
        if relative in current_code and digest != current_code[relative]:
            raise PackagingError(f"Code changed since structural validation: {relative}")
        hashes[relative] = digest

    booklet_check = validation.get("booklet")
    if (not isinstance(booklet_check, dict)
            or booklet_check.get("all_pages_match") is not True
            or booklet_check.get("sha256") != hashes["gallery_booklet.pdf"]):
        raise PackagingError("Structural report lacks a bound, fully matched booklet")
    pages = booklet_check.get("pages")
    if (not isinstance(pages, list) or len(pages) != EXPECTED_FIGURES
            or not all(isinstance(page, dict) for page in pages)):
        raise PackagingError("All 50 booklet page-comparison records are required")
    for number, (figure, checked, page) in enumerate(zip(figures, checked_figures, pages), 1):
        if (checked.get("files_sha256") != figure["sha256"]
                or checked.get("pdf_pages") != 1 or page.get("page") != number
                or page.get("id") != figure["id"] or page.get("matches_individual") is not True):
            raise PackagingError(f"Invalid structural page comparison for {figure['id']}")
        rendered, book_rendered = checked.get("pdf_render"), page.get("render")
        if not isinstance(rendered, dict) or not isinstance(book_rendered, dict):
            raise PackagingError(f"Missing render details for {figure['id']}")
        for label, record in (("individual", rendered), ("booklet", book_rendered)):
            expected_sha(record.get("sha256"), f"{figure['id']} {label} render file")
            expected_sha(record.get("pixels_sha256"), f"{figure['id']} {label} render pixels")
            if relative_parts(record.get("file"))[:2] != ("qa", "pdf_renders"):
                raise PackagingError("Render evidence is not inside qa/pdf_renders/")
        if (rendered.get("pixels_sha256") != book_rendered.get("pixels_sha256")
                or rendered.get("size") != book_rendered.get("size")):
            raise PackagingError(f"Booklet pixels do not match {figure['id']}")

    sheets = validation.get("qa_contact_sheets")
    if not isinstance(sheets, list) or len(sheets) != len(manifest.get("categories", {})):
        raise PackagingError("Structural report lacks the complete PDF-QA contact-sheet inventory")
    for sheet in sheets:
        if (not isinstance(sheet, dict)
                or relative_parts(sheet.get("file"))[0] != "contact_sheets"
                or bindings.get(sheet["file"]) != sheet.get("sha256")):
            raise PackagingError("PDF-QA contact sheets must be hash-bound")

    visual, visual_sha = verify_visual_review(
        root, manifest_sha, validation_sha, hashes["gallery_booklet.pdf"], figure_order)
    hashes["qa/visual_review.json"] = visual_sha

    return hashes, {
        "figure_count": EXPECTED_FIGURES,
        "figure_format_files": len(artifact_paths),
        "structural_status": validation["status"],
        "structural_report_sha256": validation_sha,
        "gallery_manifest_sha256": manifest_sha,
        "source_manifest_sha256": sources_sha,
        "source_input_count": len(staged_paths),
        "code_files_bound": len(current_code),
        "booklet_manifest_sha256": hashes["gallery_booklet.pdf"],
        "booklet_pages_pixel_matched": len(pages),
        "visual_status": visual["status"],
        "visual_report_sha256": visual_sha,
        "visually_reviewed_figures": len(visual["figures"]),
        "execution_and_qa_note": "notes/EXECUTION_AND_QA.md",
        "research_validation": False,
    }


def excluded(relative: str) -> bool:
    parts = relative_parts(relative)
    return (any(part.casefold() in {"__pycache__", "transfer"} for part in parts)
            or parts[-1].lower().endswith(".zip")
            or parts[:2] == ("qa", "pdf_renders"))


def collect_payload(root: Path, expected_hashes: dict[str, str]) -> list[dict]:
    selected = set(ROOT_FILES + EXTRA_FILES)
    for dirname in ROOT_DIRECTORIES:
        directory = checked_path(root, dirname, directory=True)
        for current, directories, filenames in os.walk(directory, topdown=True, followlinks=False):
            current_path = Path(current)
            keep = []
            for name in sorted(directories):
                relative = (current_path / name).relative_to(root).as_posix()
                if not excluded(relative):
                    checked_path(root, relative, directory=True)
                    keep.append(name)
            directories[:] = keep
            for name in sorted(filenames):
                relative = (current_path / name).relative_to(root).as_posix()
                if not excluded(relative):
                    checked_path(root, relative)
                    selected.add(relative)

    missing = set(expected_hashes) - selected
    if missing:
        raise PackagingError(f"Validated artifacts are outside the package allowlist: {sorted(missing)}")
    portable_names: set[str] = set()
    payload = []
    for relative in sorted(selected):
        portable = relative.casefold()
        if portable in portable_names:
            raise PackagingError(f"Case-colliding archive path: {relative}")
        portable_names.add(portable)
        size, digest = file_digest(root, relative)
        if relative in expected_hashes and digest != expected_hashes[relative]:
            raise PackagingError(f"Source/manifest SHA256 mismatch: {relative}")
        payload.append({"path": relative, "bytes": size, "sha256": digest})
    return payload


def check_destination(path: Path, replace: bool) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise PackagingError(f"Output target is not an ordinary file: {path.name}")
    if not replace:
        raise PackagingError(f"Refusing to overwrite {path.name}; use --replace only if intended")


@contextmanager
def packaging_lock(root: Path):
    lock = root / LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PackagingError(f"Packaging lock exists: {lock.name}; inspect it before manual recovery") from exc
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            current = lock.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                lock.unlink()
            else:
                print("Warning: lock identity changed; not deleting it", file=sys.stderr)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Warning: could not clean up own lock: {exc}", file=sys.stderr)


def temporary_path(root: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".package_gallery_", suffix=suffix, dir=root)
    os.close(descriptor)
    return Path(name)


def write_archive(root: Path, temporary: Path, payload: list[dict]) -> None:
    with temporary.open("w+b") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True, strict_timestamps=False) as archive:
            for record in payload:
                archive.write(checked_path(root, record["path"]), arcname=record["path"])
        stream.flush()
        os.fsync(stream.fileno())


def verify_archive(temporary: Path, payload: list[dict]) -> list[dict]:
    expected = {item["path"]: item for item in payload}
    results = []
    with zipfile.ZipFile(temporary, "r") as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise PackagingError("ZIP member inventory differs from the frozen payload")
        if any(item.is_dir() for item in members):
            raise PackagingError("Unexpected directory entry in ZIP")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise PackagingError(f"ZIP CRC check failed: {bad_member}")
        for member in members:
            relative_parts(member.filename)
            with archive.open(member, "r") as stream:
                size, digest = hash_stream(stream)
            original = expected[member.filename]
            if (size != original["bytes"] or size != member.file_size
                    or digest != original["sha256"]):
                raise PackagingError(f"Decompressed-byte SHA256/size mismatch: {member.filename}")
            results.append({
                **original,
                "zip_decompressed_bytes": size,
                "zip_decompressed_sha256": digest,
                "zip_crc32": f"{member.CRC:08x}",
                "zip_member_verified": True,
            })
    return results


def write_receipt(temporary: Path, report: dict) -> None:
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish(temporary: Path, destination: Path, replace: bool) -> None:
    check_destination(destination, replace)
    if replace:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise PackagingError(f"Output appeared during packaging; refusing overwrite: {destination.name}") from exc
    except OSError as exc:
        raise PackagingError(
            f"Atomic no-clobber publication unavailable for {destination.name}: {exc}. "
            "No unsafe overwrite fallback was attempted."
        ) from exc
    # Publication is complete. Cleanup failure must not be reported as failure
    # to publish, nor cause deletion of the now-visible destination.
    try:
        temporary.unlink()
    except OSError as exc:
        print(f"Warning: published {destination.name}, temporary cleanup failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true",
                        help="Allow replacement of the ZIP and its package_verification.json receipt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_target, receipt_target = root / ARCHIVE_NAME, root / RECEIPT_NAME
    temporary_files: list[Path] = []
    published: list[str] = []
    try:
        with packaging_lock(root):
            check_destination(archive_target, args.replace)
            check_destination(receipt_target, args.replace)
            expected_hashes, checks = verify_metadata(root)
            payload = collect_payload(root, expected_hashes)
            temporary_zip = temporary_path(root, ".zip.tmp")
            temporary_files.append(temporary_zip)
            write_archive(root, temporary_zip, payload)
            member_results = verify_archive(temporary_zip, payload)
            with temporary_zip.open("rb") as stream:
                zip_bytes, zip_sha256 = hash_stream(stream)

            report = {
                "schema": "rgsp-gallery-package-verification-v2",
                "status": "PACKAGE_VERIFIED",
                "verified_at_utc": utc_now(),
                "gallery_root": str(root),
                "archive": {"file": ARCHIVE_NAME, "bytes": zip_bytes, "sha256": zip_sha256,
                            "member_count": len(member_results), "compression": "ZIP_DEFLATED, level 6"},
                "checks": {**checks, "manifest_bound_files_verified": len(expected_hashes),
                           "zip_member_inventory_match": True,
                           "crc_testzip": {"passed": True, "bad_member": None},
                           "decompressed_sha256": {"verified_members": len(member_results), "failures": [],
                                                   "method": "zipfile.open decompressed byte streams; no extraction directory"}},
                "payload": member_results,
                "exclusions": ["transfer/**", "qa/pdf_renders/**", "**/__pycache__/**", "**/*.zip",
                               RECEIPT_NAME, LOCK_NAME, ".package_gallery_*.tmp"],
                "publication": {"replace_authorized": args.replace,
                                "mode": "os.replace" if args.replace else "atomic no-clobber os.link",
                                "atomic_pair": False,
                                "note": "ZIP and receipt are each atomically published, not a two-file transaction."},
                "new_research_runs": 0,
                "new_inference_runs": 0,
            }
            temporary_receipt = temporary_path(root, ".json.tmp")
            temporary_files.append(temporary_receipt)
            write_receipt(temporary_receipt, report)
            # Both temporary files are complete and the ZIP verified before
            # replacing either published path. The receipt stays outside ZIP.
            publish(temporary_zip, archive_target, args.replace)
            published.append(ARCHIVE_NAME)
            publish(temporary_receipt, receipt_target, args.replace)
            published.append(RECEIPT_NAME)
            print(json.dumps({"status": "PACKAGE_VERIFIED", "archive": str(archive_target),
                              "receipt": str(receipt_target), "zip_sha256": zip_sha256,
                              "payload_files": len(payload), "published": published}, ensure_ascii=False))
            return 0
    except (PackagingError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile,
            zipfile.LargeZipFile) as exc:
        print(json.dumps({"status": "PARTIAL_PUBLICATION" if published else "FAILED",
                          "error": str(exc), "published": published,
                          "note": "Published targets were not deleted or rolled back; inspect any partial publication."},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        for temporary in temporary_files:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"Warning: temporary cleanup failed for {temporary.name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
