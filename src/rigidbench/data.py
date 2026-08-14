from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

ARCHIVE_NAME = "rigidbench-eval-v1.1.tar.gz"
ARCHIVE_SIZE = 18_184_558_630
ARCHIVE_MD5 = "b563f279451e95fe2f32d7e258588636"
ARCHIVE_PART_SIZE = 268_435_456
ARCHIVE_PART_COUNT = 68
ARCHIVE_PARTS = tuple(
    (
        f"{ARCHIVE_NAME}.part{part:02d}-of-{ARCHIVE_PART_COUNT:02d}",
        min(ARCHIVE_PART_SIZE, ARCHIVE_SIZE - (part - 1) * ARCHIVE_PART_SIZE),
    )
    for part in range(1, ARCHIVE_PART_COUNT + 1)
)
DOWNLOAD_ROOT = "https://zenodo.org/api/records/21649156/files"
MARKER_NAME = ".rigidbench-eval-v1.1.complete"


def _marker_matches(marker: Path) -> bool:
    if not marker.is_file():
        return False
    try:
        return marker.read_text().strip() == f"{ARCHIVE_MD5}  {ARCHIVE_NAME}"
    except OSError:
        return False


def _download(destination: Path, force: bool = False) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    if destination.exists() and destination.stat().st_size == ARCHIVE_SIZE:
        print(f"Using existing archive: {destination}")
        return
    if partial.exists() and partial.stat().st_size == ARCHIVE_SIZE:
        partial.replace(destination)
        print(f"Using completed download: {destination}")
        return

    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > ARCHIVE_SIZE:
        partial.unlink()
        downloaded = 0

    part_start = 0
    with tqdm(
        total=ARCHIVE_SIZE,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=ARCHIVE_NAME,
    ) as progress:
        for part_name, part_size in ARCHIVE_PARTS:
            part_end = part_start + part_size
            if downloaded >= part_end:
                part_start = part_end
                continue

            part_offset = max(0, downloaded - part_start)
            headers = {"Range": f"bytes={part_offset}-"} if part_offset else {}
            url = f"{DOWNLOAD_ROOT}/{part_name}/content"
            with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                if part_offset and response.status_code != 206:
                    with partial.open("r+b") as file:
                        file.truncate(part_start)
                    progress.update(part_start - downloaded)
                    downloaded = part_start

                with partial.open("ab") as file:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            file.write(chunk)
                            downloaded += len(chunk)
                            progress.update(len(chunk))

            if downloaded != part_end:
                raise RuntimeError(f"Incomplete download of {part_name}. Run the command again to resume.")
            part_start = part_end

    if partial.stat().st_size != ARCHIVE_SIZE:
        raise RuntimeError(
            f"Incomplete download: expected {ARCHIVE_SIZE} bytes, got {partial.stat().st_size}. "
            "Run the command again to resume."
        )
    partial.replace(destination)


def _verify(path: Path) -> None:
    digest = hashlib.md5()  # noqa: S324 - verifies the checksum published by Zenodo
    with (
        path.open("rb") as file,
        tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Verifying",
        ) as progress,
    ):
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            progress.update(len(chunk))
    if digest.hexdigest() != ARCHIVE_MD5:
        raise RuntimeError(f"Checksum mismatch for {path}; delete it and download again.")


def _extract(archive: Path, destination: Path, force: bool = False) -> None:
    marker = destination / MARKER_NAME
    if _marker_matches(marker) and not force and find_data_root(destination) is not None:
        print(f"Evaluation set is already extracted under {destination}")
        return

    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(root) or member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(destination, members=members)  # noqa: S202 - paths are checked above
    marker.write_text(f"{ARCHIVE_MD5}  {ARCHIVE_NAME}\n")


def find_data_root(destination: Path) -> Path | None:
    candidates = [
        destination,
        destination / "rigidbench",
        destination / "rigidbench-eval-v1.1",
        destination / "rigidbench-eval-v1",
    ]
    candidates.extend(path.parent for path in destination.rglob("eval") if path.is_dir())
    candidates.extend(path.parent for path in destination.rglob("manifest.json") if path.is_file())
    for candidate in candidates:
        eval_dir = candidate / "eval"
        if eval_dir.is_dir() and any(path.is_dir() for path in eval_dir.iterdir()):
            return candidate
        samples_dir = candidate / "samples"
        if (
            (candidate / "manifest.json").is_file()
            and samples_dir.is_dir()
            and any(path.is_dir() for path in samples_dir.iterdir())
        ):
            return candidate
    return None


def _write_inputs_manifest(data_root: Path) -> Path:
    from rigidbench.benchmark import export_inputs_jsonl

    manifest = data_root / "inputs.jsonl"
    export_inputs_jsonl(data_root, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the 100-example RigidBench evaluation set from Zenodo.")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Download and extraction directory")
    parser.add_argument("--download-only", action="store_true", help="Do not extract the archive")
    parser.add_argument("--no-verify", action="store_true", help="Skip the published MD5 checksum")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the archive after extraction")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / MARKER_NAME
    if _marker_matches(marker) and not args.force and not args.download_only:
        data_root = find_data_root(args.output)
        if data_root:
            manifest = _write_inputs_manifest(data_root)
            print(f"Evaluation set is already extracted. Use --data-dir {data_root}")
            print(f"Model inputs: {manifest}")
            return 0

    archive = args.output / ARCHIVE_NAME
    _download(archive, force=args.force)
    if not args.no_verify:
        _verify(archive)
    if args.download_only:
        return 0

    _extract(archive, args.output, force=args.force)
    data_root = find_data_root(args.output)
    if data_root:
        manifest = _write_inputs_manifest(data_root)
        print(f"Evaluation data ready. Use --data-dir {data_root}")
        print(f"Model inputs: {manifest}")
    else:
        print(f"Extracted to {args.output}, but no RigidBench sample directory was found.")
    if not args.keep_archive:
        archive.unlink()
    return 0
