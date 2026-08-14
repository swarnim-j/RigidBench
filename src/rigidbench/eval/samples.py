from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2

from rigidbench.core.paths import SamplePaths
from rigidbench.core.sample import RenderedSample


def load_samples(
    data_dir: str | Path,
    split: str = "eval",
    task_type: str | None = None,
) -> list[RenderedSample]:
    """Load samples from the release archive or the original rendered-data layout."""
    data_root = Path(data_dir)
    samples_dir = data_root / split
    if split == "eval" and not samples_dir.is_dir():
        samples_dir = data_root / "samples"
    if not samples_dir.exists():
        return []

    manifest = _load_manifest(data_root)
    out: list[RenderedSample] = []
    for sample_dir in sorted(samples_dir.iterdir()):
        sample = _load_one(sample_dir, manifest.get(sample_dir.name), data_root)
        if sample is None:
            continue
        if task_type and sample.task_type != task_type:
            continue
        out.append(sample)
    return out


def _load_manifest(data_root: Path) -> dict[str, dict]:
    path = data_root / "manifest.json"
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of samples in {path}")
    return {str(row["sample_id"]): row for row in rows if isinstance(row, dict) and "sample_id" in row}


def _load_one(sample_dir: Path, manifest_row: dict | None, data_root: Path) -> RenderedSample | None:
    """Read a sample directory into a RenderedSample, or None if its files are incomplete."""
    if not sample_dir.is_dir():
        return None
    paths = SamplePaths(sample_dir)
    if not all(p.exists() for p in (paths.masks, paths.metadata)):
        return None

    metadata = json.loads(paths.metadata.read_text())
    for key, value in (manifest_row or {}).items():
        metadata.setdefault(key, value)

    prompt = metadata.get("prompt")
    if prompt is None and paths.prompt.is_file():
        prompt = paths.prompt.read_text().strip()
    if not prompt or "task_type" not in metadata:
        return None

    first_frame = paths.first_frame
    reference_video = None
    if not first_frame.is_file():
        if not paths.video.is_file():
            return None
        reference_video = paths.video
        first_frame = _cached_first_frame(reference_video, data_root, sample_dir.name)

    return RenderedSample(
        id=sample_dir.name,
        task_type=metadata["task_type"],
        first_frame=first_frame,
        prompt=prompt,
        gt_mask=paths.masks,
        reference_video=reference_video,
        gt_depth=paths.depth if paths.depth.exists() else None,
        trajectories=paths.trajectories if paths.trajectories.exists() else None,
        metadata=metadata,
    )


def _cached_first_frame(video: Path, data_root: Path, sample_id: str) -> Path:
    destination = data_root / ".rigidbench-cache" / "first-frames" / f"{sample_id}.png"
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode the first frame of {video}")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{sample_id}.", suffix=".png", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        if not cv2.imwrite(str(temporary), frame):
            raise RuntimeError(f"Could not write the cached first frame for {video}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def filter_samples(
    samples: list[RenderedSample],
    sample_ids: list[str] | None = None,
    max_samples: int | None = None,
) -> list[RenderedSample]:
    """Narrow `samples` to specific IDs or a maximum count."""
    if sample_ids:
        ids = set(sample_ids)
        return [s for s in samples if s.id in ids]
    if max_samples:
        return samples[:max_samples]
    return samples
