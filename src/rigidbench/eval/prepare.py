from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from rigidbench.benchmark import DURATION_SECONDS, PROTOCOL
from rigidbench.core.constants import GT_RESOLUTION
from rigidbench.core.io import write_json
from rigidbench.core.paths import OutputPaths, model_output_dir

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
FPS_TOLERANCE = 0.1


class PreparationError(RuntimeError):
    """A prediction set cannot be prepared for evaluation."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    sample_id: str
    path: Path
    sha256: str
    fps: float | None
    resolution: tuple[int, int]
    frame_count: int

    @property
    def duration_seconds(self) -> float | None:
        if self.fps is None:
            return None
        return (self.frame_count - 1) / self.fps


@dataclass(frozen=True, slots=True)
class PreparationResult:
    output_dir: Path
    fps: float
    records: tuple[dict, ...]


def discover_videos(path: str | Path) -> dict[str, Path]:
    """Return supported videos keyed by their filename stem."""
    input_path = Path(path)
    if not input_path.exists():
        raise PreparationError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in VIDEO_SUFFIXES else []
    else:
        files = sorted(
            file for file in input_path.iterdir() if file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES
        )
    if not files:
        raise PreparationError(f"No supported videos found at {input_path}")

    videos: dict[str, Path] = {}
    for file in files:
        if file.stem in videos:
            raise PreparationError(
                f"More than one video has sample ID {file.stem!r}: {videos[file.stem].name}, {file.name}"
            )
        videos[file.stem] = file
    return videos


def select_videos(
    videos: dict[str, Path],
    expected_sample_ids: Collection[str] | None = None,
    selected_sample_ids: Collection[str] | None = None,
    allow_partial: bool = False,
) -> list[Path]:
    """Validate prediction IDs and return the videos to prepare in benchmark order."""
    if expected_sample_ids is None:
        if selected_sample_ids is not None or allow_partial:
            raise PreparationError("Partial preparation requires the benchmark's expected sample IDs.")
        return [videos[sample_id] for sample_id in sorted(videos)]

    expected = _unique_ids(expected_sample_ids, "expected")
    selected = _unique_ids(selected_sample_ids, "selected") if selected_sample_ids is not None else expected
    expected_set = set(expected)
    selected_set = set(selected)
    actual_set = set(videos)

    unknown_selected = selected_set - expected_set
    if unknown_selected:
        raise PreparationError(f"Selected sample IDs are not in the benchmark: {_preview(unknown_selected)}")
    unexpected = actual_set - expected_set
    if unexpected:
        raise PreparationError(f"Unexpected prediction videos: {_preview(unexpected)}")

    missing = selected_set - actual_set
    if missing and not allow_partial:
        raise PreparationError(f"Missing prediction videos: {_preview(missing)}")

    chosen = [sample_id for sample_id in selected if sample_id in actual_set]
    if not chosen:
        raise PreparationError("No prediction videos match the selected benchmark samples.")
    return [videos[sample_id] for sample_id in chosen]


def inspect_video(path: str | Path) -> VideoInfo:
    """Decode a video once to verify it is readable and report its native properties."""
    video = Path(path)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise PreparationError(f"Could not open {video}")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
    resolution: tuple[int, int] | None = None
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise PreparationError(f"Could not decode frame {frame_count} from {video}")
            current_resolution = (int(frame.shape[1]), int(frame.shape[0]))
            if resolution is None:
                resolution = current_resolution
            elif current_resolution != resolution:
                raise PreparationError(f"Video resolution changes within {video}: {resolution} to {current_resolution}")
            frame_count += 1
    finally:
        capture.release()

    if frame_count == 0 or resolution is None:
        raise PreparationError(f"No frames decoded from {video}")
    return VideoInfo(video.stem, video, sha256_file(video), fps, resolution, frame_count)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frames(video: str | Path, output_dir: str | Path, fps: float) -> int:
    """Decode, resize, and atomically replace one evaluator frame directory."""
    source = Path(video)
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        shutil.rmtree(staged)
        raise PreparationError(f"Could not open {source}")

    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if image.size != GT_RESOLUTION:
                image = image.resize(GT_RESOLUTION, Image.Resampling.LANCZOS)
            image.save(staged / f"{count:05d}.jpg", quality=95, subsampling=0)
            count += 1
        _validate_coverage(source, count, fps)
        _remove_path(destination)
        os.replace(staged, destination)
    except BaseException:
        _remove_path(staged)
        raise
    finally:
        capture.release()
    return count


def invalidate_sample(paths: OutputPaths, sample_id: str, include_frames: bool = True) -> None:
    """Remove evaluator artifacts that depend on one prediction video."""
    artifacts = [
        paths.mask(sample_id).parent,
        paths.tracks(sample_id).parent,
        paths.depth(sample_id).parent,
        paths.per_sample_metrics(sample_id),
        paths.per_sample_per_frame(sample_id),
    ]
    if include_frames:
        artifacts.insert(0, paths.generated_dir(sample_id))
    for artifact in artifacts:
        _remove_path(artifact)
    paths.results.unlink(missing_ok=True)


def validate_videos(
    input_path: str | Path,
    *,
    fps: float | None = None,
    expected_sample_ids: Collection[str] | None = None,
    selected_sample_ids: Collection[str] | None = None,
    allow_partial: bool = False,
) -> tuple[list[VideoInfo], float]:
    """Validate a prediction set and return its videos and evaluation frame rate."""
    videos = discover_videos(input_path)
    chosen = select_videos(videos, expected_sample_ids, selected_sample_ids, allow_partial)
    infos = [inspect_video(video) for video in tqdm(chosen, desc="Validating videos")]
    evaluation_fps = _resolve_fps(infos, fps)
    for info in infos:
        _validate_coverage(info.path, info.frame_count, evaluation_fps)
    return infos, evaluation_fps


def prepare_videos(
    input_path: str | Path,
    *,
    model: str,
    output_dir: str | Path = "outputs",
    fps: float | None = None,
    force: bool = False,
    expected_sample_ids: Collection[str] | None = None,
    selected_sample_ids: Collection[str] | None = None,
    allow_partial: bool = False,
) -> PreparationResult:
    """Validate and prepare a directory of `<sample_id>.<video>` predictions."""
    infos, evaluation_fps = validate_videos(
        input_path,
        fps=fps,
        expected_sample_ids=expected_sample_ids,
        selected_sample_ids=selected_sample_ids,
        allow_partial=allow_partial,
    )

    paths = OutputPaths(model_output_dir(output_dir, model))
    existing = _read_generation_metadata(paths.generation_metadata)
    old_records = {
        record["sample_id"]: record
        for record in existing.get("videos", [])
        if isinstance(record, dict) and isinstance(record.get("sample_id"), str)
    }
    _validate_metadata_merge(existing, model, evaluation_fps, {info.sample_id for info in infos})

    records: list[dict] = []
    for info in tqdm(infos, desc="Preparing videos"):
        old = old_records.get(info.sample_id, {})
        old_hash = old.get("source_sha256", old.get("sha256"))
        frame_dir = paths.generated_dir(info.sample_id)
        existing_frames = len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0
        replace_frames = force or old_hash != info.sha256 or existing_frames != info.frame_count

        if replace_frames:
            frame_count = extract_frames(info.path, frame_dir, evaluation_fps)
            invalidate_sample(paths, info.sample_id, include_frames=False)
        else:
            frame_count = existing_frames

        records.append(
            {
                "sample_id": info.sample_id,
                "source": str(info.path.resolve()),
                "source_sha256": info.sha256,
                "frames": frame_count,
                "native_fps": info.fps,
                "native_resolution": list(info.resolution),
                "duration_seconds": (frame_count - 1) / evaluation_fps,
            }
        )

    metadata = _merged_generation_metadata(existing, model, evaluation_fps, records)
    write_json(paths.generation_metadata, metadata)
    return PreparationResult(paths.output_dir, evaluation_fps, tuple(records))


def _resolve_fps(infos: list[VideoInfo], override: float | None) -> float:
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise PreparationError("--fps must be greater than zero.")
        return float(override)
    if any(info.fps is None for info in infos):
        raise PreparationError("Could not read every input frame rate; pass --fps explicitly.")
    known = [info.fps for info in infos if info.fps is not None]
    if max(known) - min(known) > FPS_TOLERANCE:
        raise PreparationError("Input videos have different frame rates; pass --fps if this is intentional.")
    return known[0]


def _validate_coverage(path: Path, frame_count: int, fps: float) -> None:
    last_timestamp = (frame_count - 1) / fps
    if last_timestamp + 1e-6 < DURATION_SECONDS:
        raise PreparationError(
            f"{path} has {frame_count} frames at {fps:g} fps, covering only {last_timestamp:.3f}s; "
            f"RigidBench requires frames from t=0 through t={DURATION_SECONDS:g}s."
        )


def _read_generation_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"Could not read existing generation metadata at {path}: {error}") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("videos", []), list):
        raise PreparationError(f"Existing generation metadata has an invalid format: {path}")
    return metadata


def _validate_metadata_merge(existing: dict, model: str, fps: float, replacing: set[str]) -> None:
    if existing.get("model") not in {None, model}:
        raise PreparationError(f"Existing generation metadata belongs to model {existing['model']!r}, not {model!r}.")
    retained = {
        record.get("sample_id")
        for record in existing.get("videos", [])
        if isinstance(record, dict) and record.get("sample_id") not in replacing
    }
    old_fps = existing.get("fps")
    if retained and old_fps is not None and abs(float(old_fps) - fps) > FPS_TOLERANCE:
        raise PreparationError(
            f"Existing prepared videos use {float(old_fps):g} fps; use the same --fps or a new model name."
        )


def _merged_generation_metadata(existing: dict, model: str, fps: float, records: list[dict]) -> dict:
    by_id = {
        record["sample_id"]: record
        for record in existing.get("videos", [])
        if isinstance(record, dict) and isinstance(record.get("sample_id"), str)
    }
    by_id.update({record["sample_id"]: record for record in records})
    return {
        **existing,
        "protocol": PROTOCOL,
        "model": model,
        "fps": fps,
        "resolution": list(GT_RESOLUTION),
        "videos": [by_id[sample_id] for sample_id in sorted(by_id)],
    }


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _unique_ids(values: Collection[str], name: str) -> list[str]:
    ids = list(values)
    if not ids:
        raise PreparationError(f"The {name} sample ID list is empty.")
    if len(set(ids)) != len(ids):
        raise PreparationError(f"The {name} sample ID list contains duplicates.")
    return ids


def _preview(values: Collection[str], limit: int = 10) -> str:
    ordered = sorted(values)
    suffix = " ..." if len(ordered) > limit else ""
    return ", ".join(ordered[:limit]) + suffix
