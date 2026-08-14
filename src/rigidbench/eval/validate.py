from __future__ import annotations

import argparse
from pathlib import Path

from rigidbench import Benchmark
from rigidbench.data import find_data_root

from .prepare import PreparationError, validate_videos


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated videos without loading evaluation models.")
    parser.add_argument("predictions", type=Path, help="Directory containing <sample_id>.mp4 videos")
    parser.add_argument("--data", "--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fps", type=float, help="Override the detected frame rate")
    args = parser.parse_args()

    data_root = find_data_root(args.data)
    if data_root is None:
        raise SystemExit(f"No RigidBench evaluation set found under {args.data}. Run 'rigidbench download' first.")
    expected = [example.id for example in Benchmark.load(data_root)]
    try:
        videos, fps = validate_videos(args.predictions, fps=args.fps, expected_sample_ids=expected)
    except PreparationError as error:
        raise SystemExit(str(error)) from error

    print(f"Valid prediction set: {len(videos)} videos at {fps:g} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
