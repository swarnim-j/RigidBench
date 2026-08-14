from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rigidbench.core.constants import GT_FPS, GT_RESOLUTION, NEGATIVE_PROMPT, PROMPT_SUFFIX
from rigidbench.eval.samples import load_samples

PROTOCOL = "rigidbench-v1"
DURATION_SECONDS = 2.0
REFERENCE_FRAMES = int(DURATION_SECONDS * GT_FPS) + 1
EVALUATION_SIZE = 100


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    id: str
    image: Path
    scene_prompt: str
    prompt: str
    negative_prompt: str
    duration_seconds: float
    reference_fps: int
    reference_frames: int
    reference_resolution: tuple[int, int]
    task: str
    protocol: str


@dataclass(frozen=True, slots=True)
class Benchmark:
    examples: tuple[BenchmarkExample, ...]
    protocol: str = PROTOCOL

    @classmethod
    def load(cls, data_dir: str | Path, split: str = "eval") -> Benchmark:
        root = Path(data_dir)
        if not (root / split).is_dir():
            from rigidbench.data import find_data_root

            root = find_data_root(root) or root
        samples = load_samples(root, split)
        if not samples:
            raise FileNotFoundError(f"No RigidBench {split} samples found under {root}")
        examples = tuple(
            BenchmarkExample(
                id=sample.id,
                image=sample.first_frame,
                scene_prompt=sample.prompt,
                prompt=sample.prompt + PROMPT_SUFFIX,
                negative_prompt=NEGATIVE_PROMPT,
                duration_seconds=DURATION_SECONDS,
                reference_fps=GT_FPS,
                reference_frames=REFERENCE_FRAMES,
                reference_resolution=GT_RESOLUTION,
                task=sample.task_type,
                protocol=PROTOCOL,
            )
            for sample in samples
        )
        return cls(examples)

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BenchmarkExample:
        return self.examples[index]


def export_inputs_jsonl(data_dir: str | Path, output: str | Path) -> Path:
    """Atomically export the exact generation inputs for the evaluation set."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark.load(data_dir)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            for example in benchmark:
                try:
                    image = example.image.resolve().relative_to(destination.parent.resolve()).as_posix()
                except ValueError:
                    image = str(example.image.resolve())
                row = {
                    "id": example.id,
                    "image": image,
                    "scene_prompt": example.scene_prompt,
                    "prompt": example.prompt,
                    "negative_prompt": example.negative_prompt,
                    "duration_seconds": example.duration_seconds,
                    "reference_fps": example.reference_fps,
                    "reference_frames": example.reference_frames,
                    "reference_resolution": example.reference_resolution,
                    "task": example.task,
                    "protocol": example.protocol,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the exact inputs supplied to video models.")
    parser.add_argument("--data", "--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from rigidbench.data import find_data_root

    data_root = find_data_root(args.data)
    if data_root is None:
        raise SystemExit(f"No RigidBench evaluation set found under {args.data}. Run 'rigidbench download' first.")
    output = args.output or data_root / "inputs.jsonl"
    export_inputs_jsonl(data_root, output)
    print(f"Wrote {len(Benchmark.load(data_root))} model inputs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
