from __future__ import annotations

import argparse
import os

from rigidbench.core.paths import OutputPaths, model_output_dir

from ..samples import filter_samples, load_samples
from .base import make_generator


def _setup_distributed() -> tuple[int, int]:
    """Read RANK and WORLD_SIZE from env, init the torch process group, return (rank, world)."""
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        if not dist.is_initialized():
            dist.init_process_group("gloo")
    return rank, world


def run_generate(
    model: str,
    data_dir: str,
    output_dir: str,
    checkpoint: str | None = None,
    split: str = "eval",
    sample_ids: list[str] | None = None,
    max_samples: int | None = None,
    force: bool = False,
) -> None:
    """Generate predictions for one model, distributed across local ranks when requested."""
    paths = OutputPaths(model_output_dir(output_dir, model, checkpoint))
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    samples = filter_samples(load_samples(data_dir, split), sample_ids, max_samples)

    rank, world = _setup_distributed()
    my_samples = samples[rank::world] if world > 1 else samples
    tag = f"[rank {rank}/{world}] " if world > 1 else ""

    pending = [
        {
            "id": s.id,
            "prompt": s.prompt,
            "image": str(s.first_frame),
            "output_dir": str(paths.generated_dir(s.id)),
        }
        for s in my_samples
        if force or not (paths.generated_dir(s.id) / "00000.jpg").exists()
    ]
    print(f"{tag}{model}: {len(pending)}/{len(my_samples)} pending ({len(samples)} total)")

    if pending:
        with make_generator(model, checkpoint) as gen:
            gen.generate_batch(pending)

    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="Generate RigidBench predictions for a registered video model.")
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", "--data_dir", required=True)
    p.add_argument("--output-dir", "--output_dir", default="outputs")
    p.add_argument("--checkpoint")
    p.add_argument("--split", default="eval", choices=["train", "eval"])
    p.add_argument("--sample-ids", "--sample_ids")
    p.add_argument("--max-samples", "--max_samples", type=int)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    run_generate(
        args.model,
        args.data_dir,
        args.output_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        sample_ids=args.sample_ids.split(",") if args.sample_ids else None,
        max_samples=args.max_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
