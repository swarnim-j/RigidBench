import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rigidbench.train.backbones import BackboneAdapter, get_adapter


@dataclass
class Sample:
    sample_id: str
    path: Path
    text: str


def setup_distributed() -> tuple[int, int, str]:
    """Init NCCL when launched under torchrun, returning (rank, world, device)."""
    if "RANK" not in os.environ:
        return 0, 1, "cuda"
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"


def load_samples(data_dir: Path) -> list[Sample]:
    """Discover sample directories under data_dir that ship a non-empty prompt."""
    samples = []
    for sample_dir in sorted(data_dir.iterdir()):
        meta_path = sample_dir / "metadata.json"
        if not (sample_dir.is_dir() and meta_path.exists()):
            continue
        prompt = json.loads(meta_path.read_text()).get("prompt", "")
        if prompt:
            samples.append(Sample(sample_id=sample_dir.name, path=sample_dir, text=prompt))
    return samples


@torch.no_grad()
def encode_unique_prompts(
    adapter: BackboneAdapter,
    samples: list[Sample],
    device: str,
    rank: int,
    world_size: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Encode every unique prompt, sharded across ranks and gathered back to all."""
    unique_prompts = sorted({s.text for s in samples})
    rank_prompts = unique_prompts[rank::world_size]

    local_emb: dict[str, torch.Tensor] = {}
    pbar = tqdm(total=len(rank_prompts), desc=f"text rank{rank}") if rank == 0 else None
    for i in range(0, len(rank_prompts), batch_size):
        batch = rank_prompts[i : i + batch_size]
        local_emb.update(adapter.encode_prompts(batch, device))
        if pbar:
            pbar.update(len(batch))
    if pbar:
        pbar.close()

    if world_size == 1:
        return local_emb

    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local_emb)
    full = {}
    for d in gathered:
        full.update(d)
    return full


class FrameDataset(Dataset):
    """Yield evenly-spaced resized frames from each sample, returning errors instead of raising."""

    def __init__(self, samples: list[Sample], num_frames: int, h: int, w: int):
        self.samples = samples
        self.num_frames = num_frames
        self.h, self.w = h, w

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        try:
            frames_dir = s.path / "frames"
            files = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if not files:
                raise ValueError(f"No images in {frames_dir}")
            indices = torch.linspace(0, len(files) - 1, self.num_frames).long().tolist()
            frames = [Image.open(files[i]).convert("RGB").resize((self.w, self.h)) for i in indices]
            return {"sample": s, "frames": frames, "error": None}
        except Exception as e:
            return {"sample": s, "frames": None, "error": str(e)}


@torch.no_grad()
def encode_latents(
    adapter: BackboneAdapter,
    todo: list[Sample],
    prompt_emb: dict[str, torch.Tensor],
    out: Path,
    num_frames: int,
    h: int,
    w: int,
    vae_batch_size: int,
    workers: int,
    device: str,
    desc: str | None,
) -> tuple[list[dict], int]:
    """Save the text embedding per sample and VAE-encode frames in batches."""
    loader = DataLoader(
        FrameDataset(todo, num_frames, h, w),
        num_workers=workers,
        batch_size=None,
        collate_fn=lambda x: x,
        prefetch_factor=2,
    )
    pbar = tqdm(total=len(todo), desc=desc) if desc else None
    manifest: list[dict] = []
    buf_videos: list[torch.Tensor] = []
    buf_samples: list[Sample] = []
    errors = 0

    def flush():
        """VAE-encode the buffered batch and save each artifact to its own .pt file."""
        if not buf_videos:
            return
        stacked = torch.cat(buf_videos, dim=0).to(device)
        artifacts = adapter.encode_artifacts(stacked, device=device)
        for i, bs in enumerate(buf_samples):
            entry = {"text_emb": f"{bs.sample_id}_text.pt"}
            for name, tensors in artifacts.items():
                path = out / f"{bs.sample_id}_{name}.pt"
                torch.save(tensors[i : i + 1], path)
                entry[name] = path.name
            manifest.append(entry)
        buf_videos.clear()
        buf_samples.clear()

    for result in loader:
        s: Sample = result["sample"]
        torch.save(prompt_emb[s.text], out / f"{s.sample_id}_text.pt")
        if result["error"]:
            if pbar:
                tqdm.write(f"Skip {s.sample_id}: {result['error']}")
            errors += 1
        else:
            buf_videos.append(adapter.preprocess_frames(result["frames"]))
            buf_samples.append(s)
            if len(buf_videos) >= vae_batch_size:
                flush()
        if pbar:
            pbar.update(1)
    flush()
    if pbar:
        pbar.close()
    return manifest, errors


def combine_manifest_shards(out: Path, world_size: int) -> int:
    """Merge per-rank manifest shards into one manifest.jsonl, returning total count."""
    combined: list[dict] = []
    for r in range(world_size):
        p = out / f"manifest_{r}.jsonl"
        if p.exists():
            combined.extend(json.loads(line) for line in p.read_text().splitlines())
            p.unlink()
    with open(out / "manifest.jsonl", "w") as f:
        for e in combined:
            f.write(json.dumps(e) + "\n")
    return len(combined)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", default="wan")
    parser.add_argument("--model-id", "--model_id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--num-frames", "--num_frames", type=int, default=49)
    parser.add_argument("--size", type=int, nargs=2, default=[352, 640], metavar=("H", "W"))
    parser.add_argument("--workers", type=int, default=16, help="DataLoader workers per rank for frame I/O")
    parser.add_argument("--prompt-batch-size", "--prompt_batch_size", type=int, default=32)
    parser.add_argument("--vae-batch-size", "--vae_batch_size", type=int, default=8, help="Videos per VAE encode call")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    is_main = rank == 0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    h, w = args.size

    if is_main:
        print(f"Loading {args.backbone} encoders on {world_size} GPU(s)...")
    adapter = get_adapter(args.backbone, {"model_id": args.model_id})
    adapter.load(device=device, mode="preprocess", rank=rank, world_size=world_size)

    samples = load_samples(Path(args.data_dir))
    if is_main:
        n_unique = len({s.text for s in samples})
        print(f"{len(samples)} samples, {n_unique} unique prompts")

    prompt_emb = encode_unique_prompts(adapter, samples, device, rank, world_size, args.prompt_batch_size)
    if is_main:
        print(f"All ranks have {len(prompt_emb)} prompt embeddings")

    todo: list[Sample] = []
    cached: list[dict] = []
    for s in samples:
        txt = out / f"{s.sample_id}_text.pt"
        all_valid = txt.exists() and all(
            adapter.is_artifact_valid(name, out / f"{s.sample_id}_{name}.pt", args.num_frames)
            for name in adapter.sample_artifacts
        )
        if all_valid:
            entry = {"text_emb": txt.name}
            for name in adapter.sample_artifacts:
                entry[name] = f"{s.sample_id}_{name}.pt"
            cached.append(entry)
        else:
            todo.append(s)
    rank_todo = todo[rank::world_size]
    if is_main:
        print(f"Cached: {len(cached)}, To process: {len(todo)} ({len(rank_todo)} per rank)")

    manifest = list(cached) if is_main else []
    errors = 0

    if rank_todo:
        new_entries, errors = encode_latents(
            adapter,
            rank_todo,
            prompt_emb,
            out,
            args.num_frames,
            h,
            w,
            args.vae_batch_size,
            args.workers,
            device,
            desc=f"VAE rank{rank}" if is_main or world_size <= 4 else None,
        )
        manifest.extend(new_entries)

    with open(out / f"manifest_{rank}.jsonl", "w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")

    if world_size > 1:
        dist.barrier()

    if is_main:
        total = combine_manifest_shards(out, world_size)
        print(f"Done: {total} samples ({errors} errors)")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
