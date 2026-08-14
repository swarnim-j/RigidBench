from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Blender launches this file directly, so add the repository's src/ directory.
sys.path.insert(0, str(Path(__file__).parents[2]))

from rigidbench.core.io import write_json
from rigidbench.scenes.manifest import build_train_manifest

CONFIG_DIR = Path(__file__).parent / "configs"


def render_loop(samples: list[dict], output_dir: Path) -> None:
    """Render each manifest entry that hasn't already produced metadata.json."""
    from rigidbench.core.manifest import ManifestEntry
    from rigidbench.scenes.render.sample import render_sample

    failed: list[str] = []
    for i, s in enumerate(samples):
        entry = ManifestEntry.from_dict(s)
        sample_dir = output_dir / entry.sample_id
        if (sample_dir / "metadata.json").exists():
            print(f"[{i + 1}/{len(samples)}] skip {entry.sample_id} (exists)")
            continue
        print(f"[{i + 1}/{len(samples)}] render {entry.sample_id}: {entry.scene_id}/{entry.task_type}")
        try:
            render_sample(entry, str(sample_dir))
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(entry.sample_id)
    if failed:
        raise RuntimeError(f"Rendering failed for {len(failed)} sample(s): {', '.join(failed[:10])}")


def main():
    parser = argparse.ArgumentParser(description="Render RigidBench simulator samples in Blender.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", help="Use this manifest path instead of auto-generating")
    parser.add_argument("--num-samples", type=int, help="Generate a new train manifest of this size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument("--sample-ids", help="Comma-separated sample IDs to render")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    if "bpy" not in sys.modules and not args.manifest_only:
        command = ["blender", "--background", "--python", str(Path(__file__)), "--", *sys.argv[1:]]
        env = os.environ.copy()
        python_path = [path for path in sys.path if path]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path)
        try:
            subprocess.run(command, check=True, env=env)
        except FileNotFoundError as error:
            raise SystemExit("Blender was not found on PATH. Install Blender 4.5 before rendering.") from error
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest_path = Path(args.manifest)
        samples = json.loads(manifest_path.read_text())
        print(f"loaded {len(samples)} samples from {manifest_path}")
    elif args.split == "train" and args.num_samples is not None:
        samples = build_train_manifest(args.num_samples, args.seed)
        print(f"generated {len(samples)} train samples with seed {args.seed}")
    elif args.split == "train":
        manifest_path = CONFIG_DIR / "manifest_train.json"
        samples = json.loads(manifest_path.read_text())
        print(f"loaded {len(samples)} submitted train samples from {manifest_path}")
    else:
        manifest_path = CONFIG_DIR / "manifest_eval.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"eval manifest not found at {manifest_path}. "
                "Eval samples are hand-curated; author this file before running."
            )
        samples = json.loads(manifest_path.read_text())
        print(f"loaded {len(samples)} eval samples from {manifest_path}")

    manifest_output = output_dir / f"manifest_{args.split}.json"
    write_json(manifest_output, samples)
    print(f"manifest: {manifest_output}")
    if args.manifest_only:
        return

    if args.sample_ids:
        id_set = {sample_id.strip() for sample_id in args.sample_ids.split(",")}
        samples = [s for s in samples if s["sample_id"] in id_set]

    out_dir = output_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(samples)} samples to {out_dir}")
    render_loop(samples, out_dir)


if __name__ == "__main__":
    if "--" in sys.argv:
        sys.argv = [sys.argv[0], *sys.argv[sys.argv.index("--") + 1 :]]
    main()
