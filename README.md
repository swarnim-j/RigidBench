# RigidBench

RigidBench is a simulator-grounded benchmark for evaluating rigid-body physics in image-to-video models. It compares each generated video with a reference rollout from the same first frame and prompt, using ten measurements of motion, geometry, identity, background stability, and appearance across 100 examples and five tasks.

- [Evaluation set](https://doi.org/10.5281/zenodo.21649156)

## Evaluate a model

RigidBench accepts ordinary video files, so generation can happen in any framework or hosted API.

### Install

The full evaluator requires [uv](https://docs.astral.sh/uv/), Python 3.11, Linux, and an NVIDIA CUDA GPU.

```bash
uv sync --frozen --extra eval
```

Video Depth Anything is not packaged for Python, so install its official implementation once:

```bash
git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything vendor/Video-Depth-Anything
export PYTHONPATH="$PWD/vendor/Video-Depth-Anything:$PYTHONPATH"
```

Model weights are downloaded on first use.
Optional provider credentials are listed in `.env.example`; export only those needed for the command you run.

### Download the benchmark

```bash
uv run rigidbench download
```

This downloads and verifies the 18.2 GB evaluation set from Zenodo under `data/`. Zenodo stores the archive as numbered parts, which the command joins automatically. It also writes `inputs.jsonl` from the included conditioning frames and prompts. The archive is used as published; no simulator data is regenerated.

Allow roughly 40 GB of free disk space while downloading and extracting the archive. The compressed download is removed after a successful extraction unless `--keep-archive` is passed.

### Generate one video per example

Read `data/inputs.jsonl`, run your model on each supplied image and prompt, and save:

```text
predictions/my-model/
├── 00000.mp4
├── 00001.mp4
├── ...
└── 00099.mp4
```

The important requirements are:

| Requirement | Value |
| --- | --- |
| Input | supplied first frame and complete `prompt` from `inputs.jsonl` |
| Output | one video named `<id>.mp4` for every example |
| Time span | frames covering `t=0` through `t=2.0 s` |
| First frame | the conditioning frame, with the same field of view |
| Resolution and FPS | may be model-native; RigidBench records and aligns them |

Use `negative_prompt` when the model supports one. Longer videos are accepted, but only the benchmark's first two seconds are scored.

The same inputs are available from Python:

```python
from pathlib import Path
from rigidbench import Benchmark

output = Path("predictions/my-model")
output.mkdir(parents=True, exist_ok=True)

for example in Benchmark.load("data"):
    destination = output / f"{example.id}.mp4"
    if destination.exists():
        continue

    generate_video(
        image=example.image,
        prompt=example.prompt,
        negative_prompt=example.negative_prompt,
        output=destination,
    )
```

Here `generate_video` is your own local or API-backed generation function. No RigidBench model adapter is required.

### Run the evaluation

An optional CPU-only preflight checks IDs, decoding, FPS, and duration:

```bash
uv run rigidbench validate predictions/my-model
```

Then run the evaluator:

```bash
uv run rigidbench evaluate predictions/my-model
```

Preparation, validation, caching, and aggregation happen automatically. Results and resumable intermediate artifacts are written under `runs/my-model/`:

```text
runs/my-model/
├── run.json
├── generation.json
├── results.json
├── metrics/
├── masks/
├── tracks/
└── depth/
```

If an input video changes, its dependent cached artifacts are recomputed. A full result is marked official only when all 100 examples and all ten measurements are present.

For a quick non-official check, evaluate one example:

```bash
uv run rigidbench evaluate predictions/my-model --max-samples 1
```

Use `--data` or `--output` to override the default `data/` and `runs/` locations.

## Measurements

RigidBench does not combine its measurements into one score.

| Group | Measurement | Direction | What it measures |
| --- | --- | --- | --- |
| Motion and geometry | IoU | higher | overlap between reference and propagated actor masks |
| Motion and geometry | L2 | lower | normalized distance between actor-mask centroids |
| Motion and geometry | Chamfer | lower | bidirectional distance between actor masks |
| Motion and geometry | ATE | lower | 2D point-trajectory error |
| Motion and geometry | ATE-3D | lower | reconstructed 3D actor-trajectory error |
| Motion and geometry | SI-MSE | lower | scale-invariant depth error |
| Appearance and stability | SSIM | higher | full-frame structural similarity |
| Appearance and stability | LPIPS | lower | learned perceptual distance |
| Appearance and stability | IdDrift | lower | change in actor identity along its trajectory |
| Appearance and stability | BGDrift | lower | non-rigid motion in the background |

## Reproducing paper runs

Adapters for models evaluated in the paper remain available. For example:

```bash
uv sync --frozen --extra eval --extra wan
uv run rigidbench generate --model wan-2.2 --data-dir data --output-dir outputs
uv run rigidbench evaluate --model wan-2.2 --data-dir data --output-dir outputs
```

## Research workflows

These commands are not needed to evaluate a custom model.

### Render simulator samples

Rendering requires Blender 4.5.9 on `PATH`, ffmpeg, and the configured BlenderKit assets. The downloader uses your own BlenderKit account, and some assets require a paid plan.

```bash
uv sync --frozen --extra scenes
export BLENDERKIT_API_KEY=...
uv run rigidbench assets --all
uv run rigidbench render --output rendered --num-samples 100 --seed 42
```

Omit `--num-samples` to use the submitted 5,000-example training manifest. Use `--split eval` to render the fixed 100-example evaluation manifest. Every run writes the manifest it used beside the rendered split.

BlenderKit no longer lists three legacy assets used by the submitted manifests: `hallway_interior`, `apple`, and `light_wood`. Exact re-rendering of those samples requires a previously downloaded copy. This does not affect benchmark evaluation because the rendered references are included in the evaluation set.

### Fine-tune Wan

The training code consumes precomputed text embeddings and video latents:

```bash
uv sync --frozen --extra train
uv run rigidbench preprocess --data-dir rendered/train --output embeddings
uv run rigidbench train \
  --config src/rigidbench/train/configs/train_lora.yaml \
  --data embeddings/manifest.jsonl \
  --output checkpoints/lora
```

Use `train.yaml` for full DiT fine-tuning. The `probe` command contains the extraction, probe-fitting, INLP, and intervention stages used in the paper; install both the `train` and `probe` extras and run `rigidbench probe --help` for their arguments.

## Licenses

The code is released under the MIT License, and the RigidBench evaluation set is CC BY 4.0. Some rendered scenes and objects use [BlenderKit assets](https://www.blenderkit.com/docs/licenses/) under Royalty Free or CC0 licenses. The original 3D assets are not redistributed and must be obtained separately from BlenderKit under the applicable access plan.

External models keep their own licenses. In particular, most of CoTracker and the Video Depth Anything Large checkpoint used by the evaluator are licensed for non-commercial use.
