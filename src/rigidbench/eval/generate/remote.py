from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import replicate
import requests
from PIL import Image

from rigidbench.core.constants import GT_RESOLUTION, NEGATIVE_PROMPT, PROMPT_SUFFIX

from .base import BaseGenerator
from .models import REPLICATE_MODELS


class ReplicateGenerator(BaseGenerator):
    def generate(
        self,
        prompt: str,
        image: str | Path,
        output_dir: str | Path,
        force: bool = False,
    ) -> Path | None:
        """Submit one Replicate prediction synchronously and save its frames to output_dir."""
        output_dir = Path(output_dir)
        if not force and (output_dir / "00000.jpg").exists():
            return output_dir
        cfg = REPLICATE_MODELS[self.model]
        img_bytes = _resize_to_bytes(image, (cfg["width"], cfg["height"]))
        pred = replicate.predictions.create(
            model=cfg["model_id"],
            input=_build_inputs(prompt, img_bytes, cfg),
        )
        pred.wait()
        if pred.status != "succeeded":
            print(f"{self.model}: {pred.status} {pred.error or ''}")
            return None
        return _download_and_save_frames(_prediction_url(pred), output_dir)

    def generate_batch(self, samples: list[dict]) -> dict[str, Path | None]:
        """Submit all predictions concurrently bounded by REPLICATE_CONCURRENCY."""
        concurrency = int(os.environ.get("REPLICATE_CONCURRENCY", "16"))
        cfg = REPLICATE_MODELS[self.model]
        target = (cfg["width"], cfg["height"])

        async def _run_one(s: dict, sem: asyncio.Semaphore) -> tuple[str, Path | None]:
            out_dir = Path(s["output_dir"])
            if (out_dir / "00000.jpg").exists():
                return s["id"], out_dir
            async with sem:
                try:
                    img_bytes = _resize_to_bytes(s["image"], target)
                    inputs = _build_inputs(s["prompt"], img_bytes, cfg)
                    pred = await _create_with_retry(cfg["model_id"], inputs)
                    print(f"Submitted {s['id']}")
                    await pred.async_wait()
                    if pred.status != "succeeded":
                        print(f"Failed {s['id']}: status={pred.status} error={pred.error or '?'}")
                        return s["id"], None
                    await asyncio.to_thread(
                        _download_and_save_frames,
                        _prediction_url(pred),
                        out_dir,
                    )
                    print(f"Completed {s['id']}")
                    return s["id"], out_dir
                except Exception as e:
                    print(f"Failed {s['id']}: {type(e).__name__}: {e}")
                    return s["id"], None

        async def _runner() -> dict[str, Path | None]:
            sem = asyncio.Semaphore(concurrency)
            tasks = [asyncio.create_task(_run_one(s, sem)) for s in samples]
            return dict(await asyncio.gather(*tasks))

        return asyncio.run(_runner())


def _resize_to_bytes(image: str | Path, size: tuple[int, int]) -> bytes:
    """Resize an image to `size` and return its PNG bytes."""
    img = Image.open(image).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_inputs(prompt: str, img_bytes: bytes, cfg: dict) -> dict:
    """Assemble the model-specific Replicate input dict from prompt and base64 image."""
    b64 = base64.b64encode(img_bytes).decode()
    inputs = {
        "prompt": prompt + PROMPT_SUFFIX,
        cfg["duration_param"]: cfg["duration"],
        cfg["image_param"]: f"data:image/png;base64,{b64}",
        **cfg["extra"],
    }
    if cfg.get("negative_param"):
        inputs[cfg["negative_param"]] = NEGATIVE_PROMPT
    return inputs


async def _create_with_retry(model_id: str, inputs: dict, max_retries: int = 6):
    """Submit a Replicate prediction, retrying with exponential backoff on 429 rate limits."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return await replicate.predictions.async_create(model=model_id, input=inputs)
        except replicate.exceptions.ReplicateError as e:
            if "429" not in str(e) or attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2


def _prediction_url(pred) -> str:
    """Extract the output URL from a Replicate prediction (output may be string or FileOutput)."""
    out = pred.output[0] if isinstance(pred.output, list) else pred.output
    return out.url if hasattr(out, "url") else str(out)


def _download_and_save_frames(url: str, output_dir: Path) -> Path:
    """Stream the mp4 from Replicate, decode it, and save each frame as a JPEG at GT_RESOLUTION."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        for chunk in requests.get(url, stream=True).iter_content(8192):
            f.write(chunk)
        video_path = f.name
    try:
        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if img.size != GT_RESOLUTION:
                img = img.resize(GT_RESOLUTION, Image.LANCZOS)
            img.save(output_dir / f"{i:05d}.jpg", quality=95, subsampling=0)
            i += 1
        cap.release()
        return output_dir
    finally:
        os.unlink(video_path)
