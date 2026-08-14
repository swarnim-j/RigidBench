from __future__ import annotations

import subprocess
from pathlib import Path

import bpy


def setup_render(spec, seed: int) -> None:
    """Configure Cycles engine, sample count, denoiser, and GPU device."""
    scene = bpy.context.scene
    r = scene.render
    r.resolution_x, r.resolution_y = spec.resolution
    r.resolution_percentage = 100
    r.fps = spec.fps
    r.image_settings.file_format = spec.output_format
    r.use_motion_blur = False

    if spec.engine != "CYCLES":
        return

    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.seed = seed
    c.samples = spec.samples
    c.use_denoising = True
    c.denoiser = "OPTIX"
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.05
    c.max_bounces = 6
    c.diffuse_bounces = c.glossy_bounces = c.transmission_bounces = 3
    c.volume_bounces = 0
    c.use_fast_gi = False
    r.use_persistent_data = True

    if spec.use_gpu:
        c.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        types = [t[0] for t in prefs.get_device_types(bpy.context)]
        for t in ["OPTIX", "CUDA", "METAL"]:
            if t in types:
                prefs.compute_device_type = t
                break
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type != "CPU"


def render_frames(output: Path, frames: int) -> None:
    """Render each frame to output/frames/00000.png … (frames-1).png."""
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    scene = bpy.context.scene
    for f in range(frames):
        scene.frame_set(f)
        scene.render.filepath = str(frames_dir / f"{f:05d}.png")
        bpy.ops.render.render(write_still=True)


def encode_video(output: Path, fps: int) -> None:
    """ffmpeg-encode output/frames/*.png into output/video.mp4."""
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(output / "frames" / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output / "video.mp4"),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
