from __future__ import annotations

import json
from pathlib import Path

import bpy

from rigidbench.core.manifest import ManifestEntry
from rigidbench.scenes.assembly import setup_scene
from rigidbench.scenes.schema.registry import get_registry

from .cycles import encode_video, render_frames, setup_render
from .passes import process_depth, process_masks, setup_passes
from .physics import export_contacts, export_trajectories, simulate_physics


def render_sample(entry: ManifestEntry, output_dir: str) -> dict:
    """Build, simulate, render, and export GT for one manifest entry."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    scene_data = setup_scene(entry)
    actors = {name: bpy.data.objects[name] for name in scene_data["actors"] if name in bpy.data.objects}
    frames = scene_data["frames"]

    render_spec = get_registry().render()
    setup_render(render_spec, entry.seed)
    setup_passes(actors, output)

    simulate_physics(frames)
    export_trajectories(actors, frames, output)
    export_contacts(actors, frames, output)

    render_frames(output, frames)
    encode_video(output, render_spec.fps)

    process_masks(actors, output, frames)
    process_depth(output, frames)

    save_metadata(output, scene_data, render_spec)
    return scene_data


def save_metadata(output: Path, scene_data: dict, render_spec) -> None:
    """Write metadata.json (scene data + camera intrinsics/extrinsics) and prompt.txt."""
    cam = bpy.context.scene.camera
    r = bpy.context.scene.render
    fx = cam.data.lens * r.resolution_x / cam.data.sensor_width

    metadata = {
        **scene_data,
        "camera": {
            "intrinsics": {
                "fx": fx,
                "fy": fx,
                "cx": r.resolution_x / 2,
                "cy": r.resolution_y / 2,
                "width": r.resolution_x,
                "height": r.resolution_y,
            },
            "extrinsics": {
                "location": list(cam.matrix_world.translation),
                "rotation": list(cam.matrix_world.to_quaternion()),
            },
        },
        "render": {"resolution": list(render_spec.resolution), "fps": render_spec.fps},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "prompt.txt").write_text(scene_data["prompt"])
