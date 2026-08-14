from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np


def setup_passes(actors: dict, output: Path) -> None:
    """Wire IndexOB and Z passes through the compositor into per-frame EXR sequences."""
    vl = bpy.context.view_layer
    vl.use_pass_object_index = True
    vl.use_pass_z = True

    for obj in bpy.data.objects:
        obj.pass_index = 0
    for i, obj in enumerate(actors.values(), 1):
        obj.pass_index = i

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

    for name, pass_name in [("masks", "IndexOB_"), ("depth", "Depth_")]:
        d = output / name
        d.mkdir(exist_ok=True)
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.base_path = str(d)
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"
        fo.file_slots.clear()
        fo.file_slots.new(pass_name)
        tree.links.new(rl.outputs[pass_name.rstrip("_")], fo.inputs[pass_name])


def process_masks(actors: dict, output: Path, frames: int) -> None:
    """Read the IndexOB EXR sequence into one (T, N, H, W) bool array at masks.npz."""
    r = bpy.context.scene.render
    h, w = r.resolution_y, r.resolution_x
    masks = np.zeros((frames, len(actors), h, w), dtype=bool)
    for fi in range(frames):
        idx_map = _load_exr_channel(output / "masks" / f"IndexOB_{fi:04d}.exr", h, w)
        if idx_map is None:
            continue
        for i in range(len(actors)):
            masks[fi, i] = np.abs(idx_map - (i + 1)) < 0.5
    np.savez_compressed(
        output / "masks.npz",
        masks=masks,
        object_names=list(actors.keys()),
    )


def process_depth(output: Path, frames: int) -> None:
    """Read the Z-pass EXR sequence into one (T, H, W) float array at depth.npz."""
    r = bpy.context.scene.render
    h, w = r.resolution_y, r.resolution_x
    depth_all = np.zeros((frames, h, w), dtype=np.float32)
    for fi in range(frames):
        depth = _load_exr_channel(output / "depth" / f"Depth_{fi:04d}.exr", h, w)
        if depth is not None:
            depth_all[fi] = depth
    np.savez_compressed(output / "depth.npz", depth=depth_all)


def _load_exr_channel(path: Path, h: int, w: int) -> np.ndarray | None:
    """Read one channel of an OpenEXR file as a top-down (h, w) float32 array."""
    if not path.exists():
        return None
    img = bpy.data.images.load(str(path))
    px = np.array(img.pixels[:], dtype=np.float32)
    bpy.data.images.remove(img)
    if px.size == 0:
        return None
    flat = px.reshape(h, w) if px.size == h * w else px.reshape(h, w, -1)[:, :, 0]
    return np.flipud(flat)
