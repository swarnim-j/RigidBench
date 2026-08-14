from __future__ import annotations

import json
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def simulate_physics(frames: int) -> None:
    """Step the rigid-body simulator through every frame to populate Bullet's point cache."""
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, frames - 1
    if scene.rigidbody_world:
        scene.rigidbody_world.point_cache.frame_start = 0
        scene.rigidbody_world.point_cache.frame_end = frames + 30
    for f in range(frames):
        scene.frame_set(f)
    scene.frame_set(0)


def export_trajectories(actors: dict, frames: int, output: Path) -> None:
    """Per-frame position, rotation, linear and angular velocity for each actor."""
    # Quaternions written (w, x, y, z), eval reorders to (x, y, z, w).
    data: dict[str, np.ndarray] = {}
    for name, obj in actors.items():
        pos, rot, lin_vel, ang_vel = [], [], [], []
        prev_pos, prev_rot = None, None

        for f in range(frames):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()

            curr_pos = Vector(obj.matrix_world.translation)
            curr_rot = obj.matrix_world.to_quaternion()

            pos.append(list(curr_pos))
            rot.append(list(curr_rot))

            if prev_pos is not None:
                lin_vel.append(list(curr_pos - prev_pos))
                dq = curr_rot @ prev_rot.inverted()
                ang_vel.append(list(dq.to_euler()))
            else:
                lin_vel.append([0.0, 0.0, 0.0])
                ang_vel.append([0.0, 0.0, 0.0])

            prev_pos, prev_rot = curr_pos.copy(), curr_rot.copy()

        data[f"{name}_positions"] = np.array(pos, dtype=np.float32)
        data[f"{name}_rotations"] = np.array(rot, dtype=np.float32)
        data[f"{name}_linear_velocity"] = np.array(lin_vel, dtype=np.float32)
        data[f"{name}_angular_velocity"] = np.array(ang_vel, dtype=np.float32)

    np.savez_compressed(output / "trajectories.npz", **data)


def export_contacts(actors: dict, frames: int, output: Path) -> None:
    """Per-frame BVH-overlap contacts between actor pairs and passive scene geometry."""
    actor_list = list(actors.items())
    scene_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.rigid_body and obj.rigid_body.type == "PASSIVE" and obj.name not in actors
    ]

    contacts: list[dict] = []
    for f in range(frames):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        actor_bvhs = {name: _build_bvh(obj, depsgraph) for name, obj in actor_list}

        for i, (name_a, _) in enumerate(actor_list):
            bvh_a, bm_a, _ = actor_bvhs[name_a]
            for name_b, _ in actor_list[i + 1 :]:
                bvh_b, bm_b, _ = actor_bvhs[name_b]
                hit = _extract_contacts(f, bvh_a, bm_a, bvh_b, bm_b, name_a, name_b)
                if hit:
                    contacts.append(hit)

            for scene_obj in scene_objects:
                bvh_s, bm_s, eval_s = _build_bvh(scene_obj, depsgraph)
                hit = _extract_contacts(f, bvh_a, bm_a, bvh_s, bm_s, name_a, scene_obj.name)
                if hit:
                    contacts.append(hit)
                bm_s.free()
                eval_s.to_mesh_clear()

        for _, (_, bm, obj_eval) in actor_bvhs.items():
            bm.free()
            obj_eval.to_mesh_clear()

    (output / "contacts.json").write_text(json.dumps(contacts))


def _build_bvh(obj, depsgraph):
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)
    bm.faces.ensure_lookup_table()
    return BVHTree.FromBMesh(bm), bm, obj_eval


def _extract_contacts(frame, bvh_a, bm_a, bvh_b, bm_b, name_a, name_b, limit=5):
    overlap = bvh_a.overlap(bvh_b)
    if not overlap:
        return None
    pts: list[dict] = []
    for idx_a, idx_b in overlap[:limit]:
        if idx_a >= len(bm_a.faces) or idx_b >= len(bm_b.faces):
            continue
        fa, fb = bm_a.faces[idx_a], bm_b.faces[idx_b]
        pt = (fa.calc_center_median() + fb.calc_center_median()) / 2
        pts.append({"point": list(pt), "normal": list(fb.normal)})
    if not pts:
        return None
    return {"frame": frame, "obj_a": name_a, "obj_b": name_b, "contacts": pts}
