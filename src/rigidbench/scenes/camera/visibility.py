from __future__ import annotations

import bpy
from mathutils import Vector

_SAMPLES_PER_TARGET = 16


def coverage_score(camera_pos, target_objs: list) -> float:
    targets = set(target_objs)
    sample_pts = _sample_target_points(target_objs)
    if not sample_pts:
        return 0.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam = Vector(camera_pos)
    hits = 0
    for pt in sample_pts:
        direction = pt - cam
        dist = direction.length
        if dist < 1e-9:
            continue
        direction.normalize()
        hit, _, _, _, hit_obj, _ = bpy.context.scene.ray_cast(depsgraph, cam, direction, distance=dist + 1e-3)
        if hit and hit_obj in targets:
            hits += 1
    return hits / len(sample_pts)


def _sample_target_points(target_objs: list) -> list:
    points: list = []
    for obj in target_objs:
        mesh = getattr(obj, "data", None)
        polys = getattr(mesh, "polygons", None) if mesh else None
        if polys and len(polys):
            mw = obj.matrix_world
            step = max(1, len(polys) // _SAMPLES_PER_TARGET)
            for i in range(0, len(polys), step):
                points.append(mw @ Vector(polys[i].center))
        else:
            points.append(obj.matrix_world.translation.copy())
    return points
