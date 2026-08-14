from __future__ import annotations

import math
import random

import bpy
from mathutils import Vector

from ..schema.types import CameraSpec
from . import angles as _angles
from .framing import camera_position, framing_bbox
from .visibility import coverage_score

_MAX_ANGLE_RETRIES = 8
_MIN_COVERAGE = 0.50
_AZ_STEP_DEG = 15.0


def resolve_task_and_camera(
    actors: dict,
    surface_obj,
    surface_spec,
    cam: CameraSpec,
    task_cfg: dict,
    camera_rotation: float | None,
    fit_world: list[tuple[float, float]],
    rng: random.Random,
) -> tuple[float, float]:
    """Pick a task rotation that fits the surface and a camera azimuth with good visibility."""
    best: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    for _ in range(_MAX_ANGLE_RETRIES):
        task_angle = _angles.sample_from_intervals(fit_world, rng)
        _align_to_surface(actors, surface_spec.center, task_angle)
        if camera_rotation is not None:
            return task_angle, camera_rotation % 360
        candidates, scores = _score_candidate_azimuths(
            actors,
            surface_obj,
            surface_spec,
            cam,
            task_cfg,
            task_angle,
        )
        passing = [az for az, s in zip(candidates, scores) if s >= _MIN_COVERAGE]
        if passing:
            return task_angle, rng.choice(passing)
        if scores:
            top = max(range(len(scores)), key=scores.__getitem__)
            if scores[top] > best[0]:
                best = (scores[top], task_angle, candidates[top])
        _undo_alignment(actors, surface_spec.center, task_angle)
    _align_to_surface(actors, surface_spec.center, best[1])
    return best[1], best[2]


def place_camera(actors, surface_obj, surface_spec, cam: CameraSpec, az_deg: float) -> None:
    """Position the Blender camera at the resolved azimuth, framing the actors and surface."""
    camera = bpy.data.objects.get("Camera")
    if not camera:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
    bpy.context.view_layer.update()
    framing = framing_bbox(actors, surface_obj, surface_spec, cam)
    camera.location = camera_position(framing.center, framing.distance, cam.elevation, az_deg)
    camera.rotation_euler = (framing.center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = cam.lens
    camera.data.dof.use_dof = False
    bpy.context.scene.camera = camera


def _score_candidate_azimuths(actors, surface_obj, surface_spec, cam, task_cfg, task_angle):
    candidates = _candidate_azimuths(task_cfg["camera_range"], task_angle)
    framing = framing_bbox(actors, surface_obj, surface_spec, cam)
    target_objs = [surface_obj] + list(actors.values())
    scores = [
        coverage_score(
            camera_position(framing.center, framing.distance, cam.elevation, az),
            target_objs,
        )
        for az in candidates
    ]
    return candidates, scores


def _candidate_azimuths(camera_range, task_angle: float) -> list[float]:
    azs: list[float] = []
    for lo, hi in _angles.shift(camera_range, task_angle):
        az = lo
        while az <= hi:
            azs.append(az % 360)
            az += _AZ_STEP_DEG
    return azs


def _align_to_surface(actors: dict, surface_center, task_angle: float) -> None:
    bpy.context.view_layer.update()
    coords = [obj.matrix_world @ Vector(v) for obj in actors.values() for v in obj.bound_box]
    bbox_cx = (min(c.x for c in coords) + max(c.x for c in coords)) / 2
    bbox_cy = (min(c.y for c in coords) + max(c.y for c in coords)) / 2
    shift_x = surface_center[0] - bbox_cx
    shift_y = surface_center[1] - bbox_cy
    for obj in actors.values():
        obj.location.x += shift_x
        obj.location.y += shift_y
    bpy.context.view_layer.update()
    if task_angle == 0:
        return
    rad = math.radians(task_angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = surface_center[0], surface_center[1]
    for obj in actors.values():
        dx = obj.location.x - cx
        dy = obj.location.y - cy
        obj.location.x = cx + dx * cos_a - dy * sin_a
        obj.location.y = cy + dx * sin_a + dy * cos_a
        obj.rotation_euler.z += rad
    bpy.context.view_layer.update()


def _undo_alignment(actors: dict, surface_center, task_angle: float) -> None:
    if task_angle == 0:
        return
    rad = math.radians(-task_angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = surface_center[0], surface_center[1]
    for obj in actors.values():
        dx = obj.location.x - cx
        dy = obj.location.y - cy
        obj.location.x = cx + dx * cos_a - dy * sin_a
        obj.location.y = cy + dx * sin_a + dy * cos_a
        obj.rotation_euler.z += rad
    bpy.context.view_layer.update()
