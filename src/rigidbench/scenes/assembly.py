from __future__ import annotations

import random

import bpy
import numpy as np
from mathutils import Vector

from rigidbench.core.manifest import ManifestEntry

from .builder import PASSIVE_GEOMETRY_PHYSICS, SceneBuilder, attach_rigid_body, view3d_context
from .camera import angles as _angles
from .camera.resolve import place_camera, resolve_task_and_camera
from .schema.registry import get_registry
from .schema.types import RenderSpec, SceneSpec, SurfaceSpec
from .task.execute import execute_task, load_task_config


def setup_scene(entry: ManifestEntry, camera_rotation: float | None = None) -> dict:
    """Open the .blend, place actors per the task, resolve camera angles, attach physics."""
    rng = random.Random(entry.seed)
    random.seed(entry.seed)
    np.random.seed(entry.seed % (2**32))

    registry = get_registry()
    scene_spec = registry.scene(entry.scene_id)
    render_spec = registry.render()
    task_cfg = load_task_config(entry.task_type)
    if entry.allowed_objects is not None:
        for placement_spec in task_cfg["placements"]:
            placement_spec["_allowed_objects"] = list(entry.allowed_objects)
    surface_spec = _resolve_surface(scene_spec, entry.surface_name)

    _open_blend(scene_spec, render_spec)
    _hide_clutter(surface_spec)
    surface_obj = bpy.data.objects.get(entry.surface_name)
    if surface_obj is None:
        raise ValueError(f"surface {entry.surface_name!r} not present in .blend")

    builder = SceneBuilder(surface_obj, defer_physics=True)
    actors = execute_task(task_cfg, surface_obj, builder)
    footprint = _measure_footprint(actors)
    fit_world = _angles.shift(
        _angles.fit_rotations(*footprint, *surface_spec.size),
        surface_spec.long_axis_deg,
    )
    if not fit_world:
        raise ValueError(
            f"task footprint {footprint} does not fit on surface "
            f"{surface_spec.name!r} (size {surface_spec.size}) at any rotation"
        )

    task_angle, camera_azimuth = resolve_task_and_camera(
        actors,
        surface_obj,
        surface_spec,
        render_spec.camera,
        task_cfg,
        camera_rotation,
        fit_world,
        rng,
    )

    builder.apply_deferred_physics()
    _apply_surface_physics(surface_obj, task_cfg)
    place_camera(actors, surface_obj, surface_spec, render_spec.camera, camera_azimuth)
    _set_frame_range(render_spec)

    return _build_scene_data(
        entry,
        surface_spec,
        task_angle,
        camera_azimuth,
        footprint,
        actors,
        render_spec,
        task_cfg,
    )


def _resolve_surface(scene_spec: SceneSpec, surface_name: str) -> SurfaceSpec:
    if surface_name not in scene_spec.surfaces:
        raise ValueError(
            f"surface {surface_name!r} not in scene {scene_spec.id!r}. Known: {sorted(scene_spec.surfaces)}"
        )
    return scene_spec.surfaces[surface_name]


def _open_blend(scene_spec: SceneSpec, render_spec: RenderSpec) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(get_registry().scene_path(scene_spec.id)))
    bpy.ops.object.select_all(action="DESELECT")
    _ensure_view3d()
    _init_physics(render_spec.frames)
    bpy.context.scene.view_settings.exposure = scene_spec.exposure


def _hide_clutter(surface_spec: SurfaceSpec) -> None:
    for obj_name in surface_spec.clutter_hide:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.hide_render = True
            obj.hide_viewport = True


def _measure_footprint(actors: dict) -> tuple[float, float]:
    bpy.context.view_layer.update()
    xs: list[float] = []
    ys: list[float] = []
    for obj in actors.values():
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            xs.append(world.x)
            ys.append(world.y)
    return (max(xs) - min(xs), max(ys) - min(ys))


def _apply_surface_physics(surface_obj, task_cfg: dict) -> None:
    attach_rigid_body(surface_obj, active=False, physics=PASSIVE_GEOMETRY_PHYSICS)
    for prop, value in task_cfg.get("surface_physics", {}).items():
        setattr(surface_obj.rigid_body, prop, value)


def _set_frame_range(render_spec: RenderSpec) -> None:
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = render_spec.frames
    bpy.context.scene.render.fps = render_spec.fps


def _build_scene_data(
    entry: ManifestEntry,
    surface_spec: SurfaceSpec,
    task_angle: float,
    camera_azimuth: float,
    footprint: tuple[float, float],
    actors: dict,
    render_spec: RenderSpec,
    task_cfg: dict,
) -> dict:
    data = {
        "task_type": entry.task_type,
        "scene_id": entry.scene_id,
        "seed": entry.seed,
        "surface": entry.surface_name,
        "center": list(surface_spec.center),
        "task_angle": round(task_angle, 2),
        "camera_azimuth": round(camera_azimuth, 2),
        "footprint": [round(footprint[0], 3), round(footprint[1], 3)],
        "actors": {
            name: {
                "object_id": obj.get("object_id"),
                "role": obj.get("role"),
                "position": list(obj.matrix_world.translation),
            }
            for name, obj in actors.items()
        },
        "frames": render_spec.frames,
        "prompt": _format_prompt(task_cfg["prompt_template"], actors),
    }
    if entry.split is not None:
        data["split"] = entry.split
    return data


def _ensure_view3d() -> None:
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                return
        for area in win.screen.areas:
            area.type = "VIEW_3D"
            return


def _init_physics(frames: int) -> None:
    if not bpy.context.scene.rigidbody_world:
        with bpy.context.temp_override(**view3d_context()):
            bpy.ops.rigidbody.world_add()
    rbw = bpy.context.scene.rigidbody_world
    rbw.point_cache.frame_end = frames + 30
    rbw.substeps_per_frame = 30
    rbw.solver_iterations = 20


def _format_object_name(object_id: str) -> str:
    name = object_id.replace("_", " ")
    article = "an" if name[0].lower() in "aeiou" else "a"
    return f"{article} {name}"


def _format_prompt(template: str, actors: dict) -> str:
    rendered = template.format(
        **{name: _format_object_name(obj.get("object_id", "object")) for name, obj in actors.items()}
    )
    return rendered[0].upper() + rendered[1:] if rendered else rendered
