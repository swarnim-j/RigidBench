from __future__ import annotations

import bpy

from ..schema.registry import get_registry
from .layout import solve_layout
from .placement import PLACEMENTS, PlacementContext


def load_task_config(task_type: str) -> dict:
    task = get_registry().task(task_type)
    return {
        "placements": [dict(p) for p in task.placements],
        "materials": {k: list(v) for k, v in task.materials.items()},
        "prompt_template": task.prompt_template,
        "camera_range": [list(r) for r in task.camera_range],
        "surface_physics": dict(task.surface_physics),
    }


def execute_task(task_cfg: dict, surface_obj, scene) -> dict:
    """Run each placement in order, solve overlaps, return the spawned actors by name."""
    materials = task_cfg.get("materials", {})
    actors: dict = {}
    all_footprints: list[dict] = []
    ordering_constraints: list[dict] = []

    for spec in task_cfg["placements"]:
        ptype = spec["type"]
        if ptype not in PLACEMENTS:
            raise ValueError(f"unknown placement type: {ptype!r}")

        placement_obj = PLACEMENTS[ptype](spec=spec, materials=materials)
        pctx = PlacementContext(scene=scene, actors=actors, surface=surface_obj)
        result = placement_obj.execute(pctx)
        actors.update(result.actors)

        for x, y, r in result.footprints:
            all_footprints.append(
                {
                    "name": spec["name"],
                    "x": x,
                    "y": y,
                    "radius": r,
                    "static": placement_obj.is_static,
                }
            )

        if spec.get("in_front_of"):
            ordering_constraints.append(
                {
                    "type": "in_front_of",
                    "a": spec["name"],
                    "b": spec["in_front_of"],
                    "min_gap": 0.05,
                }
            )

    if len(all_footprints) > 1:
        adjusted = solve_layout(all_footprints, ordering_constraints)
        for name, (new_x, new_y) in adjusted.items():
            if name in actors:
                obj = actors[name]
                if abs(new_x - obj.location.x) > 0.002 or abs(new_y - obj.location.y) > 0.002:
                    obj.location.x = new_x
                    obj.location.y = new_y
                    bpy.context.view_layer.update()

    return actors
