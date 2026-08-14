from __future__ import annotations

import math
import random
from dataclasses import dataclass

import bpy
from mathutils import Vector

from ..builder import SceneBuilder
from ..schema.registry import get_registry

PLACEMENTS: dict[str, type] = {}


def placement(type_name: str):
    """Register the decorated placement class under `type_name` in PLACEMENTS."""

    def decorator(cls):
        PLACEMENTS[type_name] = cls
        return cls

    return decorator


@dataclass
class PlacementContext:
    scene: SceneBuilder
    actors: dict
    surface: object

    def resolve(self, name: str) -> object:
        """Look up `name` in the placed actors, or return the surface object when name is 'surface'."""
        if name == "surface":
            return self.surface
        if name not in self.actors:
            raise ValueError(f"reference {name!r} not found. Available: {list(self.actors.keys()) + ['surface']}")
        return self.actors[name]


@dataclass
class PlacementResult:
    actors: dict
    footprints: list[tuple[float, float, float]]


def sample_value(spec):
    """Sample from a [lo, hi] range (int or float) or return a scalar as-is."""
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        if isinstance(spec[0], int) and isinstance(spec[1], int):
            return random.randint(spec[0], spec[1])
        return random.uniform(spec[0], spec[1])
    return spec


def _exit_y(obj):
    bpy.context.view_layer.update()
    return max((obj.matrix_world @ Vector(v)).y for v in obj.bound_box)


def _ramp_position_y(ramp_obj, frac):
    bpy.context.view_layer.update()
    verts_y = [(ramp_obj.matrix_world @ Vector(v)).y for v in ramp_obj.bound_box]
    return min(verts_y) + frac * (max(verts_y) - min(verts_y))


class BasePlacement:
    is_static: bool = False

    def __init__(self, spec, materials):
        self.spec = spec
        self.name = spec["name"]
        self.materials = materials
        self.object_ids = self._resolve_objects()

    def _resolve_objects(self) -> list[str]:
        tags_field = self.spec.get("tags")
        if tags_field is None:
            return []
        tags = (tags_field,) if isinstance(tags_field, str) else tuple(tags_field)
        registry = get_registry()
        sets = [set(registry.objects_by_tag(t)) for t in tags]
        candidates = sorted(set.intersection(*sets)) if sets else []
        allowed = self.spec.get("_allowed_objects")
        if allowed is not None:
            candidates = [c for c in candidates if c in set(allowed)]
        return candidates

    def pick_object(self) -> str:
        if not self.object_ids:
            raise ValueError(f"placement {self.name!r} has no candidate objects (check tags)")
        return random.choice(self.object_ids)

    def pick_material(self, mat_key):
        pool = self.materials.get(mat_key)
        return random.choice(pool) if pool else None

    def execute(self, pctx: PlacementContext) -> PlacementResult:
        raise NotImplementedError


@placement("object")
class ObjectPlacement(BasePlacement):
    def execute(self, pctx: PlacementContext) -> PlacementResult:
        s = self.spec
        obj_id = self.pick_object()

        on_ref = s.get("support")
        above_ref = s.get("above")
        ref_name = on_ref or above_ref
        ref_obj = pctx.resolve(ref_name)

        cx = ref_obj.location.x if ref_name != "surface" else 0.0
        cy = 0.0

        if on_ref and on_ref != "surface" and "position" in s:
            cy = _ramp_position_y(ref_obj, sample_value(s["position"]))
            cx = ref_obj.location.x

        if s.get("in_front_of"):
            front_ref = pctx.resolve(s["in_front_of"])
            cy = _exit_y(front_ref) + sample_value(s.get("distance", 0.1))
            cx = front_ref.location.x

        if above_ref:
            height = sample_value(s.get("height", 0.5))
            if "position" in s and above_ref != "surface":
                cy = _ramp_position_y(ref_obj, sample_value(s["position"]))
                cx = ref_obj.location.x
            obj = pctx.scene.spawn(obj_id, self.name, at=(cx, cy), above=(ref_obj, height))
        else:
            obj = pctx.scene.spawn(
                obj_id,
                self.name,
                at=(cx, cy),
                on=ref_obj,
                align_to_surface=s.get("align", False),
            )

        if above_ref:
            return PlacementResult({self.name: obj}, [])
        radius = pctx.scene.bbox_radius(obj)
        return PlacementResult({self.name: obj}, [(obj.location.x, obj.location.y, radius)])


@placement("ramp")
class RampPlacement(BasePlacement):
    is_static = True

    def execute(self, pctx: PlacementContext) -> PlacementResult:
        s = self.spec
        length = sample_value(s["length"])
        height = sample_value(s["height"])
        width = sample_value(s["width"])
        mat = self.pick_material(s.get("material"))

        ref_obj = pctx.resolve(s.get("support", "surface"))
        cx, cy = 0.0, 0.0

        facing = s.get("facing", "forward")
        if s.get("gap"):
            gap = sample_value(s["gap"])
            cy = -(gap / 2 + length) if facing == "forward" else gap / 2 + length

        ramp = pctx.scene.create_ramp(
            self.name,
            at=(cx, cy),
            on=ref_obj,
            length=length,
            height=height,
            width=width,
            material=mat,
            facing=facing,
        )
        ramp["_ramp_length"] = length
        ramp["_ramp_height"] = height

        radius = max(length, width) / 2
        return PlacementResult({self.name: ramp}, [(ramp.location.x, ramp.location.y, radius)])


@placement("line")
class LinePlacement(BasePlacement):
    def execute(self, pctx: PlacementContext) -> PlacementResult:
        s = self.spec
        count = sample_value(s.get("count", 3))
        ref_obj = pctx.resolve(s.get("support", "surface"))
        gap = sample_value(s.get("gap", 0.03))

        cx, cy = 0.0, 0.0
        if s.get("in_front_of"):
            front_ref = pctx.resolve(s["in_front_of"])
            cy = _exit_y(front_ref) + sample_value(s.get("distance", 0.1))
            cx = front_ref.location.x

        max_radius = max(pctx.scene.cached_bbox_radius(oid) for oid in self.object_ids)

        result_actors = {}
        prev_obj = None
        for i in range(count):
            name = f"{self.name}_{i}"
            if i == 0:
                obj = pctx.scene.spawn(
                    self.pick_object(),
                    name,
                    at=(cx, cy + max_radius),
                    on=ref_obj,
                )
            else:
                obj = pctx.scene.spawn(
                    self.pick_object(),
                    name,
                    beside=(prev_obj, gap, ref_obj),
                )
            result_actors[name] = obj
            prev_obj = obj

        total_radius = max_radius * count + gap * (count - 1) if count > 0 else 0.1
        return PlacementResult(result_actors, [(cx, cy, total_radius / 2)])


@placement("cluster")
class ClusterPlacement(BasePlacement):
    def execute(self, pctx: PlacementContext) -> PlacementResult:
        s = self.spec
        count = sample_value(s.get("count", 3))
        gap = sample_value(s.get("gap", 0.01))
        ref_obj = pctx.resolve(s.get("support", "surface"))

        cx, cy = 0.0, 0.0
        if s.get("in_front_of"):
            front_ref = pctx.resolve(s["in_front_of"])
            cy = _exit_y(front_ref) + sample_value(s.get("distance", 0.2))
            cx = front_ref.location.x

        picked_ids = [self.pick_object() for _ in range(count)]
        max_r = max(pctx.scene.cached_bbox_radius(oid) for oid in picked_ids)
        if count == 1:
            offsets = [(0.0, 0.0)]
        else:
            R = (max_r + gap / 2) / math.sin(math.pi / count)
            offsets = [
                (
                    R * math.cos(2 * math.pi * k / count - math.pi / 2),
                    R * math.sin(2 * math.pi * k / count - math.pi / 2),
                )
                for k in range(count)
            ]

        result_actors = {}
        for i, (dx, dy) in enumerate(offsets):
            name = f"{self.name}_{i}"
            obj = pctx.scene.spawn(
                picked_ids[i],
                name,
                at=(cx + dx, cy + dy),
                on=ref_obj,
            )
            result_actors[name] = obj

        footprint_radius = max_r if count == 1 else (max_r + gap / 2) / math.sin(math.pi / count) + max_r
        return PlacementResult(result_actors, [(cx, cy, footprint_radius)])
