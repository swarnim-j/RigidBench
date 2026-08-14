from __future__ import annotations

import math
from dataclasses import dataclass

from mathutils import Vector

from ..schema.types import CameraSpec


@dataclass
class Framing:
    center: Vector
    distance: float


def framing_bbox(actors, surface_obj, surface_spec, cam: CameraSpec) -> Framing:
    coords: list[Vector] = []
    for obj in actors.values():
        coords.extend(obj.matrix_world @ Vector(v) for v in obj.bound_box)
    coords.append(Vector(surface_spec.center))
    ax_min_x = min(c.x for c in coords)
    ax_max_x = max(c.x for c in coords)
    ax_min_y = min(c.y for c in coords)
    ax_max_y = max(c.y for c in coords)
    for v in surface_obj.bound_box:
        p = surface_obj.matrix_world @ Vector(v)
        if ax_min_x <= p.x <= ax_max_x and ax_min_y <= p.y <= ax_max_y:
            coords.append(p)
    pad = cam.bbox_padding
    min_x = min(c.x for c in coords) - pad
    max_x = max(c.x for c in coords) + pad
    min_y = min(c.y for c in coords) - pad
    max_y = max(c.y for c in coords) + pad
    min_z = min(c.z for c in coords) - pad
    max_z = max(c.z for c in coords) + 1.2 * pad
    center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    scene_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
    return Framing(center, max(scene_size * cam.framing_margin, cam.min_distance))


def camera_position(center: Vector, distance: float, elevation_deg: float, az_deg: float) -> Vector:
    elev = math.radians(elevation_deg)
    az = math.radians(az_deg)
    return center + Vector(
        (
            distance * math.cos(elev) * math.sin(az),
            -distance * math.cos(elev) * math.cos(az),
            distance * math.sin(elev),
        )
    )
