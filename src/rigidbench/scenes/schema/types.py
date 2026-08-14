from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_ALLOWED_ASSET_SOURCES = {"blenderkit"}
_ALLOWED_ASSET_TYPES = {"model", "scene", "material"}
_ALLOWED_COLLISION_SHAPES = {
    "BOX",
    "SPHERE",
    "CAPSULE",
    "CYLINDER",
    "CONE",
    "CONVEX_HULL",
    "MESH",
    "COMPOUND",
}


@dataclass(frozen=True)
class AssetRef:
    source: str
    type: str
    id: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.source not in _ALLOWED_ASSET_SOURCES:
            raise ValueError(f"unknown asset source: {self.source!r}")
        if self.type not in _ALLOWED_ASSET_TYPES:
            raise ValueError(f"unknown asset type: {self.type!r}")
        if not self.id:
            raise ValueError("asset id is required")
        if self.scale <= 0:
            raise ValueError(f"asset scale must be positive, got {self.scale}")


@dataclass(frozen=True)
class PhysicsSpec:
    mass: float
    friction: float
    restitution: float
    collision_shape: str

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        if not 0.0 <= self.friction <= 2.0:
            raise ValueError(f"friction out of range [0, 2]: {self.friction}")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(f"restitution out of range [0, 1]: {self.restitution}")
        if self.collision_shape not in _ALLOWED_COLLISION_SHAPES:
            raise ValueError(f"unknown collision shape: {self.collision_shape!r}")


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    name: str
    asset: AssetRef
    physics: PhysicsSpec
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("object id is required")
        if not self.tags:
            raise ValueError(f"object {self.id!r} has no tags")
        if any(not t for t in self.tags):
            raise ValueError(f"object {self.id!r} has empty tag")


@dataclass(frozen=True)
class MaterialSpec:
    id: str
    name: str
    asset: AssetRef

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("material id is required")
        if self.asset.type != "material":
            raise ValueError(f"material {self.id!r} asset type must be 'material', got {self.asset.type!r}")


@dataclass(frozen=True)
class SurfaceSpec:
    """A flat region of a scene where tasks can be placed, with size along long_axis_deg."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float]
    long_axis_deg: float
    clutter_hide: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.center) != 3:
            raise ValueError(f"surface {self.name!r} center must be (x, y, z), got {self.center}")
        if len(self.size) != 2 or any(s <= 0 for s in self.size):
            raise ValueError(f"surface {self.name!r} size must be positive (L, W), got {self.size}")
        if self.size[0] < self.size[1]:
            raise ValueError(f"surface {self.name!r} size must list long axis first: got {self.size}")
        if not 0.0 <= self.long_axis_deg < 180.0:
            raise ValueError(f"surface {self.name!r} long_axis_deg must be in [0, 180), got {self.long_axis_deg}")


@dataclass(frozen=True)
class SceneSpec:
    id: str
    name: str
    asset: AssetRef
    surfaces: dict[str, SurfaceSpec]
    exposure: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scene id is required")
        if self.asset.type != "scene":
            raise ValueError(f"scene {self.id!r} asset type must be 'scene', got {self.asset.type!r}")
        if not self.surfaces:
            raise ValueError(f"scene {self.id!r} has no surfaces")


@dataclass(frozen=True)
class CameraSpec:
    elevation: float
    min_distance: float
    framing_margin: float
    bbox_padding: float
    lens: float


@dataclass(frozen=True)
class RenderSpec:
    resolution: tuple[int, int]
    fps: int
    frames: int
    engine: str
    samples: int
    camera: CameraSpec
    use_gpu: bool = True
    output_format: str = "PNG"

    def __post_init__(self) -> None:
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if (self.frames - 1) % 4 != 0:
            raise ValueError(f"frames must be 4n+1 for clean Wan-VAE encoding, got {self.frames}")
        if self.samples <= 0:
            raise ValueError(f"render samples must be positive, got {self.samples}")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    placements: tuple[dict[str, Any], ...]
    prompt_template: str
    camera_range: tuple[tuple[float, float], ...] = ((0.0, 360.0),)
    materials: dict[str, tuple[str, ...]] = field(default_factory=dict)
    surface_physics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name is required")
        if not self.placements:
            raise ValueError(f"task {self.name!r} has no placements")
        if not self.prompt_template:
            raise ValueError(f"task {self.name!r} has empty prompt_template")
        seen = set()
        for p in self.placements:
            if "type" not in p:
                raise ValueError(f"task {self.name!r}: placement missing 'type' field")
            if "name" not in p:
                raise ValueError(f"task {self.name!r}: placement missing 'name' field")
            if p["name"] in seen:
                raise ValueError(f"task {self.name!r}: duplicate placement name {p['name']!r}")
            seen.add(p["name"])


@dataclass(frozen=True)
class Splits:
    tasks: tuple[str, ...]
    train_scenes: tuple[str, ...]
    train_objects: tuple[str, ...]
    held_out_scenes: tuple[str, ...]
    held_out_objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("splits define no tasks")
        if not self.train_scenes:
            raise ValueError("splits define no train scenes")
        if not self.train_objects:
            raise ValueError("splits define no train objects")
        if set(self.train_scenes) & set(self.held_out_scenes):
            raise ValueError("train and held-out scenes overlap")
        if set(self.train_objects) & set(self.held_out_objects):
            raise ValueError("train and held-out objects overlap")
