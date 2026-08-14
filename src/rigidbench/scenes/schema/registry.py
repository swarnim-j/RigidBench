from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .assets import BlenderKitAssets
from .types import (
    AssetRef,
    CameraSpec,
    MaterialSpec,
    ObjectSpec,
    PhysicsSpec,
    RenderSpec,
    SceneSpec,
    Splits,
    SurfaceSpec,
    TaskSpec,
)

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


class Registry:
    def __init__(self, config_dir: Path = CONFIG_DIR) -> None:
        self.config_dir = config_dir
        self.assets = BlenderKitAssets()
        self._objects = self._load_objects()
        self._materials = self._load_materials()
        self._tasks = self._load_tasks()
        self._scenes = self._load_scenes()
        self._render = self._load_render()
        self._splits = self._load_splits()
        self._objects_by_tag = self._index_by_tag()
        self._validate_cross_refs()

    def _load_objects(self) -> dict[str, ObjectSpec]:
        out: dict[str, ObjectSpec] = {}
        for path in sorted((self.config_dir / "objects").glob("*.yaml")):
            data = _read_yaml(path)
            spec = ObjectSpec(
                id=data["id"],
                name=data["name"],
                asset=AssetRef(**data["asset"]),
                physics=PhysicsSpec(**data["physics"]),
                tags=tuple(data["tags"]),
            )
            if spec.id in out:
                raise ValueError(f"duplicate object id: {spec.id!r}")
            out[spec.id] = spec
        return out

    def _load_materials(self) -> dict[str, MaterialSpec]:
        out: dict[str, MaterialSpec] = {}
        for path in sorted((self.config_dir / "materials").glob("*.yaml")):
            data = _read_yaml(path)
            spec = MaterialSpec(
                id=data["id"],
                name=data["name"],
                asset=AssetRef(**data["asset"]),
            )
            if spec.id in out:
                raise ValueError(f"duplicate material id: {spec.id!r}")
            out[spec.id] = spec
        return out

    def _load_tasks(self) -> dict[str, TaskSpec]:
        out: dict[str, TaskSpec] = {}
        for path in sorted((self.config_dir / "tasks").glob("*.yaml")):
            data = _read_yaml(path)
            spec = TaskSpec(
                name=path.stem,
                placements=tuple(data["placements"]),
                prompt_template=data["prompt_template"],
                camera_range=tuple(tuple(c) for c in data.get("camera_range", [[0, 360]])),
                materials={k: tuple(v) for k, v in data.get("materials", {}).items()},
                surface_physics=dict(data.get("surface_physics", {})),
            )
            out[spec.name] = spec
        return out

    def _load_scenes(self) -> dict[str, SceneSpec]:
        out: dict[str, SceneSpec] = {}
        scenes_dir = self.config_dir / "scenes"
        if not scenes_dir.exists():
            return out
        for path in sorted(scenes_dir.glob("*.yaml")):
            data = _read_yaml(path)
            try:
                surfaces = {
                    name: SurfaceSpec(
                        name=name,
                        center=tuple(s["center"]),
                        size=tuple(s["size"]),
                        long_axis_deg=float(s["long_axis_deg"]),
                        clutter_hide=tuple(s.get("clutter_hide", ())),
                    )
                    for name, s in data["surfaces"].items()
                }
            except KeyError as e:
                raise ValueError(
                    f"scene {path.stem!r} surface missing required field {e}. Required: size, long_axis_deg."
                ) from None
            spec = SceneSpec(
                id=data["id"],
                name=data["name"],
                asset=AssetRef(**data["asset"]),
                surfaces=surfaces,
                exposure=data.get("exposure", 0.0),
            )
            if spec.id in out:
                raise ValueError(f"duplicate scene id: {spec.id!r}")
            out[spec.id] = spec
        return out

    def _load_render(self) -> RenderSpec:
        data = _read_yaml(self.config_dir / "render.yaml")
        return RenderSpec(
            resolution=tuple(data["resolution"]),
            fps=data["fps"],
            frames=data["frames"],
            engine=data["engine"],
            samples=data["samples"],
            use_gpu=data.get("use_gpu", True),
            output_format=data.get("output_format", "PNG"),
            camera=CameraSpec(**data["camera"]),
        )

    def _load_splits(self) -> Splits:
        data = _read_yaml(self.config_dir / "splits.yaml")
        return Splits(
            tasks=tuple(data["tasks"]),
            train_scenes=tuple(data["train"]["scenes"]),
            train_objects=tuple(data["train"]["objects"]),
            held_out_scenes=tuple(data["held_out"]["scenes"]),
            held_out_objects=tuple(data["held_out"]["objects"]),
        )

    def _index_by_tag(self) -> dict[str, tuple[str, ...]]:
        index: dict[str, list[str]] = {}
        for obj_id, obj in self._objects.items():
            for tag in obj.tags:
                index.setdefault(tag, []).append(obj_id)
        return {tag: tuple(ids) for tag, ids in index.items()}

    def _validate_cross_refs(self) -> None:
        for task in self._tasks.values():
            for placement in task.placements:
                tags = _coerce_tags(placement.get("tags"))
                for tag in tags:
                    if tag not in self._objects_by_tag:
                        raise ValueError(
                            f"task {task.name!r} placement {placement.get('name')!r} "
                            f"references unknown tag {tag!r}. "
                            f"Known: {sorted(self._objects_by_tag)}"
                        )
            for category, material_ids in task.materials.items():
                for mid in material_ids:
                    if mid not in self._materials:
                        raise ValueError(
                            f"task {task.name!r} materials.{category} references "
                            f"unknown material {mid!r}. Known: {sorted(self._materials)}"
                        )

        for task_name in self._splits.tasks:
            if task_name not in self._tasks:
                raise ValueError(f"splits references unknown task {task_name!r}. Known: {sorted(self._tasks)}")

        for scene_id in (*self._splits.train_scenes, *self._splits.held_out_scenes):
            if self._scenes and scene_id not in self._scenes:
                raise ValueError(f"splits references unknown scene {scene_id!r}. Known: {sorted(self._scenes)}")

        for obj_id in (*self._splits.train_objects, *self._splits.held_out_objects):
            if obj_id not in self._objects:
                raise ValueError(f"splits references unknown object {obj_id!r}. Known: {sorted(self._objects)}")

    def object(self, object_id: str) -> ObjectSpec:
        if object_id not in self._objects:
            raise KeyError(f"unknown object: {object_id!r}")
        return self._objects[object_id]

    def objects_by_tag(self, tag: str) -> tuple[str, ...]:
        if tag not in self._objects_by_tag:
            raise KeyError(f"unknown tag: {tag!r}. Known: {sorted(self._objects_by_tag)}")
        return self._objects_by_tag[tag]

    def material(self, material_id: str) -> MaterialSpec:
        if material_id not in self._materials:
            raise KeyError(f"unknown material: {material_id!r}")
        return self._materials[material_id]

    def task(self, name: str) -> TaskSpec:
        if name not in self._tasks:
            raise KeyError(f"unknown task: {name!r}")
        return self._tasks[name]

    def scene(self, scene_id: str) -> SceneSpec:
        if scene_id not in self._scenes:
            raise KeyError(f"unknown scene: {scene_id!r}. Known: {sorted(self._scenes)}")
        return self._scenes[scene_id]

    def render(self) -> RenderSpec:
        return self._render

    def asset_refs(self) -> tuple[AssetRef, ...]:
        """Return every configured BlenderKit asset, without duplicates."""
        refs = {spec.asset for specs in (self._objects, self._materials, self._scenes) for spec in specs.values()}
        return tuple(sorted(refs, key=lambda ref: (ref.type, ref.id)))

    def splits(self) -> Splits:
        return self._splits

    def object_path(self, object_id: str) -> Path:
        asset = self.object(object_id).asset
        path = self.assets.resolve(asset.id, asset.type)
        if path is None:
            raise FileNotFoundError(f"object {object_id!r} asset not found in {self.assets.root}")
        return path

    def material_path(self, material_id: str) -> Path:
        asset = self.material(material_id).asset
        path = self.assets.resolve(asset.id, asset.type)
        if path is None:
            raise FileNotFoundError(f"material {material_id!r} asset not found in {self.assets.root}")
        return path

    def scene_path(self, scene_id: str) -> Path:
        asset = self.scene(scene_id).asset
        path = self.assets.resolve(asset.id, asset.type)
        if path is None:
            raise FileNotFoundError(f"scene {scene_id!r} asset not found in {self.assets.root}")
        return path


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
