from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    scene_id: str
    task_type: str
    surface_name: str
    seed: int
    split: str | None = None
    allowed_objects: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        objects = d.get("objects")
        return cls(
            sample_id=d["sample_id"],
            scene_id=d["scene_id"],
            task_type=d["task_type"],
            surface_name=d["surface_name"],
            seed=int(d["seed"]),
            split=d.get("split"),
            allowed_objects=tuple(objects) if objects is not None else None,
        )
