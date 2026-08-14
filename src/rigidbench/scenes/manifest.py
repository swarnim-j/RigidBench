from __future__ import annotations

import random

from .schema.registry import get_registry


def build_train_manifest(num_samples: int, seed: int) -> list[dict]:
    """Stratified manifest of num_samples rows across (task, scene)."""
    registry = get_registry()
    splits = registry.splits()
    scenes_by_id = {sid: registry.scene(sid) for sid in splits.train_scenes}
    rng = random.Random(seed)

    samples: list[dict] = []
    tasks = list(splits.tasks)
    scenes = list(splits.train_scenes)
    per_task = num_samples // len(tasks)
    remainder = num_samples % len(tasks)

    for idx, task in enumerate(tasks):
        n = per_task + (1 if idx < remainder else 0)
        per_scene = n // len(scenes)
        scene_remainder = n % len(scenes)
        for scene_idx, scene_id in enumerate(scenes):
            scene_n = per_scene + (1 if scene_idx < scene_remainder else 0)
            spec = scenes_by_id[scene_id]
            for _ in range(scene_n):
                samples.append(
                    {
                        "sample_id": f"{len(samples):05d}",
                        "scene_id": scene_id,
                        "task_type": task,
                        "surface_name": rng.choice(list(spec.surfaces)),
                        "seed": (seed * 1000003 + len(samples)) % (2**31),
                        "split": "train",
                        "objects": list(splits.train_objects),
                    }
                )

    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s["sample_id"] = f"{i:05d}"
    return samples
