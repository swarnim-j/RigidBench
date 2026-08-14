from __future__ import annotations

from pathlib import Path

from .models import ALL_MODELS, LOCAL_MODELS


class BaseGenerator:
    def __init__(self, model: str, checkpoint: str | None = None):
        self.model = model
        self.checkpoint = checkpoint
        self._cfg = ALL_MODELS[model]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def generate(
        self,
        prompt: str,
        image: str | Path,
        output_dir: str | Path,
        force: bool = False,
    ) -> Path | None:
        raise NotImplementedError

    def generate_batch(self, samples: list[dict]) -> dict[str, Path | None]:
        return {s["id"]: self.generate(s["prompt"], s["image"], s["output_dir"]) for s in samples}


def make_generator(model: str, checkpoint: str | None = None) -> BaseGenerator:
    if model not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model}. Available: {list(ALL_MODELS)}")
    if model in LOCAL_MODELS:
        from .local import LocalGenerator

        return LocalGenerator(model, checkpoint)
    from .remote import ReplicateGenerator

    return ReplicateGenerator(model, checkpoint)
