from __future__ import annotations

from .base import BackboneAdapter
from .wan import WanAdapter

_REGISTRY: dict[str, type[BackboneAdapter]] = {
    "wan": WanAdapter,
}


def get_adapter(name: str, cfg: dict) -> BackboneAdapter:
    """Construct a registered backbone adapter by name, pulling kwargs from cfg."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name].from_config(cfg)


__all__ = ["BackboneAdapter", "WanAdapter", "get_adapter"]
