from . import contact, position  # noqa: F401  (populate registry)
from .base import TARGETS, Target, get_target, register_target

__all__ = ["TARGETS", "Target", "get_target", "register_target"]
