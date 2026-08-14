from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RenderedSample:
    id: str
    task_type: str
    first_frame: Path
    prompt: str
    gt_mask: Path
    reference_video: Path | None = None
    gt_depth: Path | None = None
    trajectories: Path | None = None
    metadata: dict = field(default_factory=dict)
