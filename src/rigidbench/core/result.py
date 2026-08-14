from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    sample_id: str
    task_type: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"sample_id": self.sample_id, "task_type": self.task_type, **self.metrics}

    @classmethod
    def from_json(cls, d: dict) -> "Result":
        metrics = {k: float(v) for k, v in d.items() if k not in ("sample_id", "task_type")}
        return cls(d["sample_id"], d["task_type"], metrics)

    def get(self, key: str, default=None):
        return self.metrics.get(key, default)
