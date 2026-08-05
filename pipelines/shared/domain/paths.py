from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Repository and data path roots."""

    root: Path

    @classmethod
    def from_cwd(cls, start: Path | None = None) -> PathConfig:
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "pipelines").is_dir():
                return cls(root=candidate)
        return cls(root=current)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def datasets_list(self) -> Path:
        return self.data_dir / "datasets.txt"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def raw_dir(self) -> Path:
        return self.datasets_dir / "raw"

    @property
    def yolo_dir(self) -> Path:
        return self.datasets_dir / "yolo"

    @property
    def yolo_yaml(self) -> Path:
        return self.yolo_dir / "data.yaml"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def detect_dir(self) -> Path:
        return self.runs_dir / "detect"

    @property
    def metrics_dir(self) -> Path:
        return self.runs_dir / "metrics"
