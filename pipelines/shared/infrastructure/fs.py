from __future__ import annotations

import shutil
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


def copy_or_link(src: Path, dst: Path, *, prefer_hardlink: bool = True) -> None:
    """Copy file, preferring hardlink when on the same filesystem."""
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    if prefer_hardlink:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def slug_from_kaggle(dataset: str) -> str:
    """samuelayman/bird-dataset -> bird-dataset."""
    return dataset.strip().split("/")[-1]
