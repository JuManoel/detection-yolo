from __future__ import annotations

import os
from pathlib import Path

from pipelines.shared.domain.paths import PathConfig

_LOADED = False


def load_project_env(start: Path | None = None) -> Path | None:
    """Load repo `.env` into os.environ (idempotent).

    Must run before importing ``kaggle`` so ``KAGGLE_API_TOKEN`` is visible.
    """
    global _LOADED
    if _LOADED:
        env_path = PathConfig.from_cwd(start).root / ".env"
        return env_path if env_path.exists() else None

    from dotenv import load_dotenv

    paths = PathConfig.from_cwd(start)
    env_path = paths.root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    _LOADED = True
    return env_path if env_path.exists() else None


def require_kaggle_api_token() -> str:
    """Return KAGGLE_API_TOKEN after loading .env, or raise."""
    load_project_env()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing KAGGLE_API_TOKEN. Copy .env.example to .env and set your "
            "token from https://www.kaggle.com/settings/api"
        )
    # Keep token in env for kaggle's import-time auth
    os.environ["KAGGLE_API_TOKEN"] = token
    return token
