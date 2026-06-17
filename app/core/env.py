from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env() -> None:
    """Load `.env` from the repository root (no-op if the file is missing)."""
    load_dotenv(_PROJECT_ROOT / ".env")
