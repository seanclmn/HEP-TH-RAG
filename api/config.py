"""Settings from the environment, falling back to .env. Also puts pipeline/ on sys.path."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))


def _dotenv() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip("'\"")
    return out


def _get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _dotenv().get(name) or default


@dataclass
class Settings:
    database_url: str
    ollama_url: str
    ollama_model: str
    ollama_think: bool = False      # qwen3 etc.: reason before answering (slower first token)
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    db_options: str | None = None   # libpq "options", e.g. "-c search_path=..." (tests)
    history_messages: int = 20      # earlier messages sent to the LLM with each question
    per_paper: int = 2              # max retrieved chunks from one paper


def load() -> Settings:
    url = _get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (see .env.example)")
    return Settings(
        database_url=url,
        ollama_url=_get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_get("OLLAMA_MODEL", "qwen3:8b"),
        ollama_think=_get("OLLAMA_THINK", "false").lower() in ("1", "true", "yes"),
        cors_origins=tuple(o.strip() for o in _get("CORS_ORIGINS", "http://localhost:3000").split(",")),
    )


settings = load()
