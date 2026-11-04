"""Settings come from the environment only. No keys or URLs are hard-coded.

OPENAI_API_KEY / OPENAI_BASE_URL are read by `openai.OpenAI()` itself (exactly like the
class sandbox), so this module never touches them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env_int(env, name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None


def _env_float(env, name: str, default: float) -> float:
    try:
        return float(env.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None


@dataclass(frozen=True)
class Settings:
    llm_model: str = "gemma-4-E4B-it"
    temperature: float = 0.0          # 0 for anything evaluated or repeatable
    max_tokens: int = 700
    agent_mode: str = "auto"          # auto | native | json   (how tool calls are expressed)
    max_steps: int = 6                # hard stop for the agent loop
    top_k: int = 5
    retrieval: str = "hybrid"         # bm25 | dense | hybrid
    min_score: float = 0.0            # relevance floor (see eval/run_threshold_sweep.py)
    embed_backend: str = "local"      # local (sentence-transformers) | proxy (OpenAI embeddings API)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    docs_path: Path = ROOT / "data" / "docs.jsonl"
    index_dir: Path = ROOT / "data" / "index"

    def validate(self) -> "Settings":
        if self.agent_mode not in {"auto", "native", "json"}:
            raise ValueError("AGENT_MODE must be auto, native or json")
        if self.retrieval not in {"bm25", "dense", "hybrid"}:
            raise ValueError("RETRIEVAL must be bm25, dense or hybrid")
        if self.embed_backend not in {"local", "proxy"}:
            raise ValueError("EMBED_BACKEND must be local or proxy")
        if not 1 <= self.max_steps <= 12:
            raise ValueError("MAX_STEPS must be between 1 and 12")
        if not 1 <= self.top_k <= 10:
            raise ValueError("TOP_K must be between 1 and 10")
        return self


def load_settings(env=None) -> Settings:
    env = os.environ if env is None else env
    base = Settings()
    return Settings(
        llm_model=env.get("LLM_MODEL", base.llm_model),
        temperature=_env_float(env, "LLM_TEMPERATURE", base.temperature),
        max_tokens=_env_int(env, "LLM_MAX_TOKENS", base.max_tokens),
        agent_mode=env.get("AGENT_MODE", base.agent_mode),
        max_steps=_env_int(env, "MAX_STEPS", base.max_steps),
        top_k=_env_int(env, "TOP_K", base.top_k),
        retrieval=env.get("RETRIEVAL", base.retrieval),
        min_score=_env_float(env, "MIN_SCORE", base.min_score),
        embed_backend=env.get("EMBED_BACKEND", base.embed_backend),
        embed_model=env.get("EMBED_MODEL", base.embed_model),
    ).validate()
