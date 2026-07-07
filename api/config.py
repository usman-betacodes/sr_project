# ═══════════════════════════════════════════════
# Load from .env at project root (see .env.example)
# ═══════════════════════════════════════════════
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value is not None else default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value is not None else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(key)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


# API server
API_HOST = _env_str("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8080)

# Qdrant
QDRANT_URL = _env_str("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None if not set (unauthenticated)
COLLECTION_NAME = _env_str("COLLECTION_NAME", "speakers")
VECTOR_SIZE = _env_int("VECTOR_SIZE", 192)

# Triton inference server
TRITON_URL = _env_str("TRITON_URL", "192.168.18.30:8040")
MODEL_NAME = _env_str("MODEL_NAME", "ecapa_speaker_verification")
INPUT_NAME = _env_str("INPUT_NAME", "audio_input")
OUTPUT_NAME = _env_str("OUTPUT_NAME", "embeddings")

# Audio processing
SAMPLE_RATE = _env_int("SAMPLE_RATE", 16000)
CHUNK_SIZE = _env_int("CHUNK_SIZE", 16000)
MIN_AUDIO_MS = _env_int("MIN_AUDIO_MS", 2000)
CLIP_DURATION_MS = _env_int("CLIP_DURATION_MS", 10000)

ALLOWED_EXTENSIONS = _env_tuple(
    "ALLOWED_EXTENSIONS",
    (".wav", ".mp3", ".mp4", ".mkv", ".avi", ".mov"),
)

# Identification logic
TOP_K = _env_int("TOP_K", 20)
MIN_HITS_REQUIRED = _env_int("MIN_HITS_REQUIRED", 3)
REJECT_THRESHOLD = _env_float("REJECT_THRESHOLD", 0.60)
FALLBACK_ENABLED = _env_bool("FALLBACK_ENABLED", True)
FALLBACK_MIN_SCORE = _env_float("FALLBACK_MIN_SCORE", 0.50)
