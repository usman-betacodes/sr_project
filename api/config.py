# ═══════════════════════════════════════════════
# config.py — Single source of truth for all settings
# To change any URL, threshold, or model name → edit here only
# ═══════════════════════════════════════════════
import os
# Qdrant (local binary → qdrant_storage/)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME   = "speakers"
VECTOR_SIZE       = 192

# Triton inference server (office network)
TRITON_URL        = "192.168.18.30:8040"
MODEL_NAME        = "ecapa_speaker_verification"
INPUT_NAME        = "audio_input"
OUTPUT_NAME       = "embeddings"

# Audio processing
SAMPLE_RATE       = 16000
CHUNK_SIZE        = 16000        # 1 second per Triton inference chunk
MIN_AUDIO_MS      = 2000         # reject uploads shorter than 2 seconds
CLIP_DURATION_MS  = 10000         # trim uploaded audio to first 10 seconds

ALLOWED_EXTENSIONS = ('.wav', '.mp3', '.mp4', '.mkv', '.avi', '.mov')

# Identification logic
TOP_K             = 20           # how many Qdrant results to fetch
MIN_HITS_REQUIRED = 3            # winner needs at least this many hits
REJECT_THRESHOLD  = 0.68         # winner's top-3 avg must exceed this
# Fallback when strict gate fails
FALLBACK_ENABLED    = True
FALLBACK_MIN_SCORE  = 0.45   # 45% — change to 0.50 for 50%