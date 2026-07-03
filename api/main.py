"""
main.py — Entry point for the Speaker Identification API

Start with:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Port map:
    localhost:8080  →  This API
    localhost:6333  →  Qdrant (qdrant_storage/)
    192.168.18.30:8040  →  Triton inference server (office)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import QDRANT_URL, COLLECTION_NAME
from triton_service import create_triton_client
from database import create_qdrant_client
from routers.identify import router as identify_router
from routers.enroll import router as enroll_router

# ─────────────────────────────────────────────
# LIFESPAN — startup & shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────
    print("[startup] Connecting to Triton...")
    try:
        app.state.triton = create_triton_client()
    except RuntimeError as e:
        print(f"[startup]  Triton unreachable: {e}")
        print("[startup]    Are you on the office network? Check 192.168.18.30:8040")
        app.state.triton = None   # API will start but /identify returns 503

    print("[startup] Connecting to Qdrant...")
    try:
        app.state.qdrant = create_qdrant_client()
        app.state.qdrant.get_collections()  # ping to verify
        print(f"[startup]  Qdrant connected at {QDRANT_URL}")
    except Exception as e:
        print(f"[startup]  Qdrant unreachable: {e}")
        print("[startup]    Run: QDRANT__STORAGE__STORAGE_PATH=./qdrant_storage ./qdrant")
        app.state.qdrant = None

    yield  # ← app runs here

    # ── SHUTDOWN ──────────────────────────────
    print("[shutdown] Cleaning up connections...")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Speaker Identification API",
    description = "Identifies Pakistani political speakers from audio using ECAPA-TDNN + Qdrant",
    version     = "1.0.0",
    lifespan    = lifespan
)

# Register routers
app.include_router(identify_router)
app.include_router(enroll_router)

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Returns live status of Triton and Qdrant.
    QA should hit this first before running any identification tests.
    """
    qdrant_status = "disconnected"
    total_vectors = 0

    # Safely check Qdrant
    if app.state.qdrant is not None:
        try:
            info = app.state.qdrant.get_collection(COLLECTION_NAME)
            qdrant_status = "connected"
            total_vectors = info.points_count
        except Exception as e:
            qdrant_status = f"error: {e}"

    triton_status = "disconnected"

    # Safely check Triton
    if app.state.triton is not None:
        try:
            triton_live   = app.state.triton.is_server_live()
            triton_status = "connected" if triton_live else "not live"
        except Exception as e:
            triton_status = f"error: {e}"

    overall = "healthy" if (qdrant_status == "connected" and triton_status == "connected") else "degraded"

    return {
        "status"        : overall,
        "qdrant"        : qdrant_status,
        "triton"        : triton_status,
        "total_vectors" : total_vectors,
        "collection"    : COLLECTION_NAME
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)