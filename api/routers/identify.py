import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from config import ALLOWED_EXTENSIONS
from audio_service import convert_to_wav
from triton_service import get_embedding
from database import identify_speaker

router = APIRouter()


@router.post("/identify")
async def identify(request: Request, file: UploadFile = File(...)):
    """
    Accepts an audio/video file and returns the identified speaker.

    The router pulls the shared Triton and Qdrant clients from app.state
    (set once at startup in main.py) — no reconnection per request.
    """
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Accepted: {ALLOWED_EXTENSIONS}"
        )

    # Write upload to a temp file for ffmpeg to process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as tmp:
        tmp.write(await file.read())
        raw_path = tmp.name

    processed_path = raw_path + "_processed.wav"

    try:
        # Guard clauses — fail fast with clean 503 if clients didn't connect at startup
        if request.app.state.triton is None:
            raise HTTPException(status_code=503, detail="Triton server is offline or failed at startup.")
        if request.app.state.qdrant is None:
            raise HTTPException(status_code=503, detail="Qdrant database is offline or failed at startup.")

        # Step 1: Convert to 16kHz mono WAV, trimmed to 5 seconds
        if not convert_to_wav(raw_path, processed_path):
            raise HTTPException(
                status_code=422,
                detail="Audio processing failed. File may be too short (min 2s) or corrupt."
            )

        # Step 2: Get 192-dim embedding from Triton
        vector = get_embedding(processed_path, request.app.state.triton)
        if vector is None:
            raise HTTPException(
                status_code=503,
                detail="Triton embedding failed. Check server connection."
            )

        # Step 3: Search Qdrant and return identification result
        result = identify_speaker(vector, request.app.state.qdrant)
        return result

    finally:
        # Always clean up — even if an exception was raised above
        for path in [raw_path, processed_path]:
            if os.path.exists(path):
                os.remove(path)