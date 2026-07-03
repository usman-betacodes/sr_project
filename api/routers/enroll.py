import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from enrollment_service import enroll_all_speakers

router = APIRouter()

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_active_job_id: str | None = None


class EnrollmentRequest(BaseModel):
    processed_dir: str = Field(
        ...,
        description="Path to the processed dataset folder (one subfolder per speaker with chunk_*.wav files)",
        examples=["/home/uthmans/Documents/BetaCodes/work_done_projects/speaker_recogition/testing"],
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return dict(job)


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_enrollment_job(
    job_id: str,
    processed_dir: str,
    triton_client,
    qdrant_client,
) -> None:
    global _active_job_id

    def on_progress(progress: dict) -> None:
        _update_job(job_id, **progress)

    try:
        _update_job(job_id, status="running", started_at=_utc_now(), message="Enrollment in progress")

        result = enroll_all_speakers(
            processed_dir=processed_dir,
            triton_client=triton_client,
            qdrant_client=qdrant_client,
            on_progress=on_progress,
        )

        _update_job(
            job_id,
            status="completed",
            finished_at=_utc_now(),
            message="Enrollment complete",
            result=result,
            speakers_done=result["speakers_processed"],
            speakers_total=result["speakers_processed"],
            vectors_enrolled=result["vectors_enrolled"],
            chunks_skipped=result["chunks_skipped"],
            chunks_failed=result["chunks_failed"],
        )
    except Exception as e:
        _update_job(
            job_id,
            status="failed",
            finished_at=_utc_now(),
            message=str(e),
            error=str(e),
        )
    finally:
        with _jobs_lock:
            if _active_job_id == job_id:
                _active_job_id = None


@router.post("/enroll")
async def start_enrollment(request: Request, body: EnrollmentRequest):
    """
    Start background enrollment from a processed dataset folder.

    Expected layout:
        <processed_dir>/<SpeakerName>/chunk_000.wav
    """
    global _active_job_id

    processed_dir = body.processed_dir.strip()
    if not processed_dir:
        raise HTTPException(status_code=400, detail="processed_dir cannot be empty.")

    if request.app.state.triton is None:
        raise HTTPException(status_code=503, detail="Triton server is offline or failed at startup.")
    if request.app.state.qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant database is offline or failed at startup.")

    with _jobs_lock:
        if _active_job_id is not None:
            active = _jobs.get(_active_job_id, {})
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Another enrollment job is already running.",
                    "active_job_id": _active_job_id,
                    "active_status": active.get("status"),
                },
            )

        job_id = str(uuid.uuid4())
        _active_job_id = job_id
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "processed_dir": processed_dir,
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "message": "Job queued",
            "current_speaker": None,
            "speakers_done": 0,
            "speakers_total": 0,
            "vectors_enrolled": 0,
            "chunks_skipped": 0,
            "chunks_failed": 0,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_enrollment_job,
        args=(job_id, processed_dir, request.app.state.triton, request.app.state.qdrant),
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/enroll/status/{job_id}",
        "processed_dir": processed_dir,
    }


@router.get("/enroll/status/{job_id}")
async def enrollment_status(job_id: str):
    """Check progress or final result of an enrollment job."""
    return _get_job(job_id)
