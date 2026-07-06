from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
import uuid
from typing import Any

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
    InferenceServerException,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))

from config import (  # noqa: E402
    CHUNK_SIZE,
    COLLECTION_NAME,
    INPUT_NAME,
    MODEL_NAME,
    OUTPUT_NAME,
    QDRANT_URL,
    SAMPLE_RATE,
    TRITON_URL,
    VECTOR_SIZE,
)

DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "raw_data")
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")

TRITON_CHUNK_SZ = CHUNK_SIZE
CHUNK_LENGTH_MS = 5000
DEFAULT_TARGET_CHUNKS = 240

MEDIA_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".m4a", ".flac")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("enroll_pipeline")


def process_raw_to_chunks(
    raw_dir: str,
    processed_dir: str,
    target_chunks: int,
) -> dict[str, int]:
    """Raw media → 5s WAV chunks per speaker. Resume-safe."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    os.makedirs(processed_dir, exist_ok=True)

    speakers = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]
    if not speakers:
        raise ValueError(f"No speaker folders found in: {raw_dir}")

    report: dict[str, int] = {}

    for speaker in speakers:
        logger.info("[%s] Phase 1 — target: %d chunks", speaker, target_chunks)

        speaker_raw = os.path.join(raw_dir, speaker)
        speaker_out = os.path.join(processed_dir, speaker)
        os.makedirs(speaker_out, exist_ok=True)

        media_files = [
            f for f in os.listdir(speaker_raw)
            if f.lower().endswith(MEDIA_EXTENSIONS)
        ]
        if not media_files:
            logger.warning("[%s] No media files found. Skipping.", speaker)
            report[speaker] = 0
            continue

        existing = [
            f for f in os.listdir(speaker_out)
            if f.startswith("chunk_") and f.endswith(".wav")
        ]
        total_chunks = len(existing)

        if total_chunks >= target_chunks:
            logger.info("[%s] Already has %d chunks. Skipping.", speaker, total_chunks)
            report[speaker] = 0
            continue

        chunks_per_file = math.ceil(target_chunks / len(media_files))
        new_this_run = 0

        for file_idx, media_file in enumerate(media_files):
            if total_chunks >= target_chunks:
                break

            media_path = os.path.join(speaker_raw, media_file)
            temp_wav = os.path.join(speaker_out, f"temp_{file_idx}.wav")

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", media_path,
                        "-vn", "-acodec", "pcm_s16le",
                        "-ar", str(SAMPLE_RATE), "-ac", "1",
                        temp_wav,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("[%s] ffmpeg failed on %s: %s", speaker, media_file, exc)
                continue

            try:
                audio = AudioSegment.from_wav(temp_wav)
                speech_parts = split_on_silence(
                    audio,
                    min_silence_len=500,
                    silence_thresh=audio.dBFS - 14,
                    keep_silence=200,
                )
                pure_speech = AudioSegment.empty()
                for part in speech_parts:
                    pure_speech += part

                extracted_from_file = 0
                for i in range(0, len(pure_speech), CHUNK_LENGTH_MS):
                    if total_chunks >= target_chunks:
                        break
                    if extracted_from_file >= chunks_per_file:
                        break

                    chunk = pure_speech[i : i + CHUNK_LENGTH_MS]
                    if len(chunk) < CHUNK_LENGTH_MS:
                        continue

                    chunk_name = f"chunk_{total_chunks:03d}.wav"
                    chunk.export(os.path.join(speaker_out, chunk_name), format="wav")
                    extracted_from_file += 1
                    total_chunks += 1
                    new_this_run += 1

            except Exception as exc:
                logger.error("[%s] Slicing failed for %s: %s", speaker, media_file, exc)
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

        logger.info("[%s] Done — %d new chunks, %d total on disk", speaker, new_this_run, total_chunks)
        report[speaker] = new_this_run

    return report


def create_triton_client() -> InferenceServerClient:
    try:
        client = InferenceServerClient(url=TRITON_URL)
        if not client.is_server_live():
            raise RuntimeError("Triton server is not live")
        if not client.is_server_ready():
            raise RuntimeError("Triton server is not ready")
        if not client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton")
        logger.info("Triton connected at %s | model: %s", TRITON_URL, MODEL_NAME)
        return client
    except InferenceServerException as exc:
        raise RuntimeError(f"Triton connection failed: {exc}") from exc


def get_embedding(wav_path: str, triton_client: InferenceServerClient) -> list[float] | None:
    try:
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        if len(waveform) == 0 or np.abs(waveform).max() < 1e-6:
            return None

        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        embeddings = []

        for i in range(0, len(waveform), TRITON_CHUNK_SZ):
            chunk = waveform[i : i + TRITON_CHUNK_SZ]
            if len(chunk) < TRITON_CHUNK_SZ // 2:
                continue
            if len(chunk) < TRITON_CHUNK_SZ:
                chunk = np.pad(chunk, (0, TRITON_CHUNK_SZ - len(chunk)))
            chunk = chunk.reshape(1, TRITON_CHUNK_SZ).astype(np.float32)

            inputs = [InferInput(INPUT_NAME, chunk.shape, "FP32")]
            inputs[0].set_data_from_numpy(chunk)
            outputs = [InferRequestedOutput(OUTPUT_NAME)]
            response = triton_client.infer(
                model_name=MODEL_NAME,
                inputs=inputs,
                outputs=outputs,
            )
            emb = response.as_numpy(OUTPUT_NAME)
            if emb is not None and emb.size > 0:
                embeddings.append(emb)

        if not embeddings:
            return None
        return np.mean(embeddings, axis=0).flatten().tolist()

    except Exception as exc:
        logger.error("Embedding failed for %s: %s", wav_path, exc)
        return None


def setup_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists", COLLECTION_NAME)


def load_enrolled_chunk_keys(client: QdrantClient) -> set[str]:
    enrolled: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            if point.payload and "speaker" in point.payload and "chunk_file" in point.payload:
                enrolled.add(f"{point.payload['speaker']}/{point.payload['chunk_file']}")
        if offset is None:
            break
    return enrolled


def enroll_chunks_to_qdrant(processed_dir: str, triton_client: InferenceServerClient) -> dict[str, Any]:
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    qdrant = QdrantClient(url=QDRANT_URL)
    logger.info("Qdrant connected at %s", QDRANT_URL)
    setup_collection(qdrant)

    enrolled_keys = load_enrolled_chunk_keys(qdrant)
    logger.info("Chunks already in DB: %d", len(enrolled_keys))

    speakers = sorted(
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    )

    total_enrolled = 0
    total_skipped = 0
    total_failed = 0

    for speaker in speakers:
        speaker_dir = os.path.join(processed_dir, speaker)
        chunk_files = sorted(
            f for f in os.listdir(speaker_dir)
            if f.startswith("chunk_") and f.endswith(".wav")
        )
        if not chunk_files:
            logger.warning("[%s] No chunks found. Skipping.", speaker)
            continue

        new_chunks = [f for f in chunk_files if f"{speaker}/{f}" not in enrolled_keys]
        if not new_chunks:
            logger.info("[SKIP] %s — all %d chunks already in DB", speaker, len(chunk_files))
            total_skipped += len(chunk_files)
            continue

        logger.info("[ENROLLING] %s — %d new / %d total", speaker, len(new_chunks), len(chunk_files))

        points: list[PointStruct] = []
        failed_streak = 0

        for chunk_file in new_chunks:
            wav_path = os.path.join(speaker_dir, chunk_file)
            embedding = get_embedding(wav_path, triton_client)

            if embedding is None:
                total_failed += 1
                failed_streak += 1
                if failed_streak >= 5:
                    raise RuntimeError(
                        "5 consecutive Triton failures — aborting. "
                        "Rerun when Triton is back; resume guard will skip enrolled chunks."
                    )
                continue

            failed_streak = 0
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "speaker": speaker,
                        "chunk_file": chunk_file,
                        "source_dir": speaker_dir,
                    },
                )
            )

        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("  Upserted %d vectors for '%s'", len(points), speaker)
            total_enrolled += len(points)

        total_skipped += len(chunk_files) - len(new_chunks)

    info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "speakers_processed": len(speakers),
        "vectors_enrolled": total_enrolled,
        "chunks_skipped": total_skipped,
        "chunks_failed": total_failed,
        "total_vectors_in_db": info.points_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full enrollment pipeline: raw media → chunks → Qdrant",
    )
    parser.add_argument(
        "--raw",
        default=DEFAULT_RAW_DIR,
        help=f"Raw data root (speaker subfolders). Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--processed",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Output folder for 5s chunks. Default: {DEFAULT_PROCESSED_DIR}",
    )
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=DEFAULT_TARGET_CHUNKS,
        help="Max 5s chunks per speaker (default: 240 = ~20 min)",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip Phase 1 — only enroll existing chunks from --processed",
    )
    parser.add_argument(
        "--skip-enroll",
        action="store_true",
        help="Skip Phase 2 — only extract chunks to --processed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_dir = os.path.abspath(args.raw)
    processed_dir = os.path.abspath(args.processed)

    logger.info("=" * 55)
    logger.info("  ENROLLMENT PIPELINE")
    logger.info("  Raw       : %s", raw_dir)
    logger.info("  Processed : %s", processed_dir)
    logger.info("=" * 55)

    extraction_report: dict[str, int] = {}
    db_summary: dict[str, Any] = {}

    try:
        if not args.skip_process:
            logger.info("PHASE 1: Raw media → 5-second chunks")
            extraction_report = process_raw_to_chunks(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                target_chunks=args.target_chunks,
            )
        else:
            logger.info("PHASE 1: Skipped (--skip-process)")

        if not args.skip_enroll:
            logger.info("PHASE 2: Chunks → Triton → Qdrant")
            triton = create_triton_client()
            db_summary = enroll_chunks_to_qdrant(processed_dir, triton)
        else:
            logger.info("PHASE 2: Skipped (--skip-enroll)")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    print("\n" + "═" * 55)
    print("  PIPELINE COMPLETE")
    if extraction_report:
        print(f"  Phase 1 — speakers processed : {len(extraction_report)}")
        print(f"  Phase 1 — new chunks created : {sum(extraction_report.values())}")
    if db_summary:
        print(f"  Phase 2 — vectors enrolled   : {db_summary['vectors_enrolled']}")
        print(f"  Phase 2 — chunks skipped     : {db_summary['chunks_skipped']}")
        print(f"  Phase 2 — chunks failed      : {db_summary['chunks_failed']}")
        print(f"  Total vectors in Qdrant      : {db_summary['total_vectors_in_db']}")
    print("═" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())
