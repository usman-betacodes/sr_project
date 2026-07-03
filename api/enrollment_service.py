import os
import uuid
import logging
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tritonclient.http import InferenceServerClient

from config import COLLECTION_NAME, VECTOR_SIZE
from triton_service import get_embedding

logger = logging.getLogger("enrollment")


def setup_collection(client: QdrantClient) -> None:
    """Creates the Qdrant collection if it doesn't exist yet."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info("Collection '%s' already exists.", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Collection '%s' created.", COLLECTION_NAME)


def _load_enrolled_chunks(client: QdrantClient) -> set[str]:
    """Returns chunk keys already in DB: 'SpeakerName/chunk_001.wav'."""
    enrolled_chunks: set[str] = set()
    offset = None

    while True:
        scroll_result, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5000,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in scroll_result:
            if point.payload and "speaker" in point.payload and "chunk_file" in point.payload:
                chunk_key = f"{point.payload['speaker']}/{point.payload['chunk_file']}"
                enrolled_chunks.add(chunk_key)

        if offset is None:
            break

    return enrolled_chunks


def enroll_all_speakers(
    processed_dir: str,
    triton_client: InferenceServerClient,
    qdrant_client: QdrantClient,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """
    Walk processed_dir (speaker folders with chunk_*.wav files),
    embed each new chunk via Triton, upsert to Qdrant.

    on_progress: optional callback receiving status updates during the run.
    """
    processed_dir = os.path.abspath(processed_dir)

    if not os.path.isdir(processed_dir):
        raise ValueError(f"Processed dataset not found: {processed_dir}")

    setup_collection(qdrant_client)
    enrolled_chunks = _load_enrolled_chunks(qdrant_client)
    logger.info("Chunks already in DB: %d", len(enrolled_chunks))

    speakers = sorted([
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    ])

    total_enrolled = 0
    total_skipped = 0
    total_failed = 0

    for speaker_idx, speaker in enumerate(speakers, start=1):
        speaker_dir = os.path.join(processed_dir, speaker)
        chunk_files = sorted([
            f for f in os.listdir(speaker_dir)
            if f.startswith("chunk_") and f.endswith(".wav")
        ])

        if on_progress:
            on_progress({
                "current_speaker": speaker,
                "speakers_done": speaker_idx - 1,
                "speakers_total": len(speakers),
                "vectors_enrolled": total_enrolled,
                "chunks_skipped": total_skipped,
                "chunks_failed": total_failed,
            })

        if not chunk_files:
            logger.warning("No chunks found for %s. Skipping.", speaker)
            continue

        new_chunks = [
            f for f in chunk_files
            if f"{speaker}/{f}" not in enrolled_chunks
        ]

        if not new_chunks:
            logger.info("[SKIP] %s — all %d chunks already in DB.", speaker, len(chunk_files))
            total_skipped += len(chunk_files)
            continue

        logger.info("[ENROLLING] %s — %d new / %d total", speaker, len(new_chunks), len(chunk_files))

        speaker_points = []
        failed_chunks = 0

        for chunk_file in new_chunks:
            wav_path = os.path.join(speaker_dir, chunk_file)
            embedding = get_embedding(wav_path, triton_client)

            if embedding is None:
                failed_chunks += 1
                if failed_chunks >= 5:
                    raise RuntimeError(
                        "5 consecutive Triton failures. "
                        "Check server health and rerun — resume guard will continue."
                    )
                continue

            speaker_points.append(
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

        if speaker_points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=speaker_points,
            )
            logger.info("Enrolled %d vectors for '%s'", len(speaker_points), speaker)
            total_enrolled += len(speaker_points)

        if failed_chunks:
            logger.warning("%d chunks failed for '%s'", failed_chunks, speaker)
            total_failed += failed_chunks

        total_skipped += len(chunk_files) - len(new_chunks)

    collection_info = qdrant_client.get_collection(COLLECTION_NAME)

    return {
        "processed_dir": processed_dir,
        "speakers_processed": len(speakers),
        "vectors_enrolled": total_enrolled,
        "chunks_skipped": total_skipped,
        "chunks_failed": total_failed,
        "total_vectors_in_db": collection_info.points_count,
    }
