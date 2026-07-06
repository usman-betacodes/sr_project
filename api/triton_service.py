import numpy as np
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
    InferenceServerException
)

from config import TRITON_URL, MODEL_NAME, INPUT_NAME, OUTPUT_NAME
from audio_service import load_waveform, slice_into_chunks


def create_triton_client() -> InferenceServerClient:
    """
    Creates and validates a Triton client connection.
    Raises RuntimeError loudly if server or model is not ready.
    Called once at app startup — reused for all requests.
    """
    try:
        client = InferenceServerClient(url=TRITON_URL)

        if not client.is_server_live():
            raise RuntimeError("Triton server is not live")
        if not client.is_server_ready():
            raise RuntimeError("Triton server is not ready")
        if not client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton")

        print(f"[triton_service] ✅ Connected to Triton at {TRITON_URL} | model: {MODEL_NAME}")
        return client

    except Exception as e:
        raise RuntimeError(f"Triton connection failed: {e}") from e


def infer_single_chunk(client: InferenceServerClient, chunk: np.ndarray) -> np.ndarray | None:
    """
    Sends a single (1, 16000) chunk to Triton and returns the raw embedding.
    Returns None on failure (logged as warning, not crash).
    """
    try:
        inputs = [InferInput(INPUT_NAME, chunk.shape, "FP32")]
        inputs[0].set_data_from_numpy(chunk)
        outputs = [InferRequestedOutput(OUTPUT_NAME)]

        response = client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        return response.as_numpy(OUTPUT_NAME)

    except Exception as e:
        print(f"[triton_service] ⚠️  Chunk inference failed: {e}")
        return None


def get_embedding(wav_path: str, client: InferenceServerClient) -> list[float] | None:
    """
    Full pipeline: load WAV → slice → infer each chunk → average → flatten.

    Returns a flat 192-dim list ready for Qdrant, or None on failure.
    """
    waveform = load_waveform(wav_path)
    if waveform is None:
        return None

    chunks = slice_into_chunks(waveform)
    if not chunks:
        print(f"[triton_service] ❌ No valid chunks from: {wav_path}")
        return None

    embeddings = []
    for chunk in chunks:
        emb = infer_single_chunk(client, chunk)
        if emb is not None and emb.size > 0:
            embeddings.append(emb)

    if not embeddings:
        print(f"[triton_service] ❌ All chunks failed for: {wav_path}")
        return None

    # Average all chunk embeddings → flatten (1,1,192) → (192,) → plain list
    final = np.mean(embeddings, axis=0).flatten().tolist()
    return final
