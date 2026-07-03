import subprocess
import numpy as np
import librosa
from pydub import AudioSegment

from config import SAMPLE_RATE, CHUNK_SIZE, MIN_AUDIO_MS, CLIP_DURATION_MS


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Converts any uploaded audio/video file to 16kHz mono WAV.
    Trims to first CLIP_DURATION_MS (5 seconds).
    Rejects files shorter than MIN_AUDIO_MS (2 seconds).

    Returns True on success, False on any failure.
    """
    try:
        # Use ffmpeg to extract/convert audio
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        audio = AudioSegment.from_wav(output_path)

        # Guard: reject clips too short for ECAPA-TDNN to process meaningfully
        if len(audio) < MIN_AUDIO_MS:
            print(f"[audio_service]  Audio too short ({len(audio)}ms). Min: {MIN_AUDIO_MS}ms")
            return False

        # Trim to first 5 seconds
        clipped = audio[:CLIP_DURATION_MS]
        clipped.export(output_path, format="wav")
        return True

    except Exception as e:
        print(f"[audio_service]  Conversion failed: {e}")
        return False


def load_waveform(wav_path: str) -> np.ndarray | None:
    """
    Loads a WAV file and returns a normalized float32 numpy waveform.
    Returns None if the file is empty or silent.
    """
    try:
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) == 0:
            print(f"[audio_service]  Empty audio: {wav_path}")
            return None

        if np.abs(waveform).max() < 1e-6:
            print(f"[audio_service]  Silent audio: {wav_path}")
            return None

        # Normalize to [-1, 1]
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform

    except Exception as e:
        print(f"[audio_service]  Load failed: {e}")
        return None


def slice_into_chunks(waveform: np.ndarray) -> list[np.ndarray]:
    """
    Slices a waveform into CHUNK_SIZE (1-second) segments.
    Pads the last chunk if needed. Skips chunks shorter than 0.5 seconds.

    Returns a list of (1, CHUNK_SIZE) float32 arrays ready for Triton.
    """
    chunks = []
    for i in range(0, len(waveform), CHUNK_SIZE):
        chunk = waveform[i:i + CHUNK_SIZE]

        # Skip very short tail (less than 0.5 sec)
        if len(chunk) < CHUNK_SIZE // 2:
            continue

        # Pad if slightly under 1 second
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

        # Shape Triton expects: (1, 16000)
        chunks.append(chunk.reshape(1, CHUNK_SIZE).astype(np.float32))

    return chunks
