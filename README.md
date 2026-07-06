# sr_project

Speaker identification API using ECAPA-TDNN (Triton) + Qdrant vector search.

## Architecture

```
Client  →  FastAPI (api/)  →  Triton (embeddings)
                          →  Qdrant (speaker vectors)
```

- **Identify**: upload audio/video → embedding → search `speakers` collection
- **Enroll**: ingest speaker chunks into Qdrant (production starts empty)

## What is in this repo

| Path | Purpose |
|------|---------|
| `api/` | FastAPI app (`/identify`, `/enroll`, `/health`) |
| `scripts/` | Offline enrollment pipeline |
| `.env.example` | Environment variable template |
| `docker-compose.yml` | API + Qdrant for production/staging |

**Not in git / not in Docker image:** `qdrant_data/` — local dev only. Production Qdrant starts **empty** and speakers are enrolled on the server.

## Quick start (Docker — recommended for production)

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — at minimum set:

```env
TRITON_URL=your-triton-host:8040
```

`QDRANT_URL` is overridden to `http://qdrant:6333` inside Docker Compose.

### 2. Build and run

```bash
docker compose up -d --build
```

Services:

| Service | URL |
|---------|-----|
| API | `http://localhost:8080` |
| Qdrant | `http://localhost:6333` |
| Health | `http://localhost:8080/health` |

### 3. Enroll speakers (production — from zero)

Qdrant has no data on first deploy. After API and Qdrant are healthy, enroll speakers:

**Option A — API** (processed chunks on server):

```bash
curl -X POST http://localhost:8080/enroll \
  -H "Content-Type: application/json" \
  -d '{"processed_dir": "/path/to/processed_dataset"}'
```

Check job status: `GET /enroll/status/{job_id}`

**Option B — CLI script** (raw media → chunks → Qdrant):

```bash
# Run inside the api container or locally with venv + ffmpeg
python scripts/enroll_pipeline.py \
  --raw ./raw_data \
  --processed ./processed_dataset
```

Expected layout:

```
processed_dataset/
  SpeakerName/
    chunk_000.wav
    chunk_001.wav
```

### 4. Identify a speaker

```bash
curl -X POST http://localhost:8080/identify \
  -F "file=@sample.wav"
```

Accepted formats: `.wav`, `.mp3`, `.mp4`, `.mkv`, `.avi`, `.mov` (configurable via `ALLOWED_EXTENSIONS`).

## Local development (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Qdrant separately (Docker)
docker run -d --name qdrant -p 6333:6333 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant

cp .env.example .env
cd api
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Environment variables

See `.env.example` for the full list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API bind host | `0.0.0.0` |
| `API_PORT` | API port | `8080` |
| `QDRANT_URL` | Qdrant HTTP URL | `http://localhost:6333` |
| `TRITON_URL` | Triton inference server | **required in prod** |
| `COLLECTION_NAME` | Qdrant collection | `speakers` |
| `MODEL_NAME` | Triton model name | `ecapa_speaker_verification` |
| `REJECT_THRESHOLD` | Min confidence to accept match | `0.60` |
| `FALLBACK_MIN_SCORE` | Low-confidence fallback threshold | `0.50` |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Triton + Qdrant status, vector count |
| `POST` | `/identify` | Upload audio → speaker name |
| `POST` | `/enroll` | Start background enrollment job |
| `GET` | `/enroll/status/{job_id}` | Enrollment job progress |

## Production notes

- **Do not ship `qdrant_data/`** — it is gitignored and excluded from the Docker image.
- Qdrant uses a **named Docker volume** (`qdrant_data`) — empty on first run, persists across restarts.
- **Triton** runs outside this stack — point `TRITON_URL` at your inference server.
- **ffmpeg** is included in the API image for audio/video conversion.
- Re-enroll or add speakers anytime via `/enroll` or `scripts/enroll_pipeline.py`.

## Build API image only

```bash
docker build -t sr-project-api .
docker run -p 8080:8080 --env-file .env sr-project-api
```

When running the API container alone, set `QDRANT_URL` to your Qdrant instance.
