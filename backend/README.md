# OOC-Verify Research Backend

FastAPI backend foundation for the Multimodal Out-of-Context (OOC) Misinformation Verification Dashboard.

---

## Purpose

The backend provides a structured API and pipeline orchestrator for verifying whether an image-caption pair is genuine or misleading (out-of-context). It receives paired inputs from the React dashboard, processes them through staged verification steps, and returns an explainable verification report.

> **CURRENT STATUS (CLIP Integration Phase)**:  
> - **Stage 3 (Cross-Modal Alignment)**: Fully operational using `openai/clip-vit-base-patch32` with automatic CUDA GPU acceleration (falling back to CPU). The model is loaded once at server startup and computes real image-caption alignment logits.
> - **Subsequent Stages (4–7)**: MLLM reasoning, external evidence retrieval, and final classification decisions remain architectural stubs. No fake predictions or synthetic confidence percentages are generated; the raw CLIP score is returned directly.

---

## Directory Structure

```text
backend/
├── app/
│   ├── __init__.py               # Package marker and version
│   ├── main.py                   # FastAPI app entrypoint, lifespan startup & CORS
│   ├── config.py                 # Environment and application settings
│   │
│   ├── api/
│   │   ├── __init__.py           # API routes export
│   │   └── routes/
│   │       ├── __init__.py       # Route grouping
│   │       ├── health.py         # GET /api/v1/health status endpoint (with CLIP info)
│   │       └── verification.py   # POST /api/v1/verify multipart endpoint
│   │
│   ├── services/
│   │   ├── __init__.py           # Services export
│   │   └── clip_service.py       # CLIP alignment service (model caching, CUDA, inference)
│   │
│   ├── schemas/
│   │   ├── __init__.py           # Schemas export
│   │   └── verification.py       # Pydantic models for request/response (with clip_score)
│   │
│   └── pipeline/
│       ├── __init__.py           # Pipeline orchestrator export
│       └── orchestrator.py       # Staged verification orchestrator (CLIP + stubs)
│
├── requirements.txt              # Python package dependencies
├── test_clip.py                  # Standalone CLIP verification test script
├── test_api_clip.py              # FastAPI integration test for CLIP verification
└── README.md                     # Documentation & setup instructions
```

---

## Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12 installed.
- **PowerShell** (Windows) or standard bash (Linux/macOS).

---

## Quickstart Setup Guide (Windows PowerShell)

### 1. Create the Virtual Environment

Run from the `backend/` directory or project root:

```powershell
# Navigate to backend
cd "backend"

# Create virtual environment in backend/.venv
python -m venv .venv
```

### 2. Activate the Virtual Environment

```powershell
# In Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# (If running into PowerShell ExecutionPolicy restrictions, run first:)
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the FastAPI Development Server

```powershell
# Run Uvicorn with auto-reload on port 8000
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **Root**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Available Endpoints

### 1. Health Check

- **Method**: `GET`
- **Route**: `/api/v1/health`
- **Description**: Returns backend health and service identifier.
- **Response Example**:
  ```json
  {
    "status": "ok",
    "service": "ooc-verify-backend"
  }
  ```

### 2. Verification Endpoint

- **Method**: `POST`
- **Route**: `/api/v1/verify`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `image` *(File, required)*: Uploaded image file (JPEG, PNG, WEBP; max 10MB).
  - `caption` *(string, required)*: Text caption or claim to evaluate.
  - `dataset` *(string, optional)*: Benchmark dataset (e.g., `NewsCLIPpings`, `VisualNews`, `Custom Pair`).
  - `configuration` *(string, optional)*: Pipeline configuration (`alignment_only`, `mllm_only`, `proposed`).
- **Response Structure**:
  ```json
  {
    "id": "ver-backend-a1b2c3d4",
    "prediction": "Genuine",
    "confidence_score": 0.0,
    "confidenceScore": 0.0,
    "clip_score": 20.617,
    "clipScore": 20.617,
    "alignment_score": 20.617,
    "alignmentScore": 20.617,
    "reason": "CLIP cross-modal alignment score: 20.6170. Pipeline stages beyond alignment remain stubs.",
    "evidence": [],
    "explanation": "Cross-modal semantic alignment was evaluated using CLIP (openai/clip-vit-base-patch32) on NVIDIA GeForce RTX 3050 6GB Laptop GPU. The real image-caption alignment score is 20.6170. Subsequent pipeline stages (MLLM reasoning, evidence retrieval, and final classification) remain architectural stubs as training and classifier models are not yet implemented.",
    "inconsistency_type": "None (CLIP alignment evaluated; classification stage stubbed)",
    "inconsistencyType": "None (CLIP alignment evaluated; classification stage stubbed)",
    "dataset": "NewsCLIPpings",
    "ground_truth_label": null,
    "groundTruthLabel": null,
    "pipeline_stages": [
      { "stage": "input_validation", "status": "completed", "detail": "Validated image and caption input" },
      { "stage": "image_text_analysis", "status": "completed", "detail": "Image decoded to RGB and caption preprocessed" },
      { "stage": "cross_modal_alignment", "status": "completed", "detail": "Real CLIP alignment score: 20.6170", "score": 20.617 },
      { "stage": "mllm_reasoning", "status": "completed", "detail": "Stub: Multimodal LLM reasoning not yet implemented" },
      { "stage": "evidence_retrieval", "status": "completed", "detail": "Stub: External web evidence retrieval not yet implemented" },
      { "stage": "evidence_analysis", "status": "completed", "detail": "Stub: Cross-source evidence reconciliation not yet implemented" },
      { "stage": "final_decision", "status": "completed", "detail": "Stub: Multi-signal classification decision not yet implemented" }
    ],
    "is_stub": true
  }
  ```

---

## Testing the Endpoints (PowerShell Examples)

### Test Health Endpoint

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -Method Get
```

### Test Verification Endpoint with an Image

```powershell
$filePath = "test_image.png"
$url = "http://127.0.0.1:8000/api/v1/verify"

$form = @{
    caption = "Fire crews contain a western ridge blaze after overnight winds."
    dataset = "NewsCLIPpings"
    configuration = "proposed"
    image = Get-Item -Path $filePath
}

$response = Invoke-RestMethod -Uri $url -Method Post -Form $form
$response | ConvertTo-Json -Depth 4
```

---

## Environment Variables

You can configure the backend by setting environment variables or creating a `.env` file in the `backend/` directory:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Primary allowed frontend origin |
| `CORS_ORIGINS` | Comma-separated list | All allowed origins for CORS |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum upload size in megabytes |

---

## Next Planned Development Stages

The architecture established in this milestone prepares the system for step-by-step AI model integration:

1. **Stage 1: Dataset Ingestion**  
   Load and serve benchmark dataset partitions (NewsCLIPpings, VisualNews, etc.) directly through backend storage and streaming endpoints.
2. **Stage 2: Cross-Modal Alignment Model**  
   Implement CLIP / SigLIP embedding similarity checks for multimodal semantic congruence.
3. **Stage 3: MLLM Integration**  
   Integrate local or API-backed vision-language reasoning models (e.g. LLaVA, Gemini Vision, or Qwen2-VL) to extract entities, actions, and temporal/geographical markers.
4. **Stage 4: Evidence Retrieval Engine**  
   Build a reverse-image search and text claim retrieval pipeline using news search APIs and archival databases.
5. **Stage 5: Evidence-Aware Decision Engine**  
   Compare retrieved factual context against image claims to assess support, contradiction, or insufficient context.
6. **Stage 6: Natural Language Explanation Generation**  
   Synthesize readable, structured, and reviewable rationale explaining exactly why an asset is genuine or misleading.
7. **Stage 7: Quantitative Evaluation & Benchmarking**  
   Execute automated confusion matrix, precision, recall, and F1 calculations across the benchmark test splits.
