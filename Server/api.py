from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import json

# Load env vars (reuses the same environment variables as bug_classifier)
load_dotenv()

# Import classifier (should be import-safe now)
try:
    # batch_classify is the new async batch classifier; keep classify_bug for single-item fallback
    from .bug_classifier import batch_classify, classify_bug
except Exception:
    # fallback to top-level import if running as script in different working dir
    from bug_classifier import batch_classify, classify_bug

app = FastAPI(title="BugClassifier API", version="0.1")

# Allow the Web frontend (served from localhost:5173) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    text: str


class ClassifyItem(BaseModel):
    text: str
    label: str
    raw: str
    team: str | None = None


class ClassifyResponse(BaseModel):
    results: list[ClassifyItem]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    # Split input into non-empty lines; treat each line as a separate bug report
    lines = [l.strip() for l in req.text.splitlines() if l.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="no bug lines found in text")

    try:
        classified = await batch_classify(lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classification error: {e}")

    results = []
    for i, l in enumerate(lines):
        res = classified[i] if i < len(classified) else None
        if isinstance(res, dict):
            label = res.get('label') or ''
            raw = res.get('reason') or json.dumps(res)
            team = res.get('team')
        else:
            label = res or ''
            raw = json.dumps(res)
            team = None
        results.append({"text": l, "label": label, "raw": raw, "team": team})

    return {"results": results}

# uvicorn entrypoint hint
# run with: uvicorn Server.api:app --reload --port 8000
