import io
import os
import re
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import (
    FastAPI, File, Form, Header,
    HTTPException, UploadFile
)
from fastapi.concurrency import run_in_threadpool

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

MODEL_PATH = os.getenv("MODEL_PATH", "week6_model.pt")

# All 4 classes from your Week 6 model checkpoint
CLASSES = [
    "diseased cotton leaf",
    "diseased cotton plant",
    "fresh cotton leaf",
    "fresh cotton plant"
]

# Classes considered diseased for referral logic
DISEASE_CLASSES = [
    "diseased cotton leaf",
    "diseased cotton plant"
]

THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

API_KEY = os.getenv("API_KEY")

MAX_BYTES = 5 * 1024 * 1024

URGENT = [
    "brown lesions",
    "lesions",
    "spots",
    "wilting",
    "curling",
    "blight",
    "rot",
    "yellowing",
    "outbreak",
    "spreading",
    "virus",
    "whitefly",
    "severe",
    "loss"
]

PATTERNS = {
    "cnic": r"\b\d{5}-\d{7}-\d\b",
    "mobile": r"(?:\+92[- ]?|\b0)3\d{2}[- ]?\d{7}\b",
    "amount": r"(?:PKR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)",
    "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"
}

state = {}


# ==========================================
# 2. REBUILD MODEL
# ==========================================

def rebuild(ck):
    name = ck["backbone"]
    k = len(ck["classes"])

    if name.startswith("resnet"):
        model = getattr(models, name)(weights=None)
        model.fc = nn.Linear(model.fc.in_features, k)
    elif name.startswith("efficientnet"):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, k)
    else:
        raise ValueError("Unsupported backbone")

    model.load_state_dict(ck["state_dict"])
    return model.eval()


# ==========================================
# 3. LOAD MODEL ONCE (LIFESPAN)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    ck = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state["ck"] = ck
    state["model"] = rebuild(ck)

    state["tf"] = transforms.Compose(
        ([transforms.Grayscale(3)] if ck["grayscale"] else [])
        + [
            transforms.Resize((ck["img_size"],) * 2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    yield
    state.clear()


app = FastAPI(
    title="Cotton Disease Vision + Text API",
    version="1.0",
    lifespan=lifespan
)


# ==========================================
# 4. RESPONSE SCHEMAS
# ==========================================

class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    latency_ms: float


class CaseRecord(BaseModel):
    prediction: Prediction
    note_masked: str
    fields: dict[str, list[str]]
    urgent_terms: list[str]
    decision: str
    reasons: list[str]
    disclaimer: str


# ==========================================
# 5. API KEY CHECK
# ==========================================

def check_key(x_api_key: str | None = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


# ==========================================
# 6. IMAGE VALIDATION
# ==========================================

async def read_image(file: UploadFile):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image"
        )

    data = await file.read()

    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image larger than 5 MB"
        )

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="File is not a readable image"
        )


# ==========================================
# 7. IMAGE CLASSIFICATION
# ==========================================

def classify(img):
    ck = state["ck"]
    t0 = time.perf_counter()

    x = state["tf"](
        img.convert("L" if ck["grayscale"] else "RGB")
    )

    with torch.inference_mode():
        p = F.softmax(
            state["model"](x.unsqueeze(0)),
            dim=1
        )[0].tolist()

    probs = {
        c: round(v, 4)
        for c, v in zip(ck["classes"], p)
    }

    return Prediction(
        label=max(probs, key=probs.get),
        probabilities=probs,
        latency_ms=round(
            (time.perf_counter() - t0) * 1000,
            1
        )
    )


# ==========================================
# 8. MASK PERSONAL INFORMATION
# ==========================================

def mask_pii(text: str) -> str:
    for key, tag in [
        ("cnic", "[CNIC]"),
        ("mobile", "[PHONE]")
    ]:
        text = re.sub(
            PATTERNS[key],
            tag,
            text,
            flags=re.IGNORECASE
        )
    return text


# ==========================================
# 9. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():
    if "ck" not in state:
        return {
            "status": "starting",
            "model_loaded": False
        }

    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "classes": state["ck"]["classes"],
        "disease_classes": DISEASE_CLASSES
    }


# ==========================================
# 10. PREDICT ENDPOINT
# ==========================================

@app.post("/predict", response_model=Prediction)
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(None)
):
    check_key(x_api_key)
    img = await read_image(file)
    return await run_in_threadpool(classify, img)


# ==========================================
# 11. ANALYZE ENDPOINT
# ==========================================

@app.post("/analyze", response_model=CaseRecord)
async def analyze(
    file: UploadFile = File(...),
    note: str = Form(...),
    x_api_key: str | None = Header(None)
):
    check_key(x_api_key)

    if len(note) > 2000:
        raise HTTPException(
            status_code=422,
            detail="Note longer than 2,000 characters"
        )

    img = await read_image(file)
    pred = await run_in_threadpool(classify, img)

    # Mask PII first
    safe = mask_pii(note)

    fields = {
        k: re.findall(
            p,
            safe,
            flags=re.IGNORECASE
        )
        for k, p in PATTERNS.items()
    }

    urgent = [
        w for w in URGENT
        if w in safe.lower()
    ]

    # Combine probabilities for disease classes
    disease_probability = sum(
        pred.probabilities.get(c, 0.0)
        for c in DISEASE_CLASSES
    )

    reasons = []

    if disease_probability >= THRESHOLD:
        reasons.append(
            "image: combined disease probability = "
            f"{disease_probability:.2f}"
        )

    if urgent:
        reasons.append(
            "note mentions: " + ", ".join(urgent)
        )

    return CaseRecord(
        prediction=pred,
        note_masked=safe,
        fields={
            k: v for k, v in fields.items() if v
        },
        urgent_terms=urgent,
        decision="REFER" if reasons else "ROUTINE",
        reasons=reasons,
        disclaimer=(
            "Educational decision support only; "
            "a qualified agricultural expert "
            "should make the final decision."
        )
    )