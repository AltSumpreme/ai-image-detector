from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from data.transforms import get_eval_transforms
from models.efficientnet import build_efficientnet

APP = FastAPI(title="AI Image Detector API", version="0.1.0")
STATE = {
    "model": None,
    "transform": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "threshold": 0.5,
}
IDX_TO_LABEL = {0: "REAL", 1: "AI_GENERATED"}


def load_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=STATE["device"])
    model = build_efficientnet(
        model_name=checkpoint.get("model_name", "efficientnet_b0"),
        num_classes=checkpoint.get("num_classes", 2),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(STATE["device"])
    model.eval()

    STATE["model"] = model
    STATE["transform"] = get_eval_transforms(checkpoint.get("image_size", 224))


@APP.on_event("startup")
def startup_event() -> None:
    checkpoint_path = Path("checkpoints/best_efficientnet.pt")
    try:
        load_checkpoint(checkpoint_path)
    except FileNotFoundError as exc:
        print(f"[startup] Warning: {exc}")


@APP.post("/predict")
async def predict(image: UploadFile = File(...)):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train and place checkpoint first.")

    raw_bytes = await image.read()
    try:
        pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    tensor = STATE["transform"](pil_image).unsqueeze(0).to(STATE["device"])

    with torch.no_grad():
        logits = STATE["model"](tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred = int(pred_idx.item())
    conf = float(confidence.item())

    if pred == 1 and conf < STATE["threshold"]:
        pred = 0

    return {
        "label": IDX_TO_LABEL[pred],
        "confidence": conf,
    }
