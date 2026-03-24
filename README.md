# AI Image Detector (MVP)

Production-leaning prototype to classify images as **REAL** vs **AI_GENERATED**.

## Folder Structure

```text
ai-image-detector/
├── api/
│   └── main.py
├── data/
│   ├── dataset.py
│   └── transforms.py
├── features/
│   └── fft.py
├── models/
│   └── efficientnet.py
├── training/
│   ├── eval.py
│   └── train.py
├── utils/
│   ├── config.py
│   └── metrics.py
├── requirements.txt
└── README.md
```

## Dataset Layout

```text
dataset/
├── real/
└── ai/
```

Target size for MVP: around **10,000 images total**, balanced (5k real / 5k AI).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m training.train --dataset-root dataset --epochs 12 --batch-size 32
```

Artifacts:
- Best model: `checkpoints/best_efficientnet.pt`
- Epoch logs include loss, accuracy, precision, recall.

## Evaluate

```bash
python -m training.eval --dataset-root dataset --checkpoint checkpoints/best_efficientnet.pt
```

Outputs:
- Accuracy
- Precision
- Recall
- Confusion matrix

## Run API

```bash
uvicorn api.main:APP --host 0.0.0.0 --port 8000
```

## Inference Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sample.jpg"
```

Expected response:

```json
{
  "label": "AI_GENERATED",
  "confidence": 0.93
}
```

## Extensibility Notes

This MVP is structured for future research upgrades:
- `features/fft.py` provides a frequency feature extraction hook.
- `utils/config.py` centralizes tunables for easier experiment control.
- Model builder (`models/efficientnet.py`) allows model swaps.
- API threshold behavior can be tuned (`STATE["threshold"]`).

Planned upgrades:
1. FFT + RGB feature fusion.
2. CLIP-based scoring module.
3. Ensemble inference.
4. Patch-level detection.
5. Continuous retraining pipeline.
