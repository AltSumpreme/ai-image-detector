# Production Multi-Signal Deepfake Detector (PyTorch)

This project implements a **modular forensic detector** for AI-generated images using multiple independent signals:
- SRM artifact branch
- FFT frequency branch
- DWT multi-scale branch
- ViT semantic branch
- Physical consistency features

## Project Structure

```text
project/
  data/
  models/
  datasets/
  utils/
  engine/
  configs/
  train.py
  eval.py
```

## Strict Dataset Layout

The code expects this exact structure (auto-created at startup):

```text
data/
  train/
    real/
      coco/
    fake/
      progan/
      diffusion/
  val/
    real/
      coco/
    fake/
      progan/
      diffusion/
  test/
    real/
      coco/
    fake/
      unseen_gan/
      unseen_diffusion/
```

> Datasets are **manually downloaded by user** and placed in the folders above.

## Train

```bash
python train.py --config configs/config.yaml
```

- Creates folders if missing.
- Validates dataset presence and image counts.
- Trains in two phases (freeze ViT, then unfreeze last layers).

## Evaluate

```bash
python eval.py --config configs/config.yaml --checkpoint checkpoints/best_multisignal.pt
```

Runs:
- seen evaluation (`val`)
- unseen evaluation (`test` with unseen generators)
- degraded evaluation (JPEG + blur + noise style distortions)
- robust stochastic prediction (5 augmentations averaged)
