# Custom General EAN Architecture (Online Dataset)

## Overview
This project implements a fully custom **Evolutionary Abstraction Network (EAN)** — a modular, dynamic neural architecture for image classification. The EAN compresses input images through an **Encoder** into a latent representation `z`, routes it through an **Abstraction Field** and a **Concept Router** (Top-k selection), aggregates outputs from dynamically evolving concept modules, and feeds them into an **Output Head** paired with a **Latent World Model** for predictive reasoning. An **Evolution Controller** continuously updates and prunes concept modules over training, making the model self-improving.

The baseline model is a small CNN trained from scratch, and the recommended backbone uses EfficientNet-B0 transfer learning as the encoder.

**Dataset:** CIFAR-100 (downloaded automatically at training time — no manual upload required).

---

## Architecture Components
| Component | Description |
|---|---|
| Encoder | EfficientNet-B0 (pretrained) or small CNN backbone |
| Abstraction Field | MLP that projects latent z into abstraction space |
| Concept Router | Computes routing scores; selects Top-k concept modules |
| Concept Modules | Dynamically evolving population of small MLPs |
| Evolution Controller | Prunes weak modules; spawns new ones each epoch |
| Latent World Model | Predicts next latent state for auxiliary loss |
| Output Head | Final classification layer |

---

## Metrics
- Accuracy
- F1 Score (macro)
- Top-K Accuracy (k=5)
- Perplexity (from auxiliary world model loss)

## Export Formats
- ONNX
- TorchScript
- PyTorch State Dict

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
bash scripts/run_train.sh
```
Or manually:
```bash
python src/train.py --config configs/default.yaml
```

### 3. Evaluate
```bash
python src/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

### 4. Export
```bash
python src/export.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

---

## Docker
```bash
docker build -t ean-classifier .
docker run --gpus all -v $(pwd)/outputs:/app/outputs ean-classifier
```

---

## Configuration
All hyperparameters are in `configs/default.yaml`. Key options:
- `model.num_concept_modules` — number of concept modules in the population
- `model.top_k` — how many modules the router selects per forward pass
- `model.latent_dim` — size of the latent representation z
- `training.epochs` — number of training epochs (default: 20)
- `dataset.name` — online dataset name (default: cifar100)

---

## Project Structure
```
.
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── outputs/
├── scripts/
│   └── run_train.sh
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── utils.py
├── tests/
│   └── test_model_forward.py
├── Dockerfile
├── requirements.txt
└── README.md
```
