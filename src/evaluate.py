"""Evaluation utilities for the EAN project.

Usage (standalone):
    python src/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import load_config, get_device, setup_logging, load_checkpoint


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_loader(
    model,
    loader,
    device: torch.device,
    cfg: dict,
    split: str = "val",
) -> Dict[str, float]:
    """Run inference over a DataLoader and return a metrics dict.

    Metrics computed:
        - accuracy        : top-1 accuracy
        - f1              : macro F1 score
        - top_k_accuracy  : top-k accuracy (k from cfg)
        - perplexity      : exp(mean cross-entropy loss) from the world model proxy
    """
    model.eval()
    k = cfg["evaluation"]["top_k_accuracy_k"]
    num_classes = cfg["dataset"]["num_classes"]

    all_preds: list = []
    all_targets: list = []
    all_probs: list = []
    total_ce_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images, update_scores=False)
            logits = outputs["logits"]  # (B, C)

            # Cross-entropy for perplexity
            ce = F.cross_entropy(logits, targets, reduction="sum")
            total_ce_loss += ce.item()
            n_samples += targets.size(0)

            probs = F.softmax(logits, dim=-1)  # (B, C)
            preds = logits.argmax(dim=-1)       # (B,)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)
    all_probs_np = np.concatenate(all_probs)  # (N, C)

    # Accuracy
    accuracy = float((all_preds_np == all_targets_np).mean())

    # Macro F1
    f1 = float(f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0))

    # Top-k accuracy
    topk_correct = 0
    for i, target in enumerate(all_targets_np):
        top_k_preds = np.argsort(all_probs_np[i])[::-1][:k]
        if target in top_k_preds:
            topk_correct += 1
    top_k_accuracy = topk_correct / max(len(all_targets_np), 1)

    # Perplexity = exp(mean_CE)
    mean_ce = total_ce_loss / max(n_samples, 1)
    perplexity = float(math.exp(min(mean_ce, 500)))  # clamp to avoid overflow

    return {
        "accuracy": accuracy,
        "f1": f1,
        "top_k_accuracy": top_k_accuracy,
        "perplexity": perplexity,
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an EAN checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging()
    device = get_device()

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if args.split == "test" else val_loader

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Checkpoint loaded.")

    metrics = evaluate_loader(model, loader, device, cfg, split=args.split)
    k = cfg["evaluation"]["top_k_accuracy_k"]

    logger.info(f"--- Evaluation Results ({args.split}) ---")
    logger.info(f"  Accuracy       : {metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro)     : {metrics['f1']:.4f}")
    logger.info(f"  Top-{k} Accuracy: {metrics['top_k_accuracy']:.4f}")
    logger.info(f"  Perplexity     : {metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()
