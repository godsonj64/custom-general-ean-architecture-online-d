"""Training script for the Evolutionary Abstraction Network (EAN).

Usage:
    python src/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

# Allow running as `python src/train.py` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import build_dataloaders, get_num_classes
from src.model import build_model, compute_loss, EvolutionController
from src.utils import (
    load_config,
    set_seed,
    get_device,
    setup_logging,
    save_checkpoint,
    count_parameters,
    ensure_dir,
)
from src.evaluate import evaluate_loader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the EAN model.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(model, cfg: dict):
    """Build an AdamW optimizer with separate LR groups for encoder vs. rest."""
    lr = cfg["training"]["learning_rate"]
    enc_lr = cfg["training"]["encoder_lr"]
    wd = cfg["training"]["weight_decay"]

    param_groups = [
        {"params": model.get_non_encoder_params(), "lr": lr},
        {"params": model.get_encoder_params(), "lr": enc_lr},
    ]
    return AdamW(param_groups, weight_decay=wd)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg: dict):
    """Build a learning rate scheduler."""
    sched_name = cfg["training"].get("scheduler", "cosine")
    epochs = cfg["training"]["epochs"]
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif sched_name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        return None


# ---------------------------------------------------------------------------
# Single epoch train loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    wm_weight: float,
    logger,
) -> float:
    """Run one training epoch and return the average total loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, update_scores=True)
        loss, _ = compute_loss(outputs, targets, world_model_weight=wm_weight)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    logger = setup_logging()
    set_seed(cfg["project"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    # Prepare output directories
    output_dir = cfg["project"]["output_dir"]
    ensure_dir(output_dir)

    # Build data loaders
    logger.info("Building dataloaders (dataset will be downloaded if not present)...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info(
        f"Dataset '{cfg['dataset']['name']}': "
        f"{len(train_loader.dataset)} train | "
        f"{len(val_loader.dataset)} val | "
        f"{len(test_loader.dataset)} test samples"
    )

    # Build model
    logger.info("Building EAN model...")
    model = build_model(cfg).to(device)
    logger.info(f"Total trainable parameters: {count_parameters(model):,}")

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Evolution controller
    evolution_ctrl = EvolutionController(cfg)

    # Loss weight
    wm_weight = cfg["training"].get("world_model_loss_weight", 0.1)

    # Checkpoint paths
    last_ckpt_path = os.path.join(output_dir, "last_model.pt")
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")

    start_epoch = 1
    best_val_acc = 0.0

    # --- Resume ---
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    total_epochs = cfg["training"]["epochs"]
    top_k_k = cfg["evaluation"]["top_k_accuracy_k"]

    logger.info(f"Starting training for {total_epochs} epochs...")

    for epoch in range(start_epoch, total_epochs + 1):
        # --- Train ---
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, wm_weight, logger)

        # --- Validate ---
        val_metrics = evaluate_loader(model, val_loader, device, cfg, split="val")
        val_acc = val_metrics["accuracy"]

        # --- Scheduler step ---
        if scheduler is not None:
            scheduler.step()

        # --- Evolution step ---
        model.concept_modules = evolution_ctrl.maybe_evolve(
            epoch, model.concept_modules, optimizer, device, logger
        )
        # Re-sync router routing_keys size if modules were replaced
        # (module count stays the same, only weights reset)

        # --- REQUIRED output format (parsed by runner) ---
        print(f"epoch {epoch}/{total_epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")
        sys.stdout.flush()

        # --- Checkpoint ---
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "cfg": cfg,
        }
        save_checkpoint(state, last_ckpt_path, is_best=is_best, best_filepath=best_ckpt_path)

        logger.info(
            f"Epoch {epoch}/{total_epochs} | loss={avg_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"val_top{top_k_k}_acc={val_metrics['top_k_accuracy']:.4f} | "
            f"perplexity={val_metrics['perplexity']:.4f} | "
            f"{'*** BEST ***' if is_best else ''}"
        )

    # --- Final test evaluation ---
    logger.info("Loading best checkpoint for final test evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_loader(model, test_loader, device, cfg, split="test")
    logger.info(
        f"Test Results | "
        f"accuracy={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | "
        f"top_{top_k_k}_accuracy={test_metrics['top_k_accuracy']:.4f} | "
        f"perplexity={test_metrics['perplexity']:.4f}"
    )

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
