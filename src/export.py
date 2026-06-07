"""Model export script for the EAN project.

Exports the trained model to:
  1. ONNX
  2. TorchScript (torch.jit.trace)
  3. PyTorch state dict (.pt)

Usage:
    python src/export.py --config configs/default.yaml --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_model
from src.utils import load_config, get_device, setup_logging, load_checkpoint, ensure_dir


# ---------------------------------------------------------------------------
# Wrapper for export (fixed routing, no score updates)
# ---------------------------------------------------------------------------

class EANExportWrapper(torch.nn.Module):
    """Thin wrapper around EAN that returns only logits for tracing/ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x, update_scores=False)
        return outputs["logits"]


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_state_dict(model, export_dir: str, logger) -> str:
    """Save the raw PyTorch state dict."""
    path = os.path.join(export_dir, "ean_state_dict.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"[Export] State dict saved to: {path}")
    return path


def export_torchscript(wrapper, dummy_input: torch.Tensor, export_dir: str, logger) -> str:
    """Trace the model and save as TorchScript."""
    path = os.path.join(export_dir, "ean_torchscript.pt")
    try:
        traced = torch.jit.trace(wrapper, dummy_input, strict=False)
        torch.jit.save(traced, path)
        logger.info(f"[Export] TorchScript saved to: {path}")
    except Exception as e:
        logger.warning(f"[Export] TorchScript export failed: {e}")
        path = None
    return path


def export_onnx(
    wrapper,
    dummy_input: torch.Tensor,
    export_dir: str,
    opset: int,
    input_names: list,
    output_names: list,
    logger,
) -> str:
    """Export the model to ONNX format."""
    path = os.path.join(export_dir, "ean_model.onnx")
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        logger.info(f"[Export] ONNX model saved to: {path}")

        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        logger.info("[Export] ONNX model verification passed.")
    except Exception as e:
        logger.warning(f"[Export] ONNX export failed: {e}")
        path = None
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Export an EAN checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging()
    device = get_device()

    export_cfg = cfg["export"]
    export_dir = export_cfg["export_dir"]
    ensure_dir(export_dir)

    input_shape = export_cfg["input_shape"]
    opset = export_cfg["onnx_opset"]

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    wrapper = EANExportWrapper(model).to(device)
    wrapper.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # 1. State dict
    export_state_dict(model, export_dir, logger)

    # 2. TorchScript
    export_torchscript(wrapper, dummy_input, export_dir, logger)

    # 3. ONNX
    export_onnx(
        wrapper,
        dummy_input,
        export_dir,
        opset=opset,
        input_names=["input_image"],
        output_names=["class_logits"],
        logger=logger,
    )

    logger.info(f"All exports written to: {export_dir}")


if __name__ == "__main__":
    main()
