"""Unit tests for the EAN model forward pass and related components."""

from __future__ import annotations

import sys
import os
import math
import copy

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import (
    EAN,
    SmallCNNEncoder,
    EfficientNetEncoder,
    AbstractionField,
    ConceptModule,
    ConceptRouter,
    LatentWorldModel,
    OutputHead,
    EvolutionController,
    build_model,
    compute_loss,
)
from src.utils import load_config


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

MINIMAL_CFG = {
    "dataset": {"num_classes": 10, "image_size": 32},
    "model": {
        "encoder": "cnn_baseline",
        "encoder_pretrained": False,
        "latent_dim": 64,
        "abstraction_dim": 32,
        "num_concept_modules": 4,
        "top_k": 2,
        "concept_hidden_dim": 64,
        "world_model_hidden_dim": 64,
        "dropout": 0.0,
    },
    "training": {
        "evolution": {
            "enabled": True,
            "prune_every_n_epochs": 2,
            "prune_fraction": 0.25,
            "score_ema_alpha": 0.9,
        }
    },
}

DEVICE = torch.device("cpu")
BATCH = 4
C, H, W = 3, 32, 32


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_dummy_batch(batch=BATCH):
    images = torch.randn(batch, C, H, W)
    targets = torch.randint(0, MINIMAL_CFG["dataset"]["num_classes"], (batch,))
    return images, targets


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------

class TestSmallCNNEncoder:
    def test_output_shape(self):
        enc = SmallCNNEncoder(latent_dim=64)
        x = torch.randn(BATCH, 3, 32, 32)
        z = enc(x)
        assert z.shape == (BATCH, 64), f"Expected ({BATCH}, 64), got {z.shape}"

    def test_output_finite(self):
        enc = SmallCNNEncoder(latent_dim=64)
        x = torch.randn(BATCH, 3, 32, 32)
        z = enc(x)
        assert torch.isfinite(z).all(), "Encoder output contains NaN or Inf"


class TestAbstractionField:
    def test_output_shape(self):
        af = AbstractionField(latent_dim=64, abstraction_dim=32)
        z = torch.randn(BATCH, 64)
        a = af(z)
        assert a.shape == (BATCH, 32)


class TestConceptModule:
    def test_residual_shape(self):
        mod = ConceptModule(abstraction_dim=32, hidden_dim=64)
        a = torch.randn(BATCH, 32)
        out = mod(a)
        assert out.shape == (BATCH, 32)


class TestConceptRouter:
    def test_output_shape(self):
        n_mods = 4
        router = ConceptRouter(abstraction_dim=32, num_modules=n_mods, top_k=2)
        modules = torch.nn.ModuleList(
            [ConceptModule(32, 64) for _ in range(n_mods)]
        )
        a = torch.randn(BATCH, 32)
        agg, weights = router(a, modules, update_scores=False)
        assert agg.shape == (BATCH, 32), f"agg shape: {agg.shape}"
        assert weights.shape == (BATCH, n_mods), f"weights shape: {weights.shape}"

    def test_routing_weights_sum_to_one(self):
        n_mods = 4
        router = ConceptRouter(abstraction_dim=32, num_modules=n_mods, top_k=2)
        modules = torch.nn.ModuleList(
            [ConceptModule(32, 64) for _ in range(n_mods)]
        )
        a = torch.randn(BATCH, 32)
        _, weights = router(a, modules, update_scores=False)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


class TestLatentWorldModel:
    def test_output_shape(self):
        wm = LatentWorldModel(latent_dim=64, hidden_dim=64)
        z = torch.randn(BATCH, 64)
        agg = torch.randn(BATCH, 64)
        z_pred = wm(z, agg)
        assert z_pred.shape == (BATCH, 64)


class TestOutputHead:
    def test_output_shape(self):
        head = OutputHead(abstraction_dim=32, num_classes=10)
        agg = torch.randn(BATCH, 32)
        logits = head(agg)
        assert logits.shape == (BATCH, 10)


# ---------------------------------------------------------------------------
# Full EAN forward pass tests
# ---------------------------------------------------------------------------

class TestEANForward:
    def setup_method(self):
        self.model = build_model(MINIMAL_CFG).to(DEVICE)
        self.model.eval()

    def test_logits_shape(self):
        images, _ = make_dummy_batch()
        with torch.no_grad():
            out = self.model(images)
        assert out["logits"].shape == (BATCH, 10)

    def test_z_shape(self):
        images, _ = make_dummy_batch()
        with torch.no_grad():
            out = self.model(images)
        assert out["z"].shape == (BATCH, 64)

    def test_z_pred_shape(self):
        images, _ = make_dummy_batch()
        with torch.no_grad():
            out = self.model(images)
        assert out["z_pred"].shape == (BATCH, 64)

    def test_routing_weights_shape(self):
        images, _ = make_dummy_batch()
        with torch.no_grad():
            out = self.model(images)
        assert out["routing_weights"].shape == (BATCH, 4)  # num_concept_modules=4

    def test_all_outputs_finite(self):
        images, _ = make_dummy_batch()
        with torch.no_grad():
            out = self.model(images)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"Output '{k}' contains NaN or Inf"

    def test_train_mode_gradient_flow(self):
        """Check that gradients flow through the full model."""
        model = build_model(MINIMAL_CFG).to(DEVICE)
        model.train()
        images, targets = make_dummy_batch()
        out = model(images)
        loss, _ = compute_loss(out, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_loss_positive(self):
        model = build_model(MINIMAL_CFG)
        model.eval()
        images, targets = make_dummy_batch()
        with torch.no_grad():
            out = model(images)
        loss, breakdown = compute_loss(out, targets)
        assert loss.item() > 0
        assert "cls_loss" in breakdown
        assert "wm_loss" in breakdown

    def test_loss_finite(self):
        model = build_model(MINIMAL_CFG)
        model.eval()
        images, targets = make_dummy_batch()
        with torch.no_grad():
            out = model(images)
        loss, _ = compute_loss(out, targets)
        assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# Evolution controller test
# ---------------------------------------------------------------------------

class TestEvolutionController:
    def test_prune_and_replace(self):
        """Evolution controller should replace weakest modules without changing count."""
        import torch.nn as nn
        model = build_model(MINIMAL_CFG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctrl = EvolutionController(MINIMAL_CFG)

        n_before = len(model.concept_modules)
        # Force a prune epoch (prune_every=2, so epoch=2 triggers)
        model.concept_modules = ctrl.maybe_evolve(
            epoch=2,
            concept_modules=model.concept_modules,
            optimizer=optimizer,
            device=DEVICE,
        )
        n_after = len(model.concept_modules)
        assert n_before == n_after, "Module count should stay the same after evolution"

    def test_no_prune_on_non_trigger_epoch(self):
        """Evolution controller should not prune on non-trigger epochs."""
        model = build_model(MINIMAL_CFG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctrl = EvolutionController(MINIMAL_CFG)

        original_ids = [id(m) for m in model.concept_modules]
        model.concept_modules = ctrl.maybe_evolve(
            epoch=1,  # prune_every=2, so epoch=1 should NOT trigger
            concept_modules=model.concept_modules,
            optimizer=optimizer,
            device=DEVICE,
        )
        new_ids = [id(m) for m in model.concept_modules]
        assert original_ids == new_ids, "Modules should not change on non-trigger epoch"


# ---------------------------------------------------------------------------
# Config loading test
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_load_default_config(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        if os.path.exists(cfg_path):
            cfg = load_config(cfg_path)
            assert "model" in cfg
            assert "dataset" in cfg
            assert "training" in cfg


if __name__ == "__main__":
    pytest.main(["-v", __file__])
