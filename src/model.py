"""Evolutionary Abstraction Network (EAN) model definition.

Architecture flow:
  image → Encoder → z (latent)
       → Abstraction Field → a (abstraction)
       → Concept Router  → top-k Concept Modules → aggregated representation
       → Output Head     → class logits
       → Latent World Model (auxiliary: predicts next z)

The Evolution Controller prunes weak concept modules and spawns replacements
at the end of designated epochs.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Small CNN baseline encoder
# ---------------------------------------------------------------------------

class SmallCNNEncoder(nn.Module):
    """A lightweight CNN encoder used as the baseline (trained from scratch)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.out_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


# ---------------------------------------------------------------------------
# EfficientNet-B0 encoder (timm)
# ---------------------------------------------------------------------------

class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder using timm; projects features to latent_dim."""

    def __init__(self, latent_dim: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,   # remove classification head
            global_pool="avg",
        )
        encoder_out_dim = self.backbone.num_features  # 1280 for EfficientNet-B0
        self.proj = nn.Sequential(
            nn.Linear(encoder_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.out_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # (B, 1280)
        return self.proj(feats)            # (B, latent_dim)


# ---------------------------------------------------------------------------
# Abstraction Field
# ---------------------------------------------------------------------------

class AbstractionField(nn.Module):
    """Projects the latent vector z into a shared abstraction space."""

    def __init__(self, latent_dim: int, abstraction_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, abstraction_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(abstraction_dim * 2, abstraction_dim),
            nn.LayerNorm(abstraction_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, abstraction_dim)


# ---------------------------------------------------------------------------
# Concept Module
# ---------------------------------------------------------------------------

class ConceptModule(nn.Module):
    """A single concept module — a small MLP that processes an abstraction vector."""

    def __init__(self, abstraction_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(abstraction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, abstraction_dim),
            nn.LayerNorm(abstraction_dim),
        )
        # Per-module usage score (tracked by EvolutionController)
        self.register_buffer("usage_score", torch.tensor(1.0))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a) + a  # residual


# ---------------------------------------------------------------------------
# Concept Router
# ---------------------------------------------------------------------------

class ConceptRouter(nn.Module):
    """Computes routing scores and selects top-k concept modules.

    Returns a (B, abstraction_dim) aggregated representation formed by
    weighted summation of the top-k module outputs.
    """

    def __init__(self, abstraction_dim: int, num_modules: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        # Learnable routing key per module
        self.routing_keys = nn.Parameter(
            torch.randn(num_modules, abstraction_dim) * 0.02
        )

    def forward(
        self,
        a: torch.Tensor,                     # (B, abstraction_dim)
        concept_modules: nn.ModuleList,      # list of ConceptModule
        update_scores: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # aggregated, routing_weights
        # Routing scores: dot product of abstraction with each module key
        scores = torch.einsum("bd,nd->bn", a, self.routing_keys)  # (B, N)
        scores = scores / math.sqrt(a.size(-1))
        weights = F.softmax(scores, dim=-1)  # (B, N)

        # Select top-k modules
        k = min(self.top_k, len(concept_modules))
        topk_weights, topk_indices = torch.topk(weights, k, dim=-1)  # (B, k)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Run each module and aggregate
        B, D = a.shape
        aggregated = torch.zeros(B, D, device=a.device, dtype=a.dtype)

        # Update module usage scores (EMA proxy — simple sum of weights)
        module_weight_sum = weights.detach().mean(dim=0)  # (N,)

        for local_k in range(k):
            # Gather the module index for each sample in the batch
            mod_indices = topk_indices[:, local_k]  # (B,)
            w_k = topk_weights[:, local_k].unsqueeze(-1)  # (B, 1)

            # Vectorised: run each unique module
            unique_mods = mod_indices.unique().tolist()
            for mod_idx in unique_mods:
                mask = (mod_indices == mod_idx)          # (B,)
                if not mask.any():
                    continue
                out = concept_modules[mod_idx](a[mask])  # (n_mask, D)
                aggregated[mask] += w_k[mask] * out

        if update_scores:
            for i, mod in enumerate(concept_modules):
                if i < module_weight_sum.size(0):
                    mod.usage_score = mod.usage_score * 0.9 + module_weight_sum[i] * 0.1

        return aggregated, weights


# ---------------------------------------------------------------------------
# Latent World Model
# ---------------------------------------------------------------------------

class LatentWorldModel(nn.Module):
    """Predicts the next latent state given current z and aggregated concept output.

    Used as an auxiliary loss to encourage the latent space to be predictive.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        """Predict next z from current z and aggregated concept representation."""
        return self.net(torch.cat([z, agg], dim=-1))  # (B, latent_dim)


# ---------------------------------------------------------------------------
# Output Head
# ---------------------------------------------------------------------------

class OutputHead(nn.Module):
    """Maps the aggregated concept representation to class logits."""

    def __init__(self, abstraction_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(abstraction_dim, abstraction_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(abstraction_dim, num_classes),
        )

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        return self.net(agg)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Evolution Controller
# ---------------------------------------------------------------------------

class EvolutionController:
    """Manages the lifecycle of concept modules: pruning weak ones and spawning replacements."""

    def __init__(self, cfg: dict):
        evo_cfg = cfg["training"]["evolution"]
        self.enabled = evo_cfg["enabled"]
        self.prune_every = evo_cfg["prune_every_n_epochs"]
        self.prune_fraction = evo_cfg["prune_fraction"]
        self.abstraction_dim = cfg["model"]["abstraction_dim"]
        self.concept_hidden_dim = cfg["model"]["concept_hidden_dim"]
        self.dropout = cfg["model"]["dropout"]

    def maybe_evolve(
        self,
        epoch: int,
        concept_modules: nn.ModuleList,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger=None,
    ) -> nn.ModuleList:
        """Prune the weakest modules and replace them with fresh ones."""
        if not self.enabled:
            return concept_modules
        if epoch % self.prune_every != 0:
            return concept_modules

        n = len(concept_modules)
        n_prune = max(1, int(n * self.prune_fraction))

        scores = torch.stack([m.usage_score for m in concept_modules])
        _, prune_indices = torch.topk(scores, n_prune, largest=False)
        prune_set = set(prune_indices.tolist())

        if logger:
            logger.info(
                f"[Evolution] Epoch {epoch}: pruning {n_prune} modules "
                f"(indices {sorted(prune_set)}) and spawning replacements."
            )

        # Replace weak modules with fresh ones
        for idx in prune_set:
            new_mod = ConceptModule(
                self.abstraction_dim, self.concept_hidden_dim, self.dropout
            ).to(device)
            concept_modules[idx] = new_mod

        # Add new module parameters to optimizer
        new_params = [
            p for i, m in enumerate(concept_modules)
            if i in prune_set
            for p in m.parameters()
        ]
        if new_params:
            optimizer.add_param_group({"params": new_params})

        return concept_modules


# ---------------------------------------------------------------------------
# Full EAN Model
# ---------------------------------------------------------------------------

class EAN(nn.Module):
    """Evolutionary Abstraction Network — the full end-to-end model."""

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        latent_dim = model_cfg["latent_dim"]
        abstraction_dim = model_cfg["abstraction_dim"]
        num_concept_modules = model_cfg["num_concept_modules"]
        top_k = model_cfg["top_k"]
        concept_hidden_dim = model_cfg["concept_hidden_dim"]
        world_model_hidden = model_cfg["world_model_hidden_dim"]
        dropout = model_cfg["dropout"]
        num_classes = cfg["dataset"]["num_classes"]

        # --- Encoder ---
        encoder_name = model_cfg["encoder"].lower()
        if encoder_name == "cnn_baseline":
            self.encoder = SmallCNNEncoder(latent_dim)
        else:
            self.encoder = EfficientNetEncoder(
                latent_dim, pretrained=model_cfg.get("encoder_pretrained", True)
            )

        # --- EAN components ---
        self.abstraction_field = AbstractionField(latent_dim, abstraction_dim, dropout)
        self.concept_modules = nn.ModuleList([
            ConceptModule(abstraction_dim, concept_hidden_dim, dropout)
            for _ in range(num_concept_modules)
        ])
        self.router = ConceptRouter(abstraction_dim, num_concept_modules, top_k)
        self.world_model = LatentWorldModel(latent_dim, world_model_hidden, dropout)
        self.output_head = OutputHead(abstraction_dim, num_classes, dropout)

    def forward(
        self,
        x: torch.Tensor,
        update_scores: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Returns a dict with:
          - 'logits'       : class logits (B, num_classes)
          - 'z'            : latent vector (B, latent_dim)
          - 'z_pred'       : predicted next latent (B, latent_dim)
          - 'routing_weights' : per-module routing weights (B, N)
        """
        z = self.encoder(x)                                        # (B, latent_dim)
        a = self.abstraction_field(z)                             # (B, abstraction_dim)
        agg, routing_weights = self.router(
            a, self.concept_modules, update_scores=update_scores
        )                                                          # (B, abstraction_dim)
        z_pred = self.world_model(z, agg)                         # (B, latent_dim)
        logits = self.output_head(agg)                            # (B, num_classes)

        return {
            "logits": logits,
            "z": z,
            "z_pred": z_pred,
            "routing_weights": routing_weights,
        }

    def get_encoder_params(self):
        """Return encoder parameters (for a separate, lower learning rate)."""
        return list(self.encoder.parameters())

    def get_non_encoder_params(self):
        """Return all non-encoder parameters."""
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> EAN:
    """Construct and return an EAN model from a config dict."""
    return EAN(cfg)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    world_model_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined classification + world-model loss.

    Args:
        outputs: dict returned by EAN.forward()
        targets: ground-truth class indices (B,)
        world_model_weight: scalar weight for the auxiliary world-model loss

    Returns:
        total_loss: weighted sum of cross-entropy and world-model MSE
        loss_breakdown: dict of individual loss values (for logging)
    """
    # Classification loss
    cls_loss = F.cross_entropy(outputs["logits"], targets)

    # World model loss: predict next z from current z (self-supervised)
    # We use a stop-gradient on the target to prevent collapse
    z_target = outputs["z"].detach()
    wm_loss = F.mse_loss(outputs["z_pred"], z_target)

    total = cls_loss + world_model_weight * wm_loss

    return total, {
        "cls_loss": cls_loss.item(),
        "wm_loss": wm_loss.item(),
        "total_loss": total.item(),
    }
