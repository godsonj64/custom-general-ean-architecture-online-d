"""Dataset loading and preprocessing for the EAN project.

CIFAR-100 (or CIFAR-10) is downloaded automatically from the internet
the first time this module is used.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_train_transform(cfg: dict) -> transforms.Compose:
    """Build the augmentation pipeline used during training."""
    image_size = cfg["dataset"]["image_size"]
    mean = cfg["dataset"]["normalize_mean"]
    std = cfg["dataset"]["normalize_std"]
    augment = cfg["dataset"].get("augment", True)

    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """Build the deterministic transform pipeline used during validation and testing."""
    image_size = cfg["dataset"]["image_size"]
    mean = cfg["dataset"]["normalize_mean"]
    std = cfg["dataset"]["normalize_std"]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _get_torchvision_dataset(name: str, root: str, train: bool, transform):
    """Return a torchvision dataset, downloading it if necessary."""
    name = name.lower()
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    elif name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset name: '{name}'. Choose 'cifar100' or 'cifar10'.")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader, test_loader) DataLoaders.

    The training split is further divided into train / val using the ratio
    specified in `cfg.dataset.train_split`.
    """
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["data_dir"]
    batch_size = cfg["dataset"]["batch_size"]
    num_workers = cfg["dataset"]["num_workers"]
    train_fraction = cfg["dataset"].get("train_split", 0.9)

    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)

    # --- Full training set (with augmentation) ---
    full_train_ds = _get_torchvision_dataset(name, root, train=True, transform=train_tf)

    # --- Full training set (without augmentation, used for validation split) ---
    full_train_ds_noaug = _get_torchvision_dataset(name, root, train=True, transform=eval_tf)

    n_total = len(full_train_ds)
    n_train = int(n_total * train_fraction)
    n_val = n_total - n_train

    # Deterministic split using a fixed generator
    gen = torch.Generator().manual_seed(42)
    train_indices, val_indices = [
        idx.tolist()
        for idx in torch.randperm(n_total, generator=gen).split([n_train, n_val])
    ]

    train_ds = Subset(full_train_ds, train_indices)
    val_ds = Subset(full_train_ds_noaug, val_indices)

    # --- Test set ---
    test_ds = _get_torchvision_dataset(name, root, train=False, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_num_classes(cfg: dict) -> int:
    """Return the number of output classes for the configured dataset."""
    return cfg["dataset"]["num_classes"]
