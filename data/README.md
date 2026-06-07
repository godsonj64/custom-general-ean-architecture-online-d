# Data Directory

This directory will be populated **automatically** when you run training.

The dataset (CIFAR-100 by default) is downloaded via `torchvision.datasets` the first time training is launched. No manual download or upload is required.

## Dataset: CIFAR-100
- **100 classes**, 60,000 32×32 colour images (50,000 train / 10,000 test)
- Images are resized to 224×224 for the EfficientNet-B0 encoder
- Download location: `data/cifar100/`

## Changing the Dataset
Update `configs/default.yaml`:
```yaml
dataset:
  name: "cifar100"   # or "cifar10"
  num_classes: 100   # match your dataset
```

## Directory Layout (after first run)
```
data/
└── cifar100/
    ├── cifar-100-python/
    │   ├── train
    │   └── test
    └── ...
```
