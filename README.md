# CSC542 Group Project

## Dataset

This project uses the [Kvasir Dataset for Classification and Segmentation](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation) from Kaggle.

### Download via Kaggle CLI

Go to [Kaggle Account Settings](https://www.kaggle.com/settings) → API → Create New Token, then copy the generated `username` and `key` values and export them as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

Then download the dataset:

```bash
kaggle datasets download abdallahwagih/kvasir-dataset-for-classification-and-segmentation
```

### Extract and Place Files

After downloading, unzip and place the data under `data/` as follows:

```bash
unzip kvasir-dataset-for-classification-and-segmentation.zip -d data/
```

The expected directory structure is:

```
Kvasir/
└── data/
    ├── kvasir-dataset/
    │   └── kvasir-dataset/          # 8 class sub-folders (500 images each)
    │       ├── dyed-lifted-polyps/
    │       ├── dyed-resection-margins/
    │       ├── esophagitis/
    │       ├── normal-cecum/
    │       ├── normal-pylorus/
    │       ├── normal-z-line/
    │       ├── polyps/
    │       └── ulcerative-colitis/
    └── kvasir-seg/
        └── Kvasir-SEG/              # Segmentation dataset
            ├── images/              # 1000 RGB images
            └── masks/               # 1000 binary masks
```

---

## Setup

```bash
conda create -n kvasir python=3.10 -y
conda activate kvasir
pip install -r requirements.txt
```

---

## Run Command

```bash
# Classification baseline (ResNet-18)
python -m classification.train \
    --data_dir data/kvasir-dataset/kvasir-dataset \
    --output_dir outputs/classification

# Segmentation baseline (plain U-Net)
python -m segmentation.train \
    --data_dir data/kvasir-seg/Kvasir-SEG \
    --output_dir outputs/segmentation

# Multi-task model (ResNet-34 shared encoder)
python -m multitask.train \
    --cls_data_dir data/kvasir-dataset/kvasir-dataset \
    --seg_data_dir data/kvasir-seg/Kvasir-SEG \
    --output_dir outputs/multitask
```

---

## Multi-task Model

### Architecture

The proposed model (`multitask/model.py`) replaces the two independent baselines with a single network that solves both tasks simultaneously via a shared encoder.

```
Input (B, 3, H, W)
        │
        ▼
┌─────────────────────────────────────────────────┐
│           ResNet-34 Encoder (shared)            │
│                                                 │
│  enc0: conv1+bn+relu  →  (B,  64, H/2,  W/2)  ─── skip0
│  pool: maxpool        →  (B,  64, H/4,  W/4)   │
│  enc1: layer1         →  (B,  64, H/4,  W/4)  ─── skip1
│  enc2: layer2         →  (B, 128, H/8,  W/8)  ─── skip2
│  enc3: layer3         →  (B, 256, H/16, W/16) ─── skip3
│  enc4: layer4         →  (B, 512, H/32, W/32)  │
└─────────────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌───────────────┐  ┌──────────────────────────────────────┐
│ Classification│  │         U-Net Decoder                │
│     Head      │  │                                      │
│               │  │  up4 + cat(skip3) → DoubleConv(256)  │
│  AvgPool      │  │  up3 + cat(skip2) → DoubleConv(128)  │
│  Flatten      │  │  up2 + cat(skip1) → DoubleConv(64)   │
│  Linear(→8)   │  │  up1 + cat(skip0) → DoubleConv(64)   │
│               │  │  up0              → DoubleConv(32)   │
│  (B, 8)       │  │  Conv1×1          → (B, 1, H, W)     │
└───────────────┘  └──────────────────────────────────────┘
        │                │
        ▼                ▼
   cls_logits        seg_logits
```

**Key design choices:**

- **Shared encoder** — ResNet-34 pretrained on ImageNet replaces the from-scratch U-Net encoder. All five encoder stages produce skip connections used by the decoder.
- **Classification head** — Attached to the deepest encoder stage (`enc4`). `AdaptiveAvgPool2d(1)` compresses spatial dimensions before the linear layer, making the head resolution-agnostic.
- **Segmentation decoder** — Standard U-Net decoder with five upsampling steps, restoring the output to the full input resolution.
- **Resolution flexibility** — Because the classification head uses global average pooling and the decoder upsamples to match the input size, the model accepts any input resolution without modification.

### Training Strategy

Training uses two separate DataLoaders (one per task) that are iterated in parallel within each epoch:

```
for each cls batch (imgs, labels):
    forward cls batch  →  cls_logits
    loss_cls = CrossEntropy(cls_logits, labels)

    fetch next seg batch (imgs, masks):   # seg loader cycles if exhausted
    forward seg batch  →  seg_logits
    loss_seg = 0.5 * BCE + 0.5 * SoftDice

    total_loss = loss_seg + λ * loss_cls
    total_loss.backward()
    optimizer.step()
```

The segmentation loss combines BCE and differentiable soft-Dice equally (`0.5 * BCE + 0.5 * Dice`). The trade-off between tasks is controlled by `--lambda_cls` (default `0.5`).

Checkpointing uses a combined validation score `val_dice + val_cls_acc` so neither task is sacrificed for the other.

### Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--lr` | 1e-4 | Adam learning rate (CosineAnnealingLR scheduler) |
| `--cls_batch` | 32 | Classification batch size |
| `--seg_batch` | 8 | Segmentation batch size |
| `--lambda_cls` | 0.5 | Classification loss weight |
| `--seed` | 42 | Random seed |
| `--output_dir` | `./outputs/multitask` | Checkpoint and results output path |
