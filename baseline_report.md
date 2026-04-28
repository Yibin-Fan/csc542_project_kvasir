# Kvasir Baseline Training Report

## Classification Task — ResNet-18

**Training config:** 30 epochs, batch size 32, lr 1e-4 (Adam + CosineAnnealingLR), ImageNet pretrained weights

| Metric | Value |
|--------|-------|
| Parameters | 11,180,616 |
| Best Val Accuracy | 93.17% (Epoch 24) |
| Test Accuracy | 93.00% |

### Per-class Results (Test Set, 75 samples per class)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| dyed-lifted-polyps | 0.9041 | 0.8800 | 0.8919 |
| dyed-resection-margins | 0.9211 | 0.9333 | 0.9272 |
| esophagitis | 0.9118 | 0.8267 | 0.8671 |
| normal-cecum | 0.9867 | 0.9867 | 0.9867 |
| normal-pylorus | 0.9868 | 1.0000 | 0.9934 |
| normal-z-line | 0.8415 | 0.9200 | 0.8790 |
| polyps | 0.9324 | 0.9200 | 0.9262 |
| ulcerative-colitis | 0.9605 | 0.9733 | 0.9669 |
| **macro avg** | **0.9306** | **0.9300** | **0.9298** |

---

## Segmentation Task — U-Net

**Training config:** 30 epochs, batch size 32, lr 1e-4 (Adam + CosineAnnealingLR)

| Metric | Value |
|--------|-------|
| Test Dice | 0.8318 |
| Test IoU | 0.7421 |

---

## Multi-task Task — Shared ResNet-34 Encoder

**Training config:** 30 epochs, classification batch size 32, segmentation batch size 8, lr 1e-4 (Adam + CosineAnnealingLR), ImageNet pretrained ResNet-34 encoder, classification loss weight `lambda_cls=0.5`

| Metric | Value |
|--------|-------|
| Parameters | 24,455,689 |
| Best Combined Val Score | 1.8030 (Epoch 20: val acc 0.9133, val Dice 0.8897) |
| Test Classification Accuracy | 89.50% |
| Test Segmentation Dice | 0.8744 |
| Test Segmentation IoU | 0.8072 |

### Comparison

| Metric | Baseline | Multi-task | Change |
|--------|----------|------------|--------|
| Classification Accuracy | 93.00% recorded baseline / 92.17% rerun | 89.50% | Lower |
| Segmentation Dice | 0.8318 | 0.8744 | +0.0426 |
| Segmentation IoU | 0.7421 | 0.8072 | +0.0651 |

The multi-task model improved segmentation performance but reduced classification accuracy. The stronger pretrained shared encoder likely helped the mask decoder, while the shared optimization objective and `lambda_cls=0.5` may have underweighted classification relative to the dense segmentation loss.
