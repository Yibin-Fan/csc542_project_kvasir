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

*To be filled in after training completes.*
