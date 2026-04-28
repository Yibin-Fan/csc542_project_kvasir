# Kvasir Experiment Results Summary

## Dataset Summary

- Classification dataset: `data/kvasir-dataset/kvasir-dataset`
  - 8 classes
  - 500 images per class
  - 4000 images total
  - Split used by script: 2800 train / 600 validation / 600 test
- Segmentation dataset: `data/kvasir-seg/Kvasir-SEG`
  - 1000 RGB images
  - 1000 binary masks
  - Split used by script: 700 train / 150 validation / 150 test

## Model Summary

| Model | Task | Architecture |
|---|---|---|
| Classification baseline | 8-class classification | ImageNet-pretrained ResNet-18 with replaced FC head |
| Segmentation baseline | Binary polyp segmentation | Vanilla U-Net trained from scratch |
| Multi-task model | Classification + segmentation | Shared ImageNet-pretrained ResNet-34 encoder, classification head, U-Net-style decoder |

## Training Setup

| Setting | Classification baseline | Segmentation baseline | Multi-task model |
|---|---:|---:|---:|
| Epochs | 30 | 30 recorded baseline result | 30 |
| Optimizer | Adam | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | Existing report records CosineAnnealingLR; current script default uses ReduceLROnPlateau | CosineAnnealingLR |
| Batch size | 32 | Existing report records 32; current script default is 8 | cls 32 / seg 8 |
| Image size | 224x224 | 256x256 | cls 224x224 / seg 256x256 |
| Seed | 42 | 42 | 42 |
| Checkpoint criterion | Best validation accuracy | Best validation Dice | Best validation Dice + validation classification accuracy |

Segmentation baseline was not rerun because `outputs/segmentation/test_results.txt` already contained valid results matching `baseline_report.md`. The current script accepts `--epochs`; if it is rerun for strict reproduction, use `--epochs 30` explicitly.

## Environment

- Conda environment: `kvasir`
- Python: 3.10.20 (`/home/zwang269/miniconda3/envs/kvasir/bin/python`)
- PyTorch: 2.11.0+cu130
- torchvision: 0.26.0+cu130
- NumPy: 2.2.6
- scikit-learn: 1.7.2
- CUDA available: True
- CUDA device count: 2
- GPU used by PyTorch default: NVIDIA RTX A5000

Dependencies were installed with `conda run -n kvasir python -m pip install -r requirements.txt`.

## Final Metrics

| Model | Classification Accuracy | Segmentation Dice | Segmentation IoU | Checkpoint |
|---|---:|---:|---:|---|
| Classification baseline, ResNet-18 | 0.9217 | N/A | N/A | `outputs/classification/best_model.pth` |
| Segmentation baseline, U-Net | N/A | 0.8318 | 0.7421 | `outputs/segmentation/best_model.pth` |
| Multi-task ResNet-34 U-Net | 0.8950 | 0.8744 | 0.8072 | `outputs/multitask/best_model.pth` |

## Validation Notes

- Classification baseline best validation accuracy: 0.9317 at epoch 29.
- Multi-task best combined validation score occurred at epoch 20:
  - validation classification accuracy: 0.9133
  - validation Dice: 0.8897
  - validation IoU: 0.8244
  - combined score: 1.8030

## Comparison

| Comparison | Baseline | Multi-task | Difference |
|---|---:|---:|---:|
| Classification accuracy | 0.9217 | 0.8950 | -0.0267 |
| Segmentation Dice | 0.8318 | 0.8744 | +0.0426 |
| Segmentation IoU | 0.7421 | 0.8072 | +0.0651 |

The multi-task model hurt classification accuracy compared with the ResNet-18 classification baseline, dropping from 92.17% to 89.50%. It improved segmentation, increasing Dice from 0.8318 to 0.8744 and IoU from 0.7421 to 0.8072.

## Analysis

The segmentation improvement is likely due to the stronger ImageNet-pretrained ResNet-34 encoder used by the multi-task model. The segmentation baseline trains a vanilla U-Net encoder from scratch on only 700 training masks, while the multi-task decoder receives pretrained hierarchical features and additional representation pressure from the classification task.

The classification drop suggests task conflict or optimization imbalance. The shared encoder is updated by both classification and segmentation losses, and the segmentation path contributes a dense pixel-level objective every step. With `lambda_cls=0.5`, the classification objective may be underweighted relative to segmentation. The multi-task classification head also shares capacity with the decoder rather than optimizing solely for classification, unlike the dedicated ResNet-18 baseline.

Overall, this setup favored segmentation quality over classification accuracy. Further experiments could tune `lambda_cls`, adjust batch scheduling, freeze early encoder layers, or checkpoint with a weighted validation score if the project objective requires a different trade-off.

## Commands Run

```bash
conda run -n kvasir python -m pip install -r requirements.txt

conda run -n kvasir python -c "import sys, torch, torchvision, numpy, sklearn; ..."

conda run -n kvasir bash -lc 'mkdir -p outputs/classification && python -m classification.train --data_dir data/kvasir-dataset/kvasir-dataset --output_dir outputs/classification --epochs 30 2>&1 | tee outputs/classification/train.log'

conda run -n kvasir bash -lc 'mkdir -p outputs/multitask && python -m multitask.train --cls_data_dir data/kvasir-dataset/kvasir-dataset --seg_data_dir data/kvasir-seg/Kvasir-SEG --output_dir outputs/multitask --epochs 30 2>&1 | tee outputs/multitask/train.log'
```

## Output Files

- `outputs/classification/train.log`
- `outputs/classification/test_report.txt`
- `outputs/classification/best_model.pth`
- `outputs/segmentation/test_results.txt`
- `outputs/segmentation/best_model.pth`
- `outputs/multitask/train.log`
- `outputs/multitask/test_results.txt`
- `outputs/multitask/best_model.pth`
