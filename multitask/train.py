"""
Multi-task training script: shared ResNet-34 encoder with U-Net segmentation
decoder and classification head.

Usage (run from the Kvasir/ directory):
    python -m multitask.train \\
        --cls_data_dir /path/to/kvasir-dataset-v2 \\
        --seg_data_dir /path/to/kvasir-seg

Training strategy
-----------------
Each epoch iterates over all classification batches.  A segmentation batch is
drawn in parallel (the segmentation loader is cycled if it runs out first).
Losses are computed separately and summed:

    total_loss = seg_loss + lambda_cls * cls_loss

where seg_loss = 0.5 * BCE + 0.5 * Dice  (differentiable soft-Dice).

Validation checkpoints the best combined score (val_dice + val_cls_acc).
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classification.dataset import get_dataloaders as cls_get_dataloaders
from segmentation.dataset import get_dataloaders as seg_get_dataloaders
from multitask.model import MultiTaskUNet, count_parameters
from utils import get_device, dice_coefficient, iou_score


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def soft_dice_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Dice loss (uses sigmoid probabilities, not hard threshold)."""
    probs = torch.sigmoid(preds)
    p = probs.view(probs.size(0), -1)
    t = targets.view(targets.size(0), -1)
    intersection = (p * t).sum(dim=1)
    dice = (2.0 * intersection + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
    return 1.0 - dice.mean()


def seg_loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE + soft-Dice, weighted equally."""
    bce  = F.binary_cross_entropy_with_logits(preds, targets)
    dice = soft_dice_loss(preds, targets)
    return 0.5 * bce + 0.5 * dice


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_classification(model: nn.Module, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        cls_logits, _ = model(imgs)
        preds = cls_logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = float((all_preds == all_labels).mean())
    return acc, all_preds, all_labels


@torch.no_grad()
def eval_segmentation(model: nn.Module, loader, device):
    model.eval()
    dice_scores, iou_scores = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        _, seg_logits = model(imgs)
        seg_logits_cpu = seg_logits.cpu()
        masks_cpu      = masks.cpu()
        dice_scores.append(dice_coefficient(seg_logits_cpu, masks_cpu).item())
        iou_scores.append(iou_score(seg_logits_cpu, masks_cpu).item())
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-task ResNet-34 U-Net on Kvasir datasets"
    )
    parser.add_argument("--cls_data_dir", type=str, required=True,
                        help="Path to kvasir-dataset-v2/ (class sub-folders)")
    parser.add_argument("--seg_data_dir", type=str, required=True,
                        help="Path to kvasir-seg/ (images/ and masks/ sub-folders)")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--cls_batch",   type=int,   default=32,
                        help="Batch size for classification loader")
    parser.add_argument("--seg_batch",   type=int,   default=8,
                        help="Batch size for segmentation loader")
    parser.add_argument("--lambda_cls",  type=float, default=0.5,
                        help="Weight for classification loss (total = seg + λ*cls)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output_dir",  type=str,   default="./outputs/multitask")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")

    device = get_device()
    print(f"Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    cls_train_loader, cls_val_loader, cls_test_loader, class_names = cls_get_dataloaders(
        args.cls_data_dir, batch_size=args.cls_batch, seed=args.seed
    )
    seg_train_loader, seg_val_loader, seg_test_loader = seg_get_dataloaders(
        args.seg_data_dir, batch_size=args.seg_batch, seed=args.seed
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = MultiTaskUNet(num_classes=len(class_names)).to(device)
    print(f"MultiTaskUNet parameters: {count_parameters(model):,}")

    ce_loss  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_combined = 0.0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()

        seg_iter = iter(seg_train_loader)   # cycled alongside cls batches

        total_loss_sum = 0.0
        total_samples  = 0

        for cls_imgs, cls_labels in tqdm(
            cls_train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
            leave=False,
        ):
            optimizer.zero_grad()

            # ── Classification step ──────────────────────────────────────
            cls_imgs   = cls_imgs.to(device)
            cls_labels = cls_labels.to(device)
            cls_logits, _ = model(cls_imgs)
            loss_cls = ce_loss(cls_logits, cls_labels)

            # ── Segmentation step ────────────────────────────────────────
            try:
                seg_imgs, seg_masks = next(seg_iter)
            except StopIteration:
                seg_iter = iter(seg_train_loader)
                seg_imgs, seg_masks = next(seg_iter)

            seg_imgs  = seg_imgs.to(device)
            seg_masks = seg_masks.to(device)
            _, seg_logits = model(seg_imgs)
            loss_seg = seg_loss_fn(seg_logits, seg_masks)

            # ── Combined loss ────────────────────────────────────────────
            loss = loss_seg + args.lambda_cls * loss_cls
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item() * cls_imgs.size(0)
            total_samples  += cls_imgs.size(0)

        scheduler.step()
        avg_loss = total_loss_sum / total_samples

        # ── Validation ───────────────────────────────────────────────────
        val_acc,  _, _ = eval_classification(model, cls_val_loader, device)
        val_dice, val_iou = eval_segmentation(model, seg_val_loader, device)

        combined = val_dice + val_acc
        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} "
            f"| val cls acc {val_acc:.4f} "
            f"| val Dice {val_dice:.4f} | val IoU {val_iou:.4f}"
        )

        if combined > best_combined:
            best_combined = combined
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (acc {val_acc:.4f}, Dice {val_dice:.4f})")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_acc, test_preds, test_labels = eval_classification(model, cls_test_loader, device)
    test_dice, test_iou = eval_segmentation(model, seg_test_loader, device)

    cls_report = classification_report(
        test_labels, test_preds, target_names=class_names, digits=4
    )
    print(f"\nTest Classification Accuracy: {test_acc:.4f}")
    print(cls_report)
    print(f"Test Segmentation  Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, "test_results.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Classification Accuracy: {test_acc:.4f}\n\n")
        f.write(cls_report)
        f.write(f"\nTest Segmentation Dice: {test_dice:.4f}\n")
        f.write(f"Test Segmentation IoU:  {test_iou:.4f}\n")
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
