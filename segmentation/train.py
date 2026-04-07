"""
Baseline segmentation training script for the Kvasir-SEG dataset.

Usage (run from the Kvasir/ directory):
    python -m segmentation.train --data_dir /path/to/kvasir-seg

The script trains a plain U-Net from scratch, checkpoints the best model by
validation Dice coefficient, and reports Dice + IoU on the held-out test set.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segmentation.dataset import get_dataloaders
from segmentation.model import UNet
from utils import get_device, dice_coefficient, iou_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device):
    """Return mean Dice and mean IoU over the loader."""
    model.eval()
    dice_scores, iou_scores = [], []
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        # Move to CPU for metric computation
        preds_cpu = preds.cpu()
        masks_cpu = masks.cpu()
        dice_scores.append(dice_coefficient(preds_cpu, masks_cpu).item())
        iou_scores.append(iou_score(preds_cpu, masks_cpu).item())
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Kvasir-SEG dataset")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Path to kvasir-seg/ (contains images/ and masks/ sub-folders)")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", type=str,   default="./outputs/segmentation")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")

    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed
    )

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {n_params:,}")

    # Training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        total = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)          # shape (B, 1, H, W)

            optimizer.zero_grad()
            preds = model(imgs)               # logits, shape (B, 1, H, W)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        train_loss = running_loss / total

        # --- Validate ---
        val_dice, val_iou = evaluate(model, val_loader, device)
        scheduler.step(val_dice)

        print(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} "
            f"| val Dice {val_dice:.4f} | val IoU {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (val Dice {best_val_dice:.4f})")

    # --- Test with best model ---
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_dice, test_iou = evaluate(model, test_loader, device)
    print(f"\nTest Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")

    result_path = os.path.join(args.output_dir, "test_results.txt")
    with open(result_path, "w") as f:
        f.write(f"Test Dice: {test_dice:.4f}\n")
        f.write(f"Test IoU:  {test_iou:.4f}\n")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
