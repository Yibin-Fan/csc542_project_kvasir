"""
Classification training script for the Kvasir dataset.

Usage (run from the Kvasir/ directory):
    python -m classification.train --data_dir /path/to/kvasir-dataset-v2

The script fine-tunes a pretrained ResNet-18, checkpoints the best model by
validation accuracy, and prints a full sklearn classification report on the
held-out test set.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

# Allow running as `python -m classification.train` from Kvasir/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classification.dataset import get_dataloaders
from classification.model import ResNetClassifier, count_parameters
from utils import get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device):
    """Return accuracy and collect all predictions + labels (on CPU)."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = (all_preds == all_labels).mean()
    return acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on Kvasir classification dataset")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Path to kvasir-dataset-v2/ (contains class sub-folders)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", type=str,   default="./outputs/classification")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")

    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed
    )

    # Model
    model = ResNetClassifier(num_classes=len(class_names)).to(device)
    print(f"ResNet-18 parameters: {count_parameters(model):,}")

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc  = correct / total

        # --- Validate ---
        val_acc, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (val acc {best_val_acc:.4f})")

    # --- Test with best model ---
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)

    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print(f"\nTest Accuracy: {test_acc:.4f}\n")
    print(report)

    report_path = os.path.join(args.output_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
