import torch


def get_device():
    """Return best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_coefficient(preds, targets, threshold=0.5, eps=1e-6):
    """
    Compute mean Dice coefficient over a batch.

    Args:
        preds:   raw logits or probabilities, shape (B, 1, H, W) or (B, H, W)
        targets: binary ground-truth masks, same shape as preds
        threshold: binarization threshold applied after sigmoid
    Returns:
        Scalar tensor (mean Dice over batch).
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    # Flatten spatial dims per sample
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()
    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)
    return dice.mean()


def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    Compute mean IoU over a batch.

    Args:
        preds:   raw logits or probabilities, shape (B, 1, H, W) or (B, H, W)
        targets: binary ground-truth masks, same shape as preds
        threshold: binarization threshold applied after sigmoid
    Returns:
        Scalar tensor (mean IoU over batch).
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()
