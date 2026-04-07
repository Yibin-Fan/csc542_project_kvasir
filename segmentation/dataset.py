import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


IMG_SIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_paths(directory: str):
    """Return sorted list of image file paths in *directory*."""
    paths = sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )
    return paths


class KvasirSegDataset(Dataset):
    """
    Dataset for Kvasir-SEG polyp segmentation.

    Expected directory layout::

        data_dir/
        ├── images/   (JPEG or PNG)
        └── masks/    (same filenames as images)

    Images and masks are matched by sorted filename order (filenames are
    identical in the Kvasir-SEG dataset).

    Augmentation (train only):
        - Random horizontal flip
        - Random vertical flip
        The same random state is applied to both image and mask to keep them
        spatially consistent (safe because ``num_workers=0``).
    """

    def __init__(self, image_paths: list, mask_paths: list, train: bool = True):
        assert len(image_paths) == len(mask_paths), \
            f"Image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize
        img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        if self.train:
            # Use a shared seed so the same spatial ops are applied to img and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            if torch.rand(1).item() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            torch.manual_seed(seed + 1)
            if torch.rand(1).item() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)

        img  = TF.to_tensor(img)
        img  = TF.normalize(img, NORMALIZE_MEAN, NORMALIZE_STD)
        mask = TF.to_tensor(mask)          # [0, 1]
        mask = (mask > 0.5).float()        # binarize (handles JPEG anti-aliasing)

        return img, mask                   # shapes: (3, H, W), (1, H, W)


def get_dataloaders(data_dir: str, batch_size: int = 8, seed: int = 42):
    """
    Build train/val/test DataLoaders for the Kvasir-SEG dataset.

    Args:
        data_dir:   Root directory containing ``images/`` and ``masks/`` sub-folders.
        batch_size: Batch size for all loaders.
        seed:       Random seed for reproducible splits.

    Returns:
        train_loader, val_loader, test_loader
    """
    img_paths  = _collect_paths(os.path.join(data_dir, "images"))
    mask_paths = _collect_paths(os.path.join(data_dir, "masks"))

    assert len(img_paths) > 0, f"No images found in {os.path.join(data_dir, 'images')}"
    assert len(img_paths) == len(mask_paths), \
        f"Image/mask count mismatch: {len(img_paths)} vs {len(mask_paths)}"

    n = len(img_paths)
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    n_test  = max(1, round(n * 0.15))
    n_val   = max(1, round(n * 0.15))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    def _make_dataset(indices, train):
        return KvasirSegDataset(
            [img_paths[i]  for i in indices],
            [mask_paths[i] for i in indices],
            train=train,
        )

    train_ds = _make_dataset(train_idx, train=True)
    val_ds   = _make_dataset(val_idx,   train=False)
    test_ds  = _make_dataset(test_idx,  train=False)

    loader_kwargs = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"Segmentation split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
