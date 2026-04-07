import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]


def _get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


class TransformSubset(Dataset):
    """Wraps a Subset (whose base dataset has transform=None) and applies a transform."""

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # img is PIL Image (base dataset has no transform)
        return self.transform(img), label


def get_dataloaders(data_dir: str, batch_size: int = 32, seed: int = 42):
    """
    Build train/val/test DataLoaders for the Kvasir classification dataset.

    Args:
        data_dir:   Root directory containing one sub-folder per class
                    (e.g. kvasir-dataset-v2/).
        batch_size: Batch size for all loaders.
        seed:       Random seed for reproducible splits.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Load without transform so PIL images pass through to TransformSubset
    base_dataset = datasets.ImageFolder(data_dir, transform=None)
    class_names = base_dataset.classes
    targets = base_dataset.targets        # list[int], length N

    n = len(base_dataset)
    # Stratified 70 / 15 / 15 split
    # Build per-class index lists then sample proportionally
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, t in enumerate(targets):
        class_indices[t].append(idx)

    rng = torch.Generator()
    rng.manual_seed(seed)

    train_idx, val_idx, test_idx = [], [], []
    for cls_idxs in class_indices.values():
        perm = torch.randperm(len(cls_idxs), generator=rng).tolist()
        n_cls = len(perm)
        n_test = max(1, round(n_cls * 0.15))
        n_val  = max(1, round(n_cls * 0.15))
        n_train = n_cls - n_val - n_test
        train_idx.extend([cls_idxs[i] for i in perm[:n_train]])
        val_idx.extend([cls_idxs[i]   for i in perm[n_train:n_train + n_val]])
        test_idx.extend([cls_idxs[i]  for i in perm[n_train + n_val:]])

    train_ds = TransformSubset(Subset(base_dataset, train_idx), _get_transforms(train=True))
    val_ds   = TransformSubset(Subset(base_dataset, val_idx),   _get_transforms(train=False))
    test_ds  = TransformSubset(Subset(base_dataset, test_idx),  _get_transforms(train=False))

    loader_kwargs = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"Dataset split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    print(f"Classes ({len(class_names)}): {class_names}")

    return train_loader, val_loader, test_loader, class_names
