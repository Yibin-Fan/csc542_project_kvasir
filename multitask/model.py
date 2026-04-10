import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """Two consecutive Conv2d + BatchNorm + ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskUNet(nn.Module):
    """
    Multi-task architecture with a shared ResNet-34 encoder.

    The encoder is split into five stages whose outputs serve as skip
    connections for the U-Net decoder.  A lightweight classification head
    is attached to the deepest encoder stage (layer4).

    Encoder (ResNet-34, ImageNet pretrained):
        enc0: conv1 + bn1 + relu  →  (B, 64,  H/2,  W/2)
        pool: maxpool             →  (B, 64,  H/4,  W/4)
        enc1: layer1              →  (B, 64,  H/4,  W/4)
        enc2: layer2              →  (B, 128, H/8,  W/8)
        enc3: layer3              →  (B, 256, H/16, W/16)
        enc4: layer4              →  (B, 512, H/32, W/32)

    Classification head (enc4 → cls_logits):
        AdaptiveAvgPool2d(1) → Flatten → Linear(512 → num_classes)

    Segmentation decoder (enc4 → seg_logits, same H×W as input):
        up4 + cat(enc3) → dec4  →  (B, 256, H/16, W/16)
        up3 + cat(enc2) → dec3  →  (B, 128, H/8,  W/8)
        up2 + cat(enc1) → dec2  →  (B, 64,  H/4,  W/4)
        up1 + cat(enc0) → dec1  →  (B, 64,  H/2,  W/2)
        up0             → dec0  →  (B, 32,  H,    W)
        final_conv              →  (B, 1,   H,    W)   raw logits

    Args:
        num_classes: Number of output classes for the classification head.

    Inputs:  (B, 3, H, W)  — any resolution; H, W should be multiples of 32
    Outputs: cls_logits (B, num_classes),  seg_logits (B, 1, H, W)
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # ── Shared encoder ──────────────────────────────────────────────────
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1   # 64 ch
        self.enc2 = resnet.layer2   # 128 ch
        self.enc3 = resnet.layer3   # 256 ch
        self.enc4 = resnet.layer4   # 512 ch

        # ── Classification head ──────────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        # ── Segmentation decoder ─────────────────────────────────────────────
        # H/32 → H/16  (concat enc3: 256)
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)

        # H/16 → H/8   (concat enc2: 128)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)

        # H/8  → H/4   (concat enc1: 64)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)

        # H/4  → H/2   (concat enc0: 64)
        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        # H/2  → H     (no skip — restore full resolution)
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cat(up_out, skip):
        """Align spatial size then concatenate along channel dim."""
        if up_out.shape[2:] != skip.shape[2:]:
            up_out = F.interpolate(up_out, size=skip.shape[2:],
                                   mode="bilinear", align_corners=False)
        return torch.cat([up_out, skip], dim=1)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)       # (B, 64,  H/2,  W/2)
        p  = self.pool(e0)      # (B, 64,  H/4,  W/4)
        e1 = self.enc1(p)       # (B, 64,  H/4,  W/4)
        e2 = self.enc2(e1)      # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)      # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)      # (B, 512, H/32, W/32)

        # Classification
        cls_logits = self.cls_head(e4)          # (B, num_classes)

        # Segmentation decoder
        d = self.dec4(self._cat(self.up4(e4), e3))
        d = self.dec3(self._cat(self.up3(d),  e2))
        d = self.dec2(self._cat(self.up2(d),  e1))
        d = self.dec1(self._cat(self.up1(d),  e0))
        d = self.dec0(self.up0(d))
        seg_logits = self.final_conv(d)         # (B, 1, H, W)

        return cls_logits, seg_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
