import torch
import torch.nn as nn
import torch.nn.functional as F


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


class UNet(nn.Module):
    """
    Standard U-Net for binary polyp segmentation.

    Args:
        in_channels:  Number of input channels (3 for RGB).
        out_channels: Number of output channels (1 for binary mask).
        features:     Channel sizes for the four encoder levels.

    Input:  (B, 3, H, W)  — expects H and W to be multiples of 16
    Output: (B, 1, H, W)  — raw logits (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
    ):
        super().__init__()
        self.downs     = nn.ModuleList()
        self.pools     = nn.ModuleList()
        self.ups       = nn.ModuleList()
        self.up_convs  = nn.ModuleList()

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(f * 2, f))

        # Output head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for up, up_conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            # Handle size mismatch caused by odd spatial dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final_conv(x)   # raw logits
