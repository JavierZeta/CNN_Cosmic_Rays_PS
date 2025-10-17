# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, x1_ch, x2_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(x1_ch, x1_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(x1_ch + x2_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetSimple(nn.Module):
    """
    U-Net that accepts channel dimension = number of frames (T).
    in_channels: number of channels (e.g. 4 frames -> in_channels=4)
    n_classes: number of output channels (e.g. 4 to predict mask per frame)
    base_ch: channel multiplicative factor (reduce for faster models)
    """
    def __init__(self, in_channels=4, n_classes=4, base_ch=16):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.bot   = DoubleConv(base_ch*8, base_ch*16)
        self.up3 = Up(base_ch*16, base_ch*8, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4, base_ch*4)
        self.up1 = Up(base_ch*4, base_ch*2, base_ch*2)
        self.up0 = Up(base_ch*2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        # x shape expected: [B, C_in, H, W]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        u3 = self.up3(x5, x4)
        u2 = self.up2(u3, x3)
        u1 = self.up1(u2, x2)
        u0 = self.up0(u1, x1)
        out = self.outc(u0)
        return out   # logits, shape [B, n_classes, H, W]
