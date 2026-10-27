import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.twinlitenet import ConvBNReLU, DepthwiseSeparableConv


class HFFDilatedContextBlock(nn.Module):
    """
    DilatedContextBlock with ESPNet-style hierarchical feature fusion: each dilated branch is summed with
    the smaller-dilation branches before concatenation, which fills the "blind spots" (gridding) that
    parallel dilated convs leave on their own.
    """
    def __init__(self, channels):
        super(HFFDilatedContextBlock, self).__init__()
        self.reduce = ConvBNReLU(channels, channels // 2, kernel_size=1, padding=0)
        self.d1 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=8, dilation=8, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.reduce(x)
        o1 = self.d1(x)
        o2 = self.d2(x)
        o4 = o2 + self.d4(x)   # hierarchical fusion: d4 also sees d2's denser sampling
        o8 = o4 + self.d8(x)
        out = self.bn(torch.cat([o1, o2, o4, o8], dim=1))
        return self.relu(out + residual)


class CoordinateAttention(nn.Module):
    """
    Coordinate attention (Hou et al., CVPR 2021). Pools each channel along whole rows and whole columns
    separately, so every pixel is re-weighted by context from its full row and column. That suits long,
    thin lane lines, at a tiny fraction of the cost of full position attention (PAM).
    """
    def __init__(self, channels, reduction=8):
        super(CoordinateAttention, self).__init__()
        mid = max(8, channels // reduction)
        self.squeeze = ConvBNReLU(channels, mid, kernel_size=1, padding=0)
        self.attn_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.attn_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)                       # (B, C, H, 1): one value per row
        pooled_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # (B, C, W, 1): one value per column
        y = self.squeeze(torch.cat([pooled_h, pooled_w], dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.attn_h(y_h))                        # (B, C, H, 1)
        a_w = torch.sigmoid(self.attn_w(y_w.permute(0, 1, 3, 2)))    # (B, C, 1, W)
        return x * a_h * a_w


class EncoderV2(nn.Module):
    """
    Output stride 8 (stops downsampling at 45x80 for a 360x640 input) and returns the 1/2 and 1/4
    features as well, so the decoders can recover detail through skip connections.
    """
    def __init__(self):
        super(EncoderV2, self).__init__()
        self.init_conv = ConvBNReLU(3, 16, stride=2)                     # 1/2: 180x320
        self.stage1_down = DepthwiseSeparableConv(16, 32, stride=2)      # 1/4: 90x160
        self.stage1_ctx = HFFDilatedContextBlock(32)
        self.stage2_down = DepthwiseSeparableConv(32, 64, stride=2)      # 1/8: 45x80
        self.stage2_ctx = nn.Sequential(HFFDilatedContextBlock(64), HFFDilatedContextBlock(64))
        # Stage 3 widens channels without a 4th downsample; the dilated blocks supply the receptive field
        self.stage3_proj = DepthwiseSeparableConv(64, 128, stride=1)
        self.stage3_ctx = nn.Sequential(HFFDilatedContextBlock(128), HFFDilatedContextBlock(128),
                                        HFFDilatedContextBlock(128))
        self.attention = CoordinateAttention(128)

    def forward(self, x):
        f2 = self.init_conv(x)
        f4 = self.stage1_ctx(self.stage1_down(f2))
        f8 = self.stage2_ctx(self.stage2_down(f4))
        f8 = self.attention(self.stage3_ctx(self.stage3_proj(f8)))
        return f2, f4, f8


class SkipDecoderHead(nn.Module):
    """
    Upsamples x2 at a time, fusing the encoder feature of the same resolution at each step, and predicts
    one logit per pixel at `out_stride` (4, 2 or 1). Below full resolution only the 1-channel logits are
    bilinearly resized, so no heavy convolution runs at 360x640. out_stride 1 adds a learned x2 upsampling
    (transposed conv, as in TwinLiteNet's UPx2) so thin lanes are predicted at full resolution instead of
    being smeared ~2x wider by stretching a half-resolution output.
    """
    def __init__(self, out_stride):
        super(SkipDecoderHead, self).__init__()
        assert out_stride in (1, 2, 4)
        self.out_stride = out_stride
        self.reduce = ConvBNReLU(128, 64, kernel_size=1, padding=0)
        self.fuse4 = DepthwiseSeparableConv(64 + 32, 48)
        if out_stride <= 2:
            self.fuse2 = DepthwiseSeparableConv(48 + 16, 32)
        if out_stride == 1:
            self.up1 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False),
                                     nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d({4: 48, 2: 32, 1: 16}[out_stride], 1, kernel_size=1)

    def forward(self, f2, f4, f8, target_size):
        x = self.reduce(f8)
        x = F.interpolate(x, size=f4.shape[2:], mode='bilinear', align_corners=False)
        x = self.fuse4(torch.cat([x, f4], dim=1))
        if self.out_stride <= 2:
            x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
            x = self.fuse2(torch.cat([x, f2], dim=1))
        if self.out_stride == 1:
            x = self.up1(x)
        logits = self.classifier(x)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)


class TwinDriveV2(nn.Module):
    """
    TwinDrive v2. Outputs tuple: (Drivable_Logits, Lane_Logits, EgoLane_Logits), each (B, 1, H, W) raw logits.
    - Drivable and ego-lane heads stop at 1/4 resolution (large smooth regions need context, not detail).
    - The lane head goes to 1/2 resolution (thin lines need detail).
    - Ego lane = BDD's "direct" drivable area (the car's own lane), supervised on BDD frames only.
    lane_out_stride=1 gives the "v2lf" (lane-fix) variant with a full-resolution lane head.
    """
    def __init__(self, lane_out_stride=2):
        super(TwinDriveV2, self).__init__()
        self.encoder = EncoderV2()
        self.drivable_decoder = SkipDecoderHead(out_stride=4)
        self.lane_decoder = SkipDecoderHead(out_stride=lane_out_stride)
        self.ego_decoder = SkipDecoderHead(out_stride=4)

    def forward(self, x):
        target_size = (x.size(2), x.size(3))
        f2, f4, f8 = self.encoder(x)
        return (self.drivable_decoder(f2, f4, f8, target_size),
                self.lane_decoder(f2, f4, f8, target_size),
                self.ego_decoder(f2, f4, f8, target_size))


class TwinDriveV2LaneFix(TwinDriveV2):
    """TwinDrive v2 with a full-resolution lane head (arch name "v2lf")."""
    def __init__(self):
        super(TwinDriveV2LaneFix, self).__init__(lane_out_stride=1)


if __name__ == "__main__":
    for cls in (TwinDriveV2, TwinDriveV2LaneFix):
        model = cls().eval()
        print(f"{cls.__name__}: {sum(p.numel() for p in model.parameters()):,} trainable parameters")
        with torch.no_grad():
            outs = model(torch.randn(2, 3, 360, 640))
        for name, o in zip(["drivable", "lanes", "ego"], outs):
            assert o.shape == (2, 1, 360, 640), f"{name} head shape mismatch: {tuple(o.shape)}"
    print("TwinDriveV2 shape checks passed.")
