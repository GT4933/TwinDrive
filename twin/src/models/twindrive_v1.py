import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """Standard Convolution -> BatchNorm -> ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Crucial for edge devices (Jetson Nano) as it drastically reduces MAC operations.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise: Applies a single filter per input channel
        self.depthwise = ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        # Pointwise: Linear combination of the depthwise outputs
        self.pointwise = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class DilatedContextBlock(nn.Module):
    """
    Captures a wider field of view without losing resolution or adding parameters.
    Essential for unstructured roads where context (like trees vs. sky) dictates the drivable area.
    """
    def __init__(self, channels):
        super(DilatedContextBlock, self).__init__()
        self.reduce = ConvBNReLU(channels, channels // 2, kernel_size=1, padding=0)
        
        # Parallel dilated convolutions mimicking ESPNet-v2 structure
        self.d1 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=8, dilation=8, bias=False)
        
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.reduce(x)
        
        # Concatenate multi-scale features
        out1 = self.d1(x)
        out2 = self.d2(x)
        out4 = self.d4(x)
        out8 = self.d8(x)
        
        out = torch.cat([out1, out2, out4, out8], dim=1)
        out = self.bn(out)
        out = self.relu(out + residual)
        return out

class SharedEncoder(nn.Module):
    """
    Extracts core geometric and semantic features from the input frame.
    These features are then passed to both the Drivable Area and Lane line decoders.
    """
    def __init__(self):
        super(SharedEncoder, self).__init__()
        # Initial spatial downsampling (640x360 -> 320x180)
        self.init_conv = ConvBNReLU(3, 16, stride=2)
        
        # Stage 1 (320x180 -> 160x90)
        self.stage1_down = DepthwiseSeparableConv(16, 32, stride=2)
        self.stage1_ctx = DilatedContextBlock(32)
        
        # Stage 2 (160x90 -> 80x45)
        self.stage2_down = DepthwiseSeparableConv(32, 64, stride=2)
        self.stage2_ctx1 = DilatedContextBlock(64)
        self.stage2_ctx2 = DilatedContextBlock(64)
        
        # Stage 3 (Deepest feature representation) (80x45 -> 40x23 approx, matching padding)
        self.stage3_down = DepthwiseSeparableConv(64, 128, stride=2)
        self.stage3_ctx = nn.Sequential(
            DilatedContextBlock(128),
            DilatedContextBlock(128),
            DilatedContextBlock(128)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.stage1_down(x)
        x = self.stage1_ctx(x)
        x = self.stage2_down(x)
        x = self.stage2_ctx1(x)
        x = self.stage2_ctx2(x)
        x = self.stage3_down(x)
        x = self.stage3_ctx(x)
        return x

class DecoderHead(nn.Module):
    """
    Takes the deeply encoded, low-resolution features and mathematically upsamples them 
    back to the original camera resolution (640x360) using Bilinear Interpolation.
    """
    def __init__(self, in_channels, num_classes=1):
        super(DecoderHead, self).__init__()
        
        # Smooth and reduce channels before upsampling
        self.conv1 = ConvBNReLU(in_channels, 64, kernel_size=1, padding=0)
        
        # Transposed convolutions can cause checkerboard artifacts. 
        # Bilinear interpolation + standard conv is cleaner for ADAS tasks.
        self.conv2 = DepthwiseSeparableConv(64, 32)
        
        # Final pixel-wise logit prediction (no ReLU/BN here, raw logits for loss function)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, target_size):
        x = self.conv1(x)
        
        # Upsample back to target spatial dimensions (Height, Width)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.final_conv(x)
        
        return x

class TwinDriveV1(nn.Module):
    """
    TwinDrive v1 Architecture.
    Outputs tuple: (Drivable_Logits, Lane_Logits), each (B, 1, H, W) raw logits.
    Apply sigmoid to get per-pixel probabilities.
    """
    def __init__(self):
        super(TwinDriveV1, self).__init__()

        # Core feature extractor
        self.encoder = SharedEncoder()

        # Multi-task heads: one logit per pixel each (binary segmentation)
        self.drivable_decoder = DecoderHead(128, num_classes=1)
        self.lane_decoder = DecoderHead(128, num_classes=1)

    def forward(self, x):
        target_size = (x.size(2), x.size(3)) # Extract original Height and Width (360, 640)
        
        # 1. Extract shared spatial features
        encoded_features = self.encoder(x)
        
        # 2. Branch into independent tasks to prevent gradient conflict
        out_drivable = self.drivable_decoder(encoded_features, target_size)
        out_lanes = self.lane_decoder(encoded_features, target_size)
        
        return out_drivable, out_lanes

# Backward compatibility alias
TwinLiteNet = TwinDriveV1

if __name__ == "__main__":
    print("=" * 60)
    print("        TWINDRIVE V1 ARCHITECTURE VERIFICATION TEST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    
    model = TwinDriveV1().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} (Ultra-Lightweight)")
    
    dummy_input = torch.randn(2, 3, 360, 640).to(device)
    print(f"\nPassing dummy tensor {dummy_input.shape} through network...")
    
    with torch.no_grad():
        drivable_preds, lane_preds = model(dummy_input)
        
    print("\n[Forward Pass Successful]")
    print(f"Drivable Area Output Shape : {drivable_preds.shape}")
    print(f"Lane Mask Output Shape     : {lane_preds.shape}")
    
    assert drivable_preds.shape == (2, 1, 360, 640), "Drivable head spatial dimension mismatch!"
    assert lane_preds.shape == (2, 1, 360, 640), "Lane head spatial dimension mismatch!"
    
    print("\nAll architecture constraints verified. Network is ready.")
    print("=" * 60)

