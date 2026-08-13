"""
模型2: 3D U-Net分割模型 (Cross-Mamba在深层使用)
输入: 子图 (B, C, S, S, S) 其中 B = 子图数量
输出: 分割结果 (B, num_classes, S, S, S)

Cross-Mamba位置:
- 在 decoder_0 (最深層, 空间尺寸 S/8) 使用，序列长度最小
- 其他层使用标准特征拼接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time
#from VBP_unshuffle import VBPSubvolumeExtractor
# ============================================================
# 1. 归一化层
# ============================================================

class InstanceNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps=eps, affine=affine)

    def forward(self, x):
        return self.instance_norm(x)


# ============================================================
# 2. 3D编码器 (4层)
# ============================================================

class Encoder3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        feat = x
        if self.downsample:
            x = self.pool(x)
        return x, feat


class Encoder3D(nn.Module):
    """
    4层U-Net编码器
    输入: (B, C, S, S, S)
    输出: 4个尺度的跳跃连接 (从深到浅)
    """
    def __init__(self, in_chans, channels=[32, 64, 128, 256]):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList()
        curr_ch = in_chans

        for out_ch in channels:
            downsample = (len(self.blocks) < len(channels) - 1)
            self.blocks.append(Encoder3DBlock(curr_ch, out_ch, downsample))
            curr_ch = out_ch

    def forward(self, x):
        skip_features = []
        for block in self.blocks:
            x, feat = block(x)
            skip_features.append(feat)
        return skip_features[::-1]


# ============================================================
# 3. Cross-Mamba 模块 (在深层使用，序列长度小)
# ============================================================

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ 使用官方 mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️  mamba_ssm未安装，使用替代方案")

    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)


class CrossMamba(nn.Module):
    """
    Cross-Mamba模块：跨空间位置的特征融合
    在深层使用 (空间尺寸小，序列长度短)
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, X_dim, Y_dim, Z_dim = x.shape

        # 展平为序列
        x_flat = rearrange(x, 'b c x y z -> b (x y z) c')

        # Mamba处理
        y = self.mamba(self.norm(x_flat))
        g = self.gate(self.norm(x_flat))

        # 门控融合
        out = y * g

        # 恢复形状
        out = rearrange(out, 'b (x y z) c -> b c x y z',
                        x=X_dim, y=Y_dim, z=Z_dim)

        return out


# ============================================================
# 4. U-Net 解码器 (Cross-Mamba在深层)
# ============================================================

class DecoderBlock(nn.Module):
    """
    标准U-Net解码器块：上采样 + 特征拼接 + 卷积
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, skip_feat):
        x = self.upsample(x)

        if x.shape[-3:] != skip_feat.shape[-3:]:
            x = F.interpolate(x, size=skip_feat.shape[-3:],
                              mode='trilinear', align_corners=False)

        x = torch.cat([x, skip_feat], dim=1)
        x = self.conv(x)

        return x


class DecoderBlockWithCrossMamba(nn.Module):
    """
    带Cross-Mamba的解码器块 (在深层使用)
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Cross-Mamba在最深层使用 (空间尺寸 S/8，序列长度最小)
        self.cross_mamba = CrossMamba(out_channels)

    def forward(self, x, skip_feat):
        x = self.upsample(x)

        if x.shape[-3:] != skip_feat.shape[-3:]:
            x = F.interpolate(x, size=skip_feat.shape[-3:],
                              mode='trilinear', align_corners=False)

        x = torch.cat([x, skip_feat], dim=1)
        x = self.conv(x)

        # Cross-Mamba (深层)
        x = self.cross_mamba(x)

        return x


class UNetDecoder(nn.Module):
    """
    4层U-Net解码器

    Cross-Mamba位置: decoder_0 (最深層, 空间尺寸 S/8)
    其他层: 标准U-Net

    对应关系:
        decoder_0 (带Cross-Mamba) ← skip_features[0] (256@S/8³)  最深
        decoder_1 (标准)          ← skip_features[1] (128@S/4³)
        decoder_2 (标准)          ← skip_features[2] (64@S/2³)
        decoder_3 (标准)          ← skip_features[3] (32@S³)      最浅
    """
    def __init__(self, channels=[256, 128, 64, 32]):
        super().__init__()

        # decoder_0: 最深 (S/8³ -> S/4³) - 带Cross-Mamba
        self.decoder_0 = DecoderBlockWithCrossMamba(
            in_channels=channels[0],
            skip_channels=channels[1],
            out_channels=channels[1]
        )

        # decoder_1: S/4³ -> S/2³ - 标准U-Net
        self.decoder_1 = DecoderBlock(
            in_channels=channels[1],
            skip_channels=channels[2],
            out_channels=channels[2]
        )

        # decoder_2: S/2³ -> S³ - 标准U-Net
        self.decoder_2 = DecoderBlock(
            in_channels=channels[2],
            skip_channels=channels[3],
            out_channels=channels[3]
        )

    def forward(self, x, skip_features):
        x = self.decoder_0(x, skip_features[1])   # S/8 -> S/4 (带Cross-Mamba)
        x = self.decoder_1(x, skip_features[2])   # S/4 -> S/2 (标准)
        x = self.decoder_2(x, skip_features[3])   # S/2 -> S (标准)
        return x


# ============================================================
# 5. 分割头
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            InstanceNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            InstanceNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 6. 完整的模型
# ============================================================

class SubvolumeUNet(nn.Module):
    """
    4层U-Net分割模型 (Cross-Mamba在深层)
    输入: (B, C, S, S, S)
    输出: (B, num_classes, S, S, S)
    """
    def __init__(self, in_chans=1, num_classes=1, subvolume_size=128,
                 encoder_channels=[32, 64, 128, 256]):
        super().__init__()

        self.subvolume_size = subvolume_size

        self.encoder = Encoder3D(in_chans, encoder_channels)
        self.decoder = UNetDecoder(channels=encoder_channels[::-1])
        self.seg_head = SegmentationHead(encoder_channels[0], num_classes)

    def forward(self, x):
        skip_features = self.encoder(x)
        deepest_feat = skip_features[0]
        decoded_feat = self.decoder(deepest_feat, skip_features)
        output = self.seg_head(decoded_feat)
        return output


# ============================================================
# 7. 子图重组模块
# ============================================================

class SubvolumeReconstructor(nn.Module):
    def __init__(self, subvolume_size=128, scale_factor=2):
        super().__init__()
        self.subvolume_size = subvolume_size
        self.scale_factor = scale_factor
        self.output_size = subvolume_size * scale_factor

    def forward(self, subvolumes):
        from einops import rearrange
        B_sub, C, X, Y, Z = subvolumes.shape
        s = self.scale_factor
        B = B_sub // (s ** 3)

        volume = rearrange(
            subvolumes,
            '(b xp yp zp) c x y z -> b c (x xp) (y yp) (z zp)',
            b=B, xp=s, yp=s, zp=s
        )

        return volume


# ============================================================
# 8. 主测试函数
# ============================================================

def test_subvolume_unet():
    print("=" * 70)
    print("3D U-Net分割模型测试 (Cross-Mamba在深层)")
    print("=" * 70)

    # ============================================================
    # 🔧 配置参数
    # ============================================================
    SUBVOLUME_SIZE = 128
    BATCH_SIZE = 4
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    ENCODER_CHANNELS = [16, 32, 64, 128]
    USE_CUDA = True
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

    print("\n" + "-" * 70)
    print("模型配置")
    print("-" * 70)
    print(f"  子图尺寸:           {SUBVOLUME_SIZE}³")
    print(f"  子图批次大小:       {BATCH_SIZE}")
    print(f"  输入通道数:         {IN_CHANNELS}")
    print(f"  输出类别数:         {NUM_CLASSES}")
    print(f"  编码器通道:         {ENCODER_CHANNELS}")
    print(f"  Cross-Mamba位置:    深层 (decoder_0, 空间尺寸 {SUBVOLUME_SIZE//8}³)")
    print(f"  设备:               {device}")
    print("-" * 70)

    print("\nU-Net结构:")
    print("  编码器:  32@S³ → 64@S/2³ → 128@S/4³ → 256@S/8³")
    print("  解码器:  256@S/8³ → 128@S/4³ → 64@S/2³ → 32@S³")
    print("  Cross-Mamba: 在 decoder_0 (S/8³) 使用")

    print("\n创建模型...")
    model = SubvolumeUNet(
        in_chans=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        subvolume_size=SUBVOLUME_SIZE,
        encoder_channels=ENCODER_CHANNELS
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量:           {total_params / 1e6:.2f}M")

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    seg_head_params = sum(p.numel() for p in model.seg_head.parameters())

    print(f"  编码器参数量:       {encoder_params / 1e6:.2f}M")
    print(f"  解码器参数量:       {decoder_params / 1e6:.2f}M")
    print(f"  分割头参数量:       {seg_head_params / 1e6:.2f}M")

    print("\n创建测试输入...")
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SUBVOLUME_SIZE, SUBVOLUME_SIZE, SUBVOLUME_SIZE).to(device)
    print(f"  输入形状:           {x.shape}")

    print("\n运行前向传播...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()

    if torch.cuda.is_available():
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    with torch.no_grad():
        output = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    print("\n" + "-" * 70)
    print("前向传播结果")
    print("-" * 70)
    print(f"  输出形状:           {output.shape}")
    print(f"  输出范围:           [{output.min():.4f}, {output.max():.4f}]")
    print(f"  推理时间:           {elapsed:.4f}s")
    print(f"  每子图时间:         {elapsed / BATCH_SIZE:.4f}s")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"  推理后 显存: {allocated:.2f}GB (已分配), {cached:.2f}GB (缓存), {peak:.2f}GB (峰值)")

    expected = (BATCH_SIZE, NUM_CLASSES, SUBVOLUME_SIZE, SUBVOLUME_SIZE, SUBVOLUME_SIZE)
    print("-" * 70)
    if output.shape == expected:
        print("✅ 形状验证通过！")
    else:
        print(f"❌ 形状错误: 期望 {expected}, 实际 {output.shape}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    test_subvolume_unet()