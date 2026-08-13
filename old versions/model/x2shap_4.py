"""
X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays
完整版 - 4个尺度输出，参数量 ~40-50M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ 使用官方 mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError("请先安装 mamba_ssm")


# ============================================================
# 1. 2D MambaVision 编码器 (4个尺度输出)
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MambaVisionMixer(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        self.inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, 2 * self.inner_dim)
        self.conv1d = nn.Conv1d(self.inner_dim, self.inner_dim, 3, padding=1, groups=self.inner_dim)
        self.mamba = Mamba(d_model=self.inner_dim, d_state=16, d_conv=4, expand=1)
        self.out_proj = nn.Linear(self.inner_dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        xz = self.in_proj(x_flat)
        x_ssm, x_conv = xz.chunk(2, dim=-1)
        x_ssm = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm = self.conv1d(x_ssm)
        x_ssm = rearrange(x_ssm, 'b d l -> b l d')
        x_ssm = self.mamba(x_ssm)
        x_conv = F.silu(x_conv)
        x_out = self.out_proj(x_ssm + x_conv)
        return rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)


class MambaVisionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.norm1(x_flat)
        x_mixer = rearrange(self.mixer(x), 'b c h w -> b (h w) c')
        x_flat = x_flat + x_mixer
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)


class MambaVisionEncoder2D(nn.Module):
    """
    2D 编码器: 输出 4 个尺度的特征图
    尺寸: 128 -> 64 -> 32 -> 16 -> 8 (4次下采样，输出4个尺度)
    通道: [64, 128, 256, 512]
    """
    def __init__(self, in_chans=1, dims=[64, 128, 256, 512], depths=[1, 1, 2, 1]):
        super().__init__()
        self.dims = dims
        self.num_scales = len(dims)

        # Stem: 初始下采样 (128 -> 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        # 构建多阶段编码器
        self.stages = nn.ModuleList()
        current_dim = dims[0]

        for i in range(self.num_scales):
            # MambaVision 块序列
            stage_blocks = nn.Sequential()
            for _ in range(depths[i]):
                stage_blocks.append(MambaVisionBlock(current_dim))
            self.stages.append(stage_blocks)

            # 下采样 (除了最后一个阶段)
            if i < self.num_scales - 1:
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(current_dim, dims[i+1], kernel_size=2, stride=2),
                        LayerNorm2d(dims[i+1]),
                        nn.GELU()
                    )
                )
                current_dim = dims[i+1]

    def forward(self, x):
        """
        返回: 列表，包含 4 个尺度的特征图
              [0]: 64@64² (stage0 输出)
              [1]: 128@32² (stage1 输出)
              [2]: 256@16² (stage2 输出)
              [3]: 512@8² (stage3 输出)
        """
        x = self.stem(x)
        features = []

        for layer in self.stages:
            x = layer(x)
            # 判断是否是 MambaVision 块序列 (不是下采样层)
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                if not isinstance(layer[0], nn.Conv2d):
                    features.append(x)

        return features  # [64@64², 128@32², 256@16², 512@8²]


# ============================================================
# 2. 体积反投影模块 (VBP) - 接收4个尺度输入
# ============================================================

class VolumetricBackprojection(nn.Module):
    """
    VBP: 接收多尺度 2D 特征 (4个尺度)，输出单一尺度 3D 特征
    """
    def __init__(self, multi_scale_channels, out_channels, volume_shape=(32, 32, 32)):
        super().__init__()
        self.multi_scale_channels = multi_scale_channels  # [64, 128, 256, 512]
        self.out_channels = out_channels
        self.volume_shape = volume_shape
        self.num_scales = len(multi_scale_channels)

        # 多尺度特征融合: 将不同尺度的 2D 特征转换到统一通道
        self.fusion_convs = nn.ModuleList()
        for i, ch in enumerate(multi_scale_channels):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 3D 投影: 将融合后的 2D 特征投影到 3D 空间
        self.project_3d = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels)
        )

        # 可学习的位置权重
        self.position_weights = nn.Parameter(torch.ones(1, 1, volume_shape[0], volume_shape[1], volume_shape[2]))

    def forward(self, f_multi_scale):
        """
        f_multi_scale: 列表，包含 4 个尺度的 2D 特征
                       [64@64², 128@32², 256@16², 512@8²]
        返回: (B, out_channels, D, H, W) 3D 特征体积
        """
        B = f_multi_scale[0].shape[0]
        D, H_vol, W_vol = self.volume_shape
        target_h = H_vol
        target_w = W_vol

        # Step 1: 将所有尺度的 2D 特征上采样到统一尺寸
        fused_2d = []
        for i, f in enumerate(f_multi_scale):
            if f.shape[-2:] != (target_h, target_w):
                f_up = F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
            else:
                f_up = f
            f_proj = self.fusion_convs[i](f_up)
            fused_2d.append(f_proj)

        # Step 2: 融合多尺度特征 (求和)
        fused_2d = sum(fused_2d)  # (B, out_channels, H_vol, W_vol)

        # Step 3: 投影到 3D 空间
        f_3d = fused_2d[:, :, None, :, :].expand(-1, -1, D, -1, -1)

        # Step 4: 3D 卷积细化
        out = self.project_3d(f_3d)

        # Step 5: 应用位置权重
        pos_weight = self.position_weights.expand(B, -1, -1, -1, -1)
        out = out * torch.sigmoid(pos_weight)

        return out


# ============================================================
# 3. 3D Encoder (产生4个尺度的特征金字塔)
# ============================================================

class Encoder3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
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
    3D Encoder: 处理 VBP 输出的 3D 特征
    输出4个尺度的特征金字塔
    """
    def __init__(self, in_chans, channels=[64, 128, 256, 512]):
        super().__init__()
        self.channels = channels

        self.blocks = nn.ModuleList()
        curr_ch = in_chans

        for i, out_ch in enumerate(channels):
            downsample = (i < len(channels) - 1)
            self.blocks.append(Encoder3DBlock(curr_ch, out_ch, downsample))
            curr_ch = out_ch

    def forward(self, x):
        """
        x: (B, C, D, H, W) 基础 3D 特征体积
        返回: 从深到浅的特征列表 [512(小尺寸), 256, 128, 64(大尺寸)]
        """
        features = []
        for block in self.blocks:
            x, feat = block(x)
            features.append(feat)
        return features[::-1]  # [512(8³), 256(16³), 128(32³), 64(64³)]


# ============================================================
# 4. Cross-Mamba 模块 (4个)
# ============================================================

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm_ap = nn.LayerNorm(dim)
        self.norm_lat = nn.LayerNorm(dim)
        self.mamba_ap = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_lat = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.gate_ap = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.gate_lat = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, f_ap, f_lat):
        B, C, D, H, W = f_ap.shape

        f_ap_flat = rearrange(f_ap, 'b c d h w -> b (d h w) c')
        f_lat_flat = rearrange(f_lat, 'b c d h w -> b (d h w) c')

        y1 = self.mamba_ap(self.norm_ap(f_ap_flat))
        g1 = self.gate_ap(self.norm_ap(f_lat_flat))
        out1 = rearrange(y1 * g1, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        y2 = self.mamba_lat(self.norm_lat(f_lat_flat))
        g2 = self.gate_lat(self.norm_lat(f_ap_flat))
        out2 = rearrange(y2 * g2, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        return out1 + out2


# ============================================================
# 5. 残差块 (U-Net 升采样路径) - 4层
# ============================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, current_feat, skip_feat):
        if skip_feat.shape[-3:] != current_feat.shape[-3:]:
            skip_feat = F.interpolate(skip_feat, size=current_feat.shape[-3:],
                                      mode='trilinear', align_corners=False)
        combined = torch.cat([current_feat, skip_feat], dim=1)
        out = self.conv(combined)
        out = out + current_feat
        out = self.relu(out)
        return out


# ============================================================
# 6. 分割头 (双通道输出)
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 7. 完整的 X2Shape 模型 (4个尺度)
# ============================================================

class X2Shape(nn.Module):
    def __init__(self, img_size=128, in_chans=1, num_classes=2,
                 dims_2d=[64, 128, 256, 512], depths_2d=[1, 1, 2, 1],
                 dims_3d=[64, 128, 256, 512], vbp_output_channels=64):
        super().__init__()
        self.img_size = img_size
        self.num_levels = len(dims_3d)  # 4层

        # 体积尺寸: 64 -> 32 -> 16 -> 8
        vol_sizes = [img_size // 2, img_size // 4, img_size // 8, img_size // 16]

        # ========== 1. 两个独立的 2D 编码器 ==========
        self.encoder_2d_ap = MambaVisionEncoder2D(in_chans, dims_2d, depths_2d)
        self.encoder_2d_lat = MambaVisionEncoder2D(in_chans, dims_2d, depths_2d)

        # ========== 2. VBP 模块 ==========
        self.vbp_ap = VolumetricBackprojection(dims_2d, vbp_output_channels, (vol_sizes[0], vol_sizes[0], vol_sizes[0]))
        self.vbp_lat = VolumetricBackprojection(dims_2d, vbp_output_channels, (vol_sizes[0], vol_sizes[0], vol_sizes[0]))

        # ========== 3. 两个独立的 3D Encoder ==========
        self.encoder_3d_ap = Encoder3D(vbp_output_channels, dims_3d)
        self.encoder_3d_lat = Encoder3D(vbp_output_channels, dims_3d)

        # ========== 4. Cross-Mamba 模块 (4个) ==========
        self.cross_mambas = nn.ModuleList([
            CrossMamba(512),  # 最深尺度 (8³)
            CrossMamba(256),  # 次深尺度 (16³)
            CrossMamba(128),  # 次浅尺度 (32³)
            CrossMamba(64)    # 最浅尺度 (64³)
        ])

        # ========== 5. 上采样层 (3个，从深到浅) ==========
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),  # 8³ -> 16³
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),  # 16³ -> 32³
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)    # 32³ -> 64³
        ])

        # ========== 6. 残差块 (3个) ==========
        self.res_blocks = nn.ModuleList([
            ResidualBlock3D(256),
            ResidualBlock3D(128),
            ResidualBlock3D(64)
        ])

        # ========== 7. 最终残差块 ==========
        self.final_res_block = ResidualBlock3D(64)

        # ========== 8. 分割头 ==========
        self.seg_head = SegmentationHead(64, num_classes)

        # ========== 9. 原始 VBP 跳跃连接通道调整 ==========
        self.vbp_jump_adjust = nn.Conv3d(vbp_output_channels, 64, kernel_size=1)

    def forward(self, x_ap, x_lat):
        # Step 1: 2D 编码 (4个尺度)
        f_ap_2d = self.encoder_2d_ap(x_ap)  # [64@64², 128@32², 256@16², 512@8²]
        f_lat_2d = self.encoder_2d_lat(x_lat)

        # Step 2: VBP (多尺度 2D -> 单一尺度 3D)
        f_ap_3d_base = self.vbp_ap(f_ap_2d)   # (B, 64, 64, 64, 64)
        f_lat_3d_base = self.vbp_lat(f_lat_2d)

        vbp_jump_ap = f_ap_3d_base
        vbp_jump_lat = f_lat_3d_base

        # Step 3: 3D Encoder (多尺度金字塔)
        # 输出: [512@8³, 256@16³, 128@32³, 64@64³]
        f_ap_3d_multi = self.encoder_3d_ap(f_ap_3d_base)
        f_lat_3d_multi = self.encoder_3d_lat(f_lat_3d_base)

        # Step 4: Cross-Mamba (对应尺度分发)
        cross_outputs = []
        for i in range(self.num_levels):
            cross_feat = self.cross_mambas[i](f_ap_3d_multi[i], f_lat_3d_multi[i])
            cross_outputs.append(cross_feat)

        # Step 5: U-Net 升采样路径 (从深到浅)
        current_feat = cross_outputs[0]  # (512, 8³)

        for i in range(1, self.num_levels):
            upsampled = self.upsample_layers[i-1](current_feat)
            current_cross = cross_outputs[i]

            if upsampled.shape[-3:] != current_cross.shape[-3:]:
                upsampled = F.interpolate(upsampled, size=current_cross.shape[-3:],
                                          mode='trilinear', align_corners=False)

            current_feat = self.res_blocks[i-1](current_cross, upsampled)

        # Step 6: 最终残差块 (与原始 VBP 特征融合)
        vbp_jump = vbp_jump_ap + vbp_jump_lat

        if vbp_jump.shape[1] != current_feat.shape[1]:
            vbp_jump = self.vbp_jump_adjust(vbp_jump)

        if current_feat.shape[-3:] != vbp_jump.shape[-3:]:
            current_feat = F.interpolate(current_feat, size=vbp_jump.shape[-3:],
                                         mode='trilinear', align_corners=False)

        final_feat = self.final_res_block(current_feat, vbp_jump)

        # Step 7: 分割头
        output = self.seg_head(final_feat)

        # 上采样到原始尺寸 (64³ -> 128³)
        if output.shape[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size, self.img_size),
                                   mode='trilinear', align_corners=False)

        return output


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("X2Shape 模型测试 (4个尺度，完整版)")
    print("=" * 60)

    model = X2Shape(
        img_size=128, in_chans=1, num_classes=2,
        dims_2d=[64, 128, 256, 512],
        depths_2d=[1, 1, 2, 1],
        dims_3d=[64, 128, 256, 512],
        vbp_output_channels=64
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.2f}M")

    # 完整统计各模块参数量
    enc_2d_params = sum(p.numel() for p in model.encoder_2d_ap.parameters())
    vbp_params = sum(p.numel() for p in model.vbp_ap.parameters())
    enc_3d_params = sum(p.numel() for p in model.encoder_3d_ap.parameters())
    cross_params = sum(p.numel() for p in model.cross_mambas.parameters())

    # 遗漏模块
    upsample_params = sum(p.numel() for p in model.upsample_layers.parameters())
    res_block_params = sum(p.numel() for p in model.res_blocks.parameters())
    final_res_params = sum(p.numel() for p in model.final_res_block.parameters())
    seg_head_params = sum(p.numel() for p in model.seg_head.parameters())
    vbp_jump_params = sum(p.numel() for p in model.vbp_jump_adjust.parameters())

    print(f"\n模块参数量:")
    print(f"  2D Encoder: {enc_2d_params / 1e6:.2f}M")
    print(f"  VBP: {vbp_params / 1e6:.2f}M")
    print(f"  3D Encoder: {enc_3d_params / 1e6:.2f}M")
    print(f"  Cross-Mamba (4个): {cross_params / 1e6:.2f}M")
    print(f"  Upsample Layers (3个): {upsample_params / 1e6:.2f}M")
    print(f"  Residual Blocks (3个): {res_block_params / 1e6:.2f}M")
    print(f"  Final Residual Block: {final_res_params / 1e6:.2f}M")
    print(f"  Segmentation Head: {seg_head_params / 1e6:.2f}M")
    print(f"  VBP Jump Adjust: {vbp_jump_params / 1e6:.2f}M")

    # 验证总和
    calculated_total = (enc_2d_params + vbp_params + enc_3d_params + cross_params +
                        upsample_params + res_block_params + final_res_params +
                        seg_head_params + vbp_jump_params)
    print(f"\n计算总参数量: {calculated_total / 1e6:.2f}M")
    print(f"实际总参数量: {total_params / 1e6:.2f}M")

    # 显存估算
    print(f"\n显存估算:")
    print(f"  模型参数: {total_params / 1e6:.2f}M × 4字节 ≈ {total_params * 4 / 1024 ** 2:.0f}MB")
    print(f"  梯度: ≈ {total_params * 4 / 1024 ** 2:.0f}MB")
    print(f"  优化器状态: ≈ {total_params * 8 / 1024 ** 2:.0f}MB")
    print(f"  激活值: 约 2-3GB")
    print(f"  总计: 约 {total_params * 16 / 1024 ** 2 + 2500:.0f}MB")

    # 测试前向传播
    batch_size = 2
    x_ap = torch.randn(batch_size, 1, 128, 128).to(device)
    x_lat = torch.randn(batch_size, 1, 128, 128).to(device)

    print(f"\n前向传播测试...")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(x_ap, x_lat)

    print(f"输入 AP: {x_ap.shape}")
    print(f"输出: {output.shape}")

    expected = (batch_size, 2, 128, 128, 128)
    if output.shape == expected:
        print("\n✅ 测试通过！输出为双通道 (左, 右)")
    else:
        print(f"\n❌ 形状错误: 期望 {expected}, 实际 {output.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")