"""
X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays
简化版U-Net架构 - 3层2D编码器 + 3层3D编码器 + 2层解码器

架构说明：
1. AP和LAT分别经过3层2D编码和VBP反投影
2. 两个VBP输出的体积相加并归一化（双视图融合）-> 单一融合体积 (64³)
3. 融合后的体积通过3层3D编码器产生多尺度特征
4. 2层解码器使用反卷积上采样 + 特征拼接（跳跃连接）
5. 解码器输出与VBP跳跃连接融合后输出

VBP路径：
- 主路径：VBP融合体积 → 3D编码器 → U-Net解码器 → 解码器输出
- 跳跃连接路径：VBP融合体积 → 直接作为跳跃连接 → 与解码器输出融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
import os
import nibabel as nib
from datetime import datetime

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ 使用官方 mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError("请先安装 mamba_ssm")


# ============================================================
# 1. 归一化层
# ============================================================

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps, affine=affine)

    def forward(self, x):
        return self.instance_norm(x)


class InstanceNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps=eps, affine=affine)

    def forward(self, x):
        return self.instance_norm(x)


# ============================================================
# 2. 2D MambaVision 编码器 (3层)
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
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = rearrange(x, 'b c h w -> b (h w) c')
            return_4d = True
        else:
            B, N, C = x.shape
            x_flat = x
            return_4d = False

        xz = self.in_proj(x_flat)
        x_ssm, x_conv = xz.chunk(2, dim=-1)

        x_ssm = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm = self.conv1d(x_ssm)
        x_ssm = rearrange(x_ssm, 'b d l -> b l d')
        x_ssm = self.mamba(x_ssm)

        x_conv = F.silu(x_conv)
        x_out = self.out_proj(x_ssm + x_conv)

        if return_4d:
            return rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            return x_out


class SelfAttention2D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = rearrange(x, 'b c h w -> b (h w) c')
            return_4d = True
        else:
            B, N, C = x.shape
            x_flat = x
            return_4d = False
            H = W = int(math.sqrt(N))

        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if return_4d:
            return rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            return x_out


class MambaVisionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, use_attention=False, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if use_attention:
            self.mixer = SelfAttention2D(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        else:
            self.mixer = MambaVisionMixer(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        x_flat = x_flat + self.mixer(self.norm1(x_flat))
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)


class MambaVisionEncoder2D(nn.Module):
    """
    2D MambaVision编码器 (3层)
    输入: (B, 1, 256, 256)
    输出: 3个尺度的特征图
        - level_0: (B, 32, 128, 128)
        - level_1: (B, 64, 64, 64)
        - level_2: (B, 128, 32, 32)
    """
    def __init__(self, in_chans=1, dims=[32, 64, 128], depths=[1, 1, 2],
                 use_attention_layers=[False, True, True], dropout=0.0):
        super().__init__()
        self.dims = dims
        self.num_scales = len(dims)

        # Stem: 初始下采样 (256 -> 128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        # 构建多阶段编码器
        self.stages = nn.ModuleList()
        current_dim = dims[0]

        for i in range(self.num_scales):
            # 每个阶段包含多个MambaVisionBlock
            stage_blocks = nn.Sequential()
            for j in range(depths[i]):
                use_attn = use_attention_layers[i] if i < len(use_attention_layers) else False
                stage_blocks.append(
                    MambaVisionBlock(current_dim, use_attention=use_attn,
                                   num_heads=8, dropout=dropout)
                )
            self.stages.append(stage_blocks)

            # 下采样到下一阶段 (除了最后一层)
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
        x = self.stem(x)  # (B, dims[0], 128, 128)
        features = []

        for layer in self.stages:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                if not isinstance(layer[0], nn.Conv2d):
                    features.append(x)

        return features  # [32@128², 64@64², 128@32²]


# ============================================================
# 3. 体积反投影模块 (VBP)
# ============================================================

class LocalFeatureExpand(nn.Module):
    def __init__(self, in_channels, out_channels, volume_shape):
        super().__init__()
        self.volume_shape = volume_shape

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.refine_3d = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f_2d, proj_type, angle_deg, volume_shape):
        B, C, H, W = f_2d.shape
        X_dim, Y_dim, Z_dim = volume_shape

        f_proc = self.expand_conv(f_2d)

        if proj_type == 'ap':
            f_3d = f_proc[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)
        else:
            angle_rad = angle_deg * math.pi / 180.0
            tan_angle = math.tan(angle_rad)

            f_3d = torch.zeros(B, C, X_dim, Y_dim, Z_dim, device=f_proc.device)
            y_center = (Y_dim - 1) / 2

            for y in range(Y_dim):
                offset_z = int(-(y - y_center) * tan_angle)
                offset_z = max(min(offset_z, Z_dim - 1), -(Z_dim - 1))

                if offset_z != 0:
                    shifted = torch.roll(f_proc, shifts=offset_z, dims=3)
                    if offset_z > 0:
                        shifted[:, :, :, :offset_z] = 0
                    else:
                        shifted[:, :, :, offset_z:] = 0
                else:
                    shifted = f_proc

                f_3d[:, :, :, y, :] = shifted

            y_weights = torch.zeros(Y_dim, device=f_proc.device)
            for y in range(Y_dim):
                y_norm = (y - y_center) / (Y_dim / 2)
                y_weights[y] = math.exp(-y_norm ** 2 * 2)
            y_weights = y_weights.view(1, 1, 1, Y_dim, 1)
            f_3d = f_3d * y_weights

        f_3d = self.refine_3d(f_3d)
        return f_3d


class GlobalFeatureExpand(nn.Module):
    def __init__(self, feat_dim, embed_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f_ap_3d, f_lat_3d):
        B, C, X_dim, Y_dim, Z_dim = f_ap_3d.shape
        N = X_dim * Y_dim * Z_dim

        f_ap_flat = rearrange(f_ap_3d, 'b c x y z -> b (x y z) c')
        f_lat_flat = rearrange(f_lat_3d, 'b c x y z -> b (x y z) c')

        x_coords = torch.linspace(-1, 1, X_dim, device=f_ap_3d.device)
        y_coords = torch.linspace(-1, 1, Y_dim, device=f_ap_3d.device)
        z_coords = torch.linspace(-1, 1, Z_dim, device=f_ap_3d.device)

        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        pos_encoding = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1)

        pos_embed = self.pos_mlp(pos_encoding)

        concat_feat = torch.cat([f_ap_flat, f_lat_flat, pos_embed], dim=-1)
        feat_embed = self.fusion_mlp(concat_feat)

        attn_scores = torch.norm(feat_embed, dim=-1, keepdim=True)
        attn_weights = F.softmax(attn_scores, dim=1)

        weighted_sum = torch.sum(attn_weights * feat_embed, dim=1, keepdim=True)

        global_feat = self.output_mlp(weighted_sum)
        global_feat = global_feat.expand(-1, N, -1)

        global_feat_3d = rearrange(global_feat, 'b (x y z) c -> b c x y z',
                                   x=X_dim, y=Y_dim, z=Z_dim)

        return global_feat_3d


class VolumetricBackprojection(nn.Module):
    def __init__(self, multi_scale_channels, out_channels, volume_shape=(64, 64, 64),
                 embed_dim=64, dropout=0.0):
        super().__init__()
        self.multi_scale_channels = multi_scale_channels
        self.out_channels = out_channels
        self.volume_shape = volume_shape
        self.num_scales = len(multi_scale_channels)
        self.embed_dim = embed_dim

        self.fusion_convs = nn.ModuleList()
        for ch in multi_scale_channels:
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                    InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            )

        self.local_expand = LocalFeatureExpand(out_channels, out_channels, volume_shape)
        self.global_expand = GlobalFeatureExpand(out_channels, embed_dim)

        self.final_fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        X, Y, Z = volume_shape
        self.position_weights = nn.Parameter(torch.ones(1, 1, X, Y, Z))

    def forward(self, f_multi_scale, angle, proj_type='ap', f_other_view=None):
        B = f_multi_scale[0].shape[0]
        X_dim, Y_dim, Z_dim = self.volume_shape

        if isinstance(angle, torch.Tensor):
            if angle.numel() == 1:
                angle_deg = angle.item()
            else:
                angle_deg = angle[0].item() if len(angle) > 0 else 90.0
        else:
            angle_deg = float(angle)

        fused_2d = []
        for i, f in enumerate(f_multi_scale):
            if f.shape[-2:] != (X_dim, Z_dim):
                f_up = F.interpolate(f, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)
            else:
                f_up = f
            f_proj = self.fusion_convs[i](f_up)
            fused_2d.append(f_proj)

        fused_2d = sum(fused_2d)

        local_feat = self.local_expand(fused_2d, proj_type, angle_deg, self.volume_shape)

        if f_other_view is not None:
            global_feat = self.global_expand(local_feat, f_other_view)
        else:
            global_feat = local_feat

        combined = torch.cat([local_feat, global_feat], dim=1)
        out = self.final_fusion(combined)

        pos_weight = self.position_weights.expand(B, -1, -1, -1, -1)
        out = out * torch.sigmoid(pos_weight)

        return out


# ============================================================
# 4. 3层U-Net 3D编码器
# ============================================================

class Encoder3DBlock(nn.Module):
    """U-Net编码器块：卷积 + 下采样"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        # 保存跳跃连接 (下采样前的特征)
        feat = self.conv(x)
        # 下采样
        x_down = self.pool(feat)
        return x_down, feat


class Encoder3D(nn.Module):
    """
    3层U-Net 3D编码器
    输入: (B, C, 64, 64, 64)
    输出: 3个尺度的跳跃连接特征 (从深到浅)
        - level_0 (最深): (B, 256, 16, 16, 16)
        - level_1: (B, 128, 32, 32, 32)
        - level_2 (最浅): (B, 64, 64, 64, 64)
    """
    def __init__(self, in_chans, channels=[64, 128, 256]):
        super().__init__()
        self.channels = channels

        self.blocks = nn.ModuleList()
        curr_ch = in_chans

        for out_ch in channels:
            self.blocks.append(Encoder3DBlock(curr_ch, out_ch))
            curr_ch = out_ch

    def forward(self, x):
        skip_features = []
        for block in self.blocks:
            x, feat = block(x)
            skip_features.append(feat)
        # 返回从深到浅: [256@16³, 128@32³, 64@64³]
        return skip_features[::-1]


# ============================================================
# 5. 2层U-Net 解码器 (上采样 + 特征拼接)
# ============================================================

class DecoderBlock(nn.Module):
    """
    U-Net解码器块：上采样 + 特征拼接 + 卷积
    """
    def __init__(self, x_in, skip_in, x_out):
        super().__init__()
        # 上采样: x_in -> x_out
        self.upsample = nn.ConvTranspose3d(x_in, x_out, kernel_size=2, stride=2)

        # 卷积: (x_out + skip_in) -> x_out
        self.conv = nn.Sequential(
            nn.Conv3d(x_out + skip_in, x_out, kernel_size=3, padding=1),
            InstanceNorm3d(x_out),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(x_out, x_out, kernel_size=3, padding=1),
            InstanceNorm3d(x_out),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, skip_feat):
        # 1. 上采样
        x = self.upsample(x)

        # 2. 调整尺寸匹配
        if x.shape[-3:] != skip_feat.shape[-3:]:
            x = F.interpolate(x, size=skip_feat.shape[-3:],
                            mode='trilinear', align_corners=False)

        # 3. 特征拼接 (跳跃连接)
        x = torch.cat([x, skip_feat], dim=1)

        # 4. 卷积精炼
        x = self.conv(x)

        return x


class UNetDecoder(nn.Module):
    """
    2层U-Net解码器：使用跳跃连接逐步恢复分辨率

    skip_features: [256@16³, 128@32³, 64@64³] (从深到浅)

    解码过程:
        level_0: 256@16³ -> 上采样 + 拼接128@32³ -> 128@32³
        level_1: 128@32³ -> 上采样 + 拼接64@64³ -> 64@64³
    """
    def __init__(self):
        super().__init__()

        self.decoder_0 = DecoderBlock(
            x_in=256, skip_in=128, x_out=128
        )
        self.decoder_1 = DecoderBlock(
            x_in=128, skip_in=64, x_out=64
        )

    def forward(self, x, skip_features):
        # skip_features: [256@16³, 128@32³, 64@64³]
        x = self.decoder_0(x, skip_features[1])  # 256@16³ -> 128@32³
        x = self.decoder_1(x, skip_features[2])  # 128@32³ -> 64@64³
        return x


# ============================================================
# 6. 完整的 X2Shape 模型 (3层简化版)
# ============================================================

class X2Shape(nn.Module):
    """
    X2Shape完整模型 - 3层简化版

    架构:
    1. AP和LAT分别经过3层2D编码和VBP反投影
    2. 两个VBP输出的体积相加并归一化 -> 融合体积 (64@64³)
    3. 融合体积通过3层U-Net编码器产生多尺度特征
    4. 2层U-Net解码器使用反卷积上采样 + 特征拼接
    5. 解码器输出与VBP跳跃连接融合后输出

    VBP路径:
        - 主路径: VBP融合体积 → 3D编码器 → U-Net解码器 → 解码器输出
        - 跳跃连接路径: VBP融合体积 → 直接作为跳跃连接 → 与解码器输出融合
    """
    def __init__(self, img_size=256, in_chans=1, num_classes=1,
                 dims_2d=[32, 64, 128], depths_2d=[1, 1, 2],
                 encoder_channels=[64, 128, 256],
                 vbp_output_channels=64,
                 vbp_embed_dim=64):
        super().__init__()
        self.img_size = img_size

        # 体积尺寸: (X, Y, Z) = (64, 64, 64)
        vol_sizes = [img_size // 4, img_size // 4, img_size // 4]

        # ========== 1. 两个独立的2D编码器 (3层) ==========
        use_attn_layers = [False, True, True]  # 后两层使用自注意力
        self.encoder_2d_ap = MambaVisionEncoder2D(
            in_chans, dims_2d, depths_2d,
            use_attention_layers=use_attn_layers
        )
        self.encoder_2d_lat = MambaVisionEncoder2D(
            in_chans, dims_2d, depths_2d,
            use_attention_layers=use_attn_layers
        )

        # ========== 2. VBP模块 ==========
        self.vbp_ap = VolumetricBackprojection(
            dims_2d, vbp_output_channels, vol_sizes,
            embed_dim=vbp_embed_dim
        )
        self.vbp_lat = VolumetricBackprojection(
            dims_2d, vbp_output_channels, vol_sizes,
            embed_dim=vbp_embed_dim
        )

        # ========== 3. 双视图融合 ==========
        self.fusion_norm = InstanceNorm3d(vbp_output_channels)

        # ========== 4. 3层U-Net 3D编码器 ==========
        self.encoder_3d = Encoder3D(vbp_output_channels, encoder_channels)

        # ========== 5. 2层U-Net解码器 ==========
        self.decoder = UNetDecoder()

        # ========== 6. 最终精炼块 ==========
        # decoder输出: 64@64³, VBP跳跃连接: 64@64³
        self.final_refine = nn.Sequential(
            nn.Conv3d(64 + 64, 64, kernel_size=3, padding=1),
            InstanceNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            InstanceNorm3d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # ========== 7. 分割头 ==========
        self.seg_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            InstanceNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )

        # ========== 8. 上采样到256 (64 -> 128 -> 256) ==========
        self.upsample_to_256 = nn.Sequential(
            nn.ConvTranspose3d(num_classes, num_classes, kernel_size=2, stride=2),  # 64->128
            nn.ConvTranspose3d(num_classes, num_classes, kernel_size=2, stride=2),  # 128->256
        )

    def forward(self, x_ap, x_lat, angle=None):
        """
        前向传播

        Args:
            x_ap: AP视图 (B, C, H, W) - H=W=256
            x_lat: LAT视图 (B, C, H, W) - H=W=256
            angle: 投影角度 (度)，默认90°

        Returns:
            output: 3D分割体积 (B, num_classes, 256, 256, 256)
        """
        if angle is None:
            angle = 90.0

        # ===== Step 1: 2D编码 (3层) =====
        f_ap_2d = self.encoder_2d_ap(x_ap)  # [32@128², 64@64², 128@32²]
        f_lat_2d = self.encoder_2d_lat(x_lat)

        # ===== Step 2: VBP反投影 =====
        f_ap_3d = self.vbp_ap(f_ap_2d, angle, proj_type='ap')
        f_lat_3d = self.vbp_lat(f_lat_2d, angle, proj_type='lat')

        # ===== Step 3: 双视图融合 =====
        fused_3d = f_ap_3d + f_lat_3d
        fused_3d = self.fusion_norm(fused_3d)

        # 保存VBP跳跃连接
        vbp_jump = fused_3d  # (B, 64, 64, 64, 64)

        # ===== Step 4: 3层U-Net编码器 =====
        # skip_features: [256@16³, 128@32³, 64@64³] (从深到浅)
        skip_features = self.encoder_3d(fused_3d)

        # ===== Step 5: 2层U-Net解码器 =====
        # 最深层的特征作为解码器输入
        deepest_feat = skip_features[0]  # (B, 256, 16, 16, 16)
        decoded_feat = self.decoder(deepest_feat, skip_features)  # (B, 64, 64, 64, 64)

        # ===== Step 6: 与VBP跳跃连接融合 =====
        # 拼接: 解码器输出(64) + VBP跳跃连接(64) -> 128
        combined = torch.cat([decoded_feat, vbp_jump], dim=1)  # (B, 128, 64, 64, 64)

        # 精炼: 128 -> 64
        final_feat = self.final_refine(combined)  # (B, 64, 64, 64, 64)

        # ===== Step 7: 分割头 =====
        seg_out = self.seg_head(final_feat)  # (B, 1, 64, 64, 64)

        # ===== Step 8: 上采样到256 =====
        output = self.upsample_to_256(seg_out)  # (B, 1, 256, 256, 256)

        return output


# ============================================================
# 7. 辅助函数
# ============================================================

def load_nifti_as_tensor(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)

    if data.ndim == 2:
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    elif data.ndim == 3:
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}")

    return tensor


def save_tensor_as_nifti(tensor, file_path, ref_nii_path=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if tensor.dim() == 5:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif tensor.dim() == 4:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()
    else:
        data = tensor.cpu().numpy()

    if data.ndim == 2:
        data = data[None, ...]

    if ref_nii_path and os.path.exists(ref_nii_path):
        ref_nii = nib.load(ref_nii_path)
        affine = ref_nii.affine
    else:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)
    print(f"  已保存: {file_path}")


# ============================================================
# 8. 主程序
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # 🔧 在这里修改您的路径配置
    # ============================================================

    DATA_PATH = "/mnt/d/med_data/biron/data1/train_any/00_30"
    OUTPUT_PATH = "/mnt/d/med_data/biron/data1/VBP"
    ANGLE = 30.0
    USE_GPU = True

    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

    print("=" * 70)
    print("X2Shape 3D器官重建推理 (3层简化版)")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"数据路径: {DATA_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"投影角度: {ANGLE}°")
    print("=" * 70)

    ap_path = os.path.join(DATA_PATH, "ap.nii.gz")
    lat_path = os.path.join(DATA_PATH, "lat.nii.gz")
    mask_path = os.path.join(DATA_PATH, "mask.nii.gz")

    if not os.path.exists(ap_path):
        print(f"\n❌ 错误: AP文件不存在: {ap_path}")
        exit(1)

    if not os.path.exists(lat_path):
        print(f"\n❌ 错误: LAT文件不存在: {lat_path}")
        exit(1)

    print("\n创建模型...")
    model = X2Shape(
        img_size=256,
        in_chans=1,
        num_classes=1,
        dims_2d=[32, 64, 128],
        depths_2d=[1, 1, 2],
        encoder_channels=[64, 128, 256],
        vbp_output_channels=64,
        vbp_embed_dim=64
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")

    print("\n加载数据...")
    ap_tensor = load_nifti_as_tensor(ap_path).to(device)
    lat_tensor = load_nifti_as_tensor(lat_path).to(device)

    print(f"AP形状: {ap_tensor.shape}")
    print(f"LAT形状: {lat_tensor.shape}")

    if ap_tensor.shape[-2:] != (256, 256):
        ap_tensor = F.interpolate(ap_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    if lat_tensor.shape[-2:] != (256, 256):
        lat_tensor = F.interpolate(lat_tensor, size=(256, 256), mode='bilinear', align_corners=False)

    print("\n模型推理中...")
    model.eval()
    start_time = datetime.now()

    with torch.no_grad():
        output = model(ap_tensor, lat_tensor, angle=ANGLE)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"推理完成，耗时: {elapsed:.2f} 秒")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")

    case_name = os.path.basename(DATA_PATH)
    output_dir = os.path.join(OUTPUT_PATH, case_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n保存结果...")
    output_file = os.path.join(output_dir, f"output_{ANGLE}deg.nii.gz")
    save_tensor_as_nifti(output, output_file, mask_path if os.path.exists(mask_path) else None)

    if os.path.exists(mask_path):
        print("  加载Mask...")
        mask_tensor = load_nifti_as_tensor(mask_path).to(device)

        if mask_tensor.shape[-3:] != output.shape[-3:]:
            mask_resized = F.interpolate(mask_tensor, size=output.shape[-3:],
                                         mode='trilinear', align_corners=False)
        else:
            mask_resized = mask_tensor

        combined = output + mask_resized
        combined_file = os.path.join(output_dir, f"combined_{ANGLE}deg.nii.gz")
        save_tensor_as_nifti(combined, combined_file, mask_path)

    print("\n" + "=" * 70)
    print(f"✅ 推理完成！结果已保存到: {output_dir}")
    print("=" * 70)