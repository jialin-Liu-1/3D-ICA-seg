"""
X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays
适配版 - 支持通道减半和160尺寸
输入: 160x160, 输出: 160x160x160 单通道
支持自定义角度反投影
维度约定: (B, C, X, Y, Z) 对应 (batch, channel, axis0, axis1, axis2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import nibabel as nib
import os

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
    def __init__(self, dim, mlp_ratio=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),  # 添加Dropout
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)   # 添加Dropout
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.norm1(x_flat)
        x_mixer = rearrange(self.mixer(x), 'b c h w -> b (h w) c')
        x_flat = x_flat + x_mixer
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)


class MambaVisionEncoder2D(nn.Module):
    def __init__(self, in_chans=1, dims=[32, 64, 128, 256], depths=[1, 1, 2, 1], dropout=0.2):
        super().__init__()
        self.dims = dims
        self.num_scales = len(dims)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        current_dim = dims[0]

        for i in range(self.num_scales):
            stage_blocks = nn.Sequential()
            for _ in range(depths[i]):
                stage_blocks.append(MambaVisionBlock(current_dim, dropout=dropout))
            self.stages.append(stage_blocks)

            if i < self.num_scales - 1:
                self.stages.append(
                    nn.Sequential(
                        nn.Conv2d(current_dim, dims[i + 1], kernel_size=2, stride=2),
                        LayerNorm2d(dims[i + 1]),
                        nn.GELU()
                    )
                )
                current_dim = dims[i + 1]

    def forward(self, x):
        x = self.stem(x)
        features = []
        for layer in self.stages:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                if not isinstance(layer[0], nn.Conv2d):
                    features.append(x)
        return features


# ============================================================
# 2. 体积反投影模块 (VBP)
# ============================================================
class VolumetricBackprojection(nn.Module):
    def __init__(self, multi_scale_channels, out_channels, volume_shape=(40, 40, 40), dropout=0.2):
        super().__init__()
        self.multi_scale_channels = multi_scale_channels
        self.out_channels = out_channels
        # volume_shape = (X, Y, Z)
        self.volume_shape = volume_shape
        self.num_scales = len(multi_scale_channels)

        self.angle = None
        self.projection_type = None

        self.fusion_convs = nn.ModuleList()
        for i, ch in enumerate(multi_scale_channels):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),  # 添加2D Dropout
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout)   # 添加2D Dropout
                )
            )

        self.project_3d = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  # 添加3D Dropout
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)   # 添加3D Dropout
        )

        self.position_weights = nn.Parameter(torch.ones(1, 1, volume_shape[0], volume_shape[1], volume_shape[2]))

    def backproject_single_angle(self, f_2d, angle_deg, proj_type='ap'):
        """
        f_2d: (C, H, W) = (C, Z, X) 或 (C, X, Z)
        proj_type: 'ap' 或 'lat'
        返回: (C, X, Y, Z)
        """
        C, H, W = f_2d.shape
        X_dim, Y_dim, Z_dim = self.volume_shape  # (X, Y, Z)

        # 将 f_2d 解释为 (C, X, Z) 而不是 (C, Z, X)
        # 即 H = X, W = Z
        if H == Z_dim and W == X_dim:
            # 已经是 (C, X, Z) 格式
            f_2d_xz = f_2d
        elif H == X_dim and W == Z_dim:
            # 是 (C, Z, X) 格式，需要转置
            f_2d_xz = f_2d.permute(0, 2, 1)  # (C, Z, X) -> (C, X, Z)
        else:
            # 假设是 (C, X, Z)
            f_2d_xz = f_2d

        if proj_type == 'ap':
            # AP反投影：沿Y轴均匀复制
            # (C, X, Z) -> (C, X, Y, Z)
            f_3d = f_2d_xz[:, :, None, :].expand(-1, -1, Y_dim, -1)  # (C, X, Y, Z)

        else:  # 'lat'
            if angle_deg == 90:
                f_3d = f_2d_xz[:, :, None, :].expand(-1, -1, Y_dim, -1)
            else:
                # LAT反投影：绕X轴旋转 -angle_deg 后，沿Y轴展开
                # 每个Y层在Z方向偏移
                angle_rad = torch.tensor(angle_deg * np.pi / 180.0, device=f_2d.device)
                tan_angle = torch.tan(angle_rad)

                f_3d = torch.zeros(C, X_dim, Y_dim, Z_dim, device=f_2d.device)
                y_center = (Y_dim - 1) / 2

                for y in range(Y_dim):
                    # 计算Z偏移
                    offset_z = int(-(y - y_center) * tan_angle)
                    offset_z = max(min(offset_z, Z_dim - 1), -(Z_dim - 1))

                    if offset_z != 0:
                        # 在Z方向（f_2d_xz的dim=2）平移
                        shifted = torch.roll(f_2d_xz, shifts=offset_z, dims=2)
                        if offset_z > 0:
                            shifted[:, :, :offset_z] = 0
                        else:
                            shifted[:, :, offset_z:] = 0
                    else:
                        shifted = f_2d_xz

                    f_3d[:, :, y, :] = shifted

                # 高斯权重平滑
                import math
                y_weights = torch.zeros(Y_dim, device=f_2d.device)
                for y in range(Y_dim):
                    y_norm = (y - y_center) / (Y_dim / 2)
                    y_weights[y] = math.exp(-y_norm ** 2 * 2)
                y_weights = y_weights.view(1, 1, Y_dim, 1)
                f_3d = f_3d * y_weights

        f_3d = torch.clamp(f_3d, min=1e-8, max=1e6)

        return f_3d  # (C, X, Y, Z)

    def forward(self, f_multi_scale, angle=None, proj_type='ap'):
        B = f_multi_scale[0].shape[0]
        X_dim, Y_dim, Z_dim = self.volume_shape

        self.projection_type = proj_type

        # 处理角度
        if angle is not None:
            if isinstance(angle, torch.Tensor):
                if angle.numel() == 1:
                    self.angle = angle.item()
                    use_per_sample = False
                elif angle.numel() == B:
                    self.angle = angle.cpu().tolist()
                    use_per_sample = True
                else:
                    raise ValueError(f"Angle tensor shape {angle.shape} not compatible with batch size {B}")
            else:
                self.angle = angle
                use_per_sample = False
        elif self.angle is None:
            self.angle = 90
            use_per_sample = False
        else:
            use_per_sample = isinstance(self.angle, list)

        # 融合多尺度特征
        fused_2d = []
        for i, f in enumerate(f_multi_scale):
            # f 的形状是 (B, C, H, W)，其中 H=40, W=40
            # 我们需要将其解释为 (B, C, X, Z) 即 H=X, W=Z
            if f.shape[-2:] != (X_dim, Z_dim):
                f_up = F.interpolate(f, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)
            else:
                f_up = f
            f_proj = self.fusion_convs[i](f_up)
            fused_2d.append(f_proj)

        fused_2d = sum(fused_2d)  # (B, C, X, Z)

        # 反投影
        if use_per_sample:
            f_3d_list = []
            for b in range(B):
                f_single = fused_2d[b]
                angle_deg = self.angle[b] if isinstance(self.angle, list) else self.angle
                f_3d_single = self.backproject_single_angle(f_single, angle_deg, self.projection_type)
                f_3d_list.append(f_3d_single)
            f_3d = torch.stack(f_3d_list, dim=0)  # (B, C, X, Y, Z)
        else:
            angle_deg = self.angle
            if isinstance(angle_deg, torch.Tensor):
                angle_deg = angle_deg.item()

            if self.projection_type == 'ap' or angle_deg == 90:
                # (B, C, X, Z) -> (B, C, X, Y, Z)
                f_3d = fused_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)  # (B, C, X, Y, Z)
            else:
                # LAT反投影
                angle_rad = torch.tensor(angle_deg * np.pi / 180.0, device=fused_2d.device)
                tan_angle = torch.tan(angle_rad)

                f_3d = torch.zeros(B, self.out_channels, X_dim, Y_dim, Z_dim, device=fused_2d.device)
                y_center = (Y_dim - 1) / 2

                for y in range(Y_dim):
                    offset_z = int(-(y - y_center) * tan_angle)
                    offset_z = max(min(offset_z, Z_dim - 1), -(Z_dim - 1))

                    if offset_z != 0:
                        # 在Z方向（fused_2d的dim=3）平移
                        shifted = torch.roll(fused_2d, shifts=offset_z, dims=3)
                        if offset_z > 0:
                            shifted[:, :, :, :offset_z] = 0
                        else:
                            shifted[:, :, :, offset_z:] = 0
                    else:
                        shifted = fused_2d

                    f_3d[:, :, :, y, :] = shifted  # (B, C, X, Z) -> (B, C, X, Y, Z)

                # 高斯权重平滑
                import math
                y_weights = torch.zeros(Y_dim, device=fused_2d.device)
                for y in range(Y_dim):
                    y_norm = (y - y_center) / (Y_dim / 2)
                    y_weights[y] = math.exp(-y_norm ** 2 * 2)
                y_weights = y_weights.view(1, 1, 1, Y_dim, 1)
                f_3d = f_3d * y_weights

        # 3D卷积处理
        out = self.project_3d(f_3d)

        # 位置权重
        pos_weight = self.position_weights.expand(B, -1, -1, -1, -1)
        out = out * torch.sigmoid(pos_weight)
        out = torch.clamp(out, min=-10, max=10)

        return out


# ============================================================
# 3. 3D Encoder
# ============================================================

class Encoder3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  # 添加3D Dropout
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)   # 添加3D Dropout
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
    def __init__(self, in_chans, channels=[32, 64, 128, 256], dropout=0.2):
        super().__init__()
        self.channels = channels

        self.blocks = nn.ModuleList()
        curr_ch = in_chans

        for i, out_ch in enumerate(channels):
            downsample = (i < len(channels) - 1)
            self.blocks.append(Encoder3DBlock(curr_ch, out_ch, downsample, dropout=dropout))
            curr_ch = out_ch

    def forward(self, x):
        features = []
        for block in self.blocks:
            x, feat = block(x)
            features.append(feat)
        return features[::-1]


# ============================================================
# 4. Cross-Mamba 模块
# ============================================================

class CrossMamba(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dim = dim

        self.norm_ap = nn.LayerNorm(dim)
        self.norm_lat = nn.LayerNorm(dim)
        self.mamba_ap = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_lat = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.gate_ap = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout)  # 添加Dropout
        )
        self.gate_lat = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout)  # 添加Dropout
        )

    def forward(self, f_ap, f_lat):
        # 输入: (B, C, X, Y, Z)
        B, C, X_dim, Y_dim, Z_dim = f_ap.shape

        f_ap_flat = rearrange(f_ap, 'b c x y z -> b (x y z) c')
        f_lat_flat = rearrange(f_lat, 'b c x y z -> b (x y z) c')

        y1 = self.mamba_ap(self.norm_ap(f_ap_flat))
        g1 = self.gate_ap(self.norm_ap(f_lat_flat))
        out1 = rearrange(y1 * g1, 'b (x y z) c -> b c x y z', x=X_dim, y=Y_dim, z=Z_dim)

        y2 = self.mamba_lat(self.norm_lat(f_lat_flat))
        g2 = self.gate_lat(self.norm_lat(f_ap_flat))
        out2 = rearrange(y2 * g2, 'b (x y z) c -> b c x y z', x=X_dim, y=Y_dim, z=Z_dim)

        return out1 + out2


# ============================================================
# 5. 残差块 (U-Net 升采样路径)
# ============================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  # 添加3D Dropout
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.Dropout3d(dropout)   # 添加3D Dropout
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
# 6. 分割头
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  # 添加3D Dropout
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),  # 添加3D Dropout
            nn.Conv3d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 7. 完整的 X2Shape 模型
# ============================================================

class X2Shape(nn.Module):
    def __init__(self, img_size=160, in_chans=1, num_classes=1,
                 dims_2d=[32, 64, 128, 256], depths_2d=[1, 1, 2, 1],
                 dims_3d=[32, 64, 128, 256], vbp_output_channels=64,
                 dropout=0.2):  # 添加dropout参数
        super().__init__()
        self.img_size = img_size
        self.num_levels = len(dims_3d)

        # 体积尺寸: (X, Y, Z) = (40, 40, 40)
        vol_sizes = [img_size // 4, img_size // 4, img_size // 4]  # (X, Y, Z)

        # ========== 1. 两个独立的 2D 编码器 ==========
        self.encoder_2d_ap = MambaVisionEncoder2D(in_chans, dims_2d, depths_2d, dropout=dropout)
        self.encoder_2d_lat = MambaVisionEncoder2D(in_chans, dims_2d, depths_2d, dropout=dropout)

        # ========== 2. VBP 模块 ==========
        self.vbp_ap = VolumetricBackprojection(dims_2d, vbp_output_channels, vol_sizes, dropout=dropout)
        self.vbp_lat = VolumetricBackprojection(dims_2d, vbp_output_channels, vol_sizes, dropout=dropout)

        # ========== 3. 两个独立的 3D Encoder ==========
        self.encoder_3d_ap = Encoder3D(vbp_output_channels, dims_3d, dropout=dropout)
        self.encoder_3d_lat = Encoder3D(vbp_output_channels, dims_3d, dropout=dropout)

        # ========== 4. Cross-Mamba 模块 ==========
        self.cross_mambas = nn.ModuleList([
            CrossMamba(dims_3d[3], dropout=dropout),
            CrossMamba(dims_3d[2], dropout=dropout),
            CrossMamba(dims_3d[1], dropout=dropout),
            CrossMamba(dims_3d[0], dropout=dropout)
        ])

        # ========== 5. 上采样层 ==========
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose3d(dims_3d[3], dims_3d[2], kernel_size=2, stride=2),
            nn.ConvTranspose3d(dims_3d[2], dims_3d[1], kernel_size=2, stride=2),
            nn.ConvTranspose3d(dims_3d[1], dims_3d[0], kernel_size=2, stride=2)
        ])

        # ========== 6. 残差块 ==========
        self.res_blocks = nn.ModuleList([
            ResidualBlock3D(dims_3d[2], dropout=dropout),
            ResidualBlock3D(dims_3d[1], dropout=dropout),
            ResidualBlock3D(dims_3d[0], dropout=dropout)
        ])

        # ========== 7. 最终残差块 ==========
        self.final_res_block = ResidualBlock3D(dims_3d[0], dropout=dropout)

        # ========== 8. 分割头 ==========
        self.seg_head = SegmentationHead(dims_3d[0], num_classes, dropout=dropout)

        # ========== 9. VBP跳跃连接通道调整 ==========
        self.vbp_jump_adjust = nn.Conv3d(vbp_output_channels, dims_3d[0], kernel_size=1)

    def forward(self, x_ap, x_lat, angle=None):
        """
        前向传播

        输入: x_ap, x_lat 形状为 (B, C, X, Y) 其中 X=256, Y=256
        输出: (B, C, X, Y, Z) 其中 X=256, Y=256, Z=256
        """
        # Step 1: 2D 编码
        f_ap_2d = self.encoder_2d_ap(x_ap)
        f_lat_2d = self.encoder_2d_lat(x_lat)

        # Step 2: VBP
        f_ap_3d_base = self.vbp_ap(f_ap_2d, angle, proj_type='ap')
        f_lat_3d_base = self.vbp_lat(f_lat_2d, angle, proj_type='lat')

        vbp_jump_ap = f_ap_3d_base
        vbp_jump_lat = f_lat_3d_base

        # Step 3: 3D Encoder
        f_ap_3d_multi = self.encoder_3d_ap(f_ap_3d_base)
        f_lat_3d_multi = self.encoder_3d_lat(f_lat_3d_base)

        # Step 4: Cross-Mamba
        cross_outputs = []
        for i in range(self.num_levels):
            cross_feat = self.cross_mambas[i](f_ap_3d_multi[i], f_lat_3d_multi[i])
            cross_outputs.append(cross_feat)

        # Step 5: U-Net升采样
        current_feat = cross_outputs[0]

        for i in range(1, self.num_levels):
            upsampled = self.upsample_layers[i-1](current_feat)
            current_cross = cross_outputs[i]

            if upsampled.shape[-3:] != current_cross.shape[-3:]:
                upsampled = F.interpolate(upsampled, size=current_cross.shape[-3:],
                                          mode='trilinear', align_corners=False)

            current_feat = self.res_blocks[i-1](current_cross, upsampled)

        # Step 6: 最终残差块
        vbp_jump = vbp_jump_ap + vbp_jump_lat

        if vbp_jump.shape[1] != current_feat.shape[1]:
            vbp_jump = self.vbp_jump_adjust(vbp_jump)

        if current_feat.shape[-3:] != vbp_jump.shape[-3:]:
            current_feat = F.interpolate(current_feat, size=vbp_jump.shape[-3:],
                                         mode='trilinear', align_corners=False)

        final_feat = self.final_res_block(current_feat, vbp_jump)

        # Step 7: 分割头
        output = self.seg_head(final_feat)

        # Step 8: 上采样到目标尺寸
        if output.shape[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size, self.img_size),
                                   mode='trilinear', align_corners=False)

        return output  # (B, C, X, Y, Z)


# ============================================================
# 测试代码
# ============================================================

def load_nifti_as_tensor(file_path):
    """加载NIFTI文件并转换为PyTorch张量"""
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    # 添加batch和channel维度
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def save_tensor_as_nifti(tensor, file_path, ref_nii_path=None):
    """将PyTorch张量保存为NIFTI文件"""
    # tensor: (B, C, X, Y, Z)
    if tensor.dim() == 5:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()  # (X, Y, Z)
    else:
        data = tensor.cpu().numpy()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if ref_nii_path and os.path.exists(ref_nii_path):
        ref_nii = nib.load(ref_nii_path)
        affine = ref_nii.affine
    else:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)
    print(f"  已保存: {file_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("X2Shape 模型测试 - 支持角度控制的反投影")
    print("=" * 60)

    model = X2Shape(
        img_size=256,
        in_chans=1,
        num_classes=1,
        dims_2d=[32, 64, 128, 256],
        depths_2d=[1, 1, 2, 1],
        dims_3d=[32, 64, 128, 256],
        vbp_output_channels=64,
        dropout=0.2  # 设置dropout
    ).to(device)

    model.eval()

    test_cases = [(0, 90), (0, 30), (0, 60), (0, 70)]
    base_path = r"/mnt/d/med_data/biron/data1/train_any"
    output_base = r"/mnt/d/med_data/biron/data1/VBP"

    print(f"\n测试数据路径: {base_path}")
    print(f"输出路径: {output_base}")
    print(f"设备: {device}")

    with torch.no_grad():
        for case_num, angle in test_cases:
            print(f"\n{'=' * 50}")
            print(f"处理病例 {case_num}，角度 {angle}°")
            print(f"{'=' * 50}")

            case_folder = f"{case_num}_{angle}"
            ap_path = os.path.join(base_path, case_folder, "ap.nii.gz")
            lat_path = os.path.join(base_path, case_folder, "lat.nii.gz")
            mask_path = os.path.join(base_path, case_folder, "mask.nii.gz")

            if not all([os.path.exists(p) for p in [ap_path, lat_path, mask_path]]):
                print(f"  跳过: 文件不存在")
                continue

            try:
                print(f"  加载AP: {ap_path}")
                ap_tensor = load_nifti_as_tensor(ap_path).to(device)

                print(f"  加载LAT: {lat_path}")
                lat_tensor = load_nifti_as_tensor(lat_path).to(device)

                print(f"  加载Mask: {mask_path}")
                mask_tensor = load_nifti_as_tensor(mask_path).to(device)

                print(f"  AP形状: {ap_tensor.shape}")
                print(f"  LAT形状: {lat_tensor.shape}")
                print(f"  Mask形状: {mask_tensor.shape}")

                # 模型前向传播
                output = model(ap_tensor, lat_tensor, angle=angle)

                print(f"  输出形状: {output.shape}")
                print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

                output_folder = os.path.join(output_base, case_folder)
                output_path = os.path.join(output_folder, "vbp_output.nii.gz")

                # 确保mask和output维度一致 (都是 X, Y, Z)
                if mask_tensor.shape[-3:] != output.shape[-3:]:
                    mask_resized = F.interpolate(mask_tensor, size=output.shape[-3:],
                                                 mode='trilinear', align_corners=False)
                else:
                    mask_resized = mask_tensor

                # 相加
                combined = output + mask_resized
                combined_path = os.path.join(output_folder, "combined_output.nii.gz")

                # 保存 (不需要额外转置，维度已经是 X, Y, Z)
                save_tensor_as_nifti(output, output_path, mask_path)
                save_tensor_as_nifti(combined, combined_path, mask_path)

                print(f"  ✓ 成功保存到: {output_folder}")

            except Exception as e:
                print(f"  ✗ 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)