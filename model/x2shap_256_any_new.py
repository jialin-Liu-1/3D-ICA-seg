"""
X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays
测试程序 - 包含完整模型测试和纯反投影测试
使用真旋转方法，VBP体积固定为64³
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import nibabel as nib
import os
import math
import time
import gc

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ 使用官方 mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    raise ImportError("请先安装 mamba_ssm")


# ============================================================
# 1. 简化的2D编码器 (两层卷积)
# ============================================================

class Simple2DEncoder(nn.Module):
    """简化的2D编码器：两层卷积，输出单尺度特征"""
    def __init__(self, in_chans=1, out_channels=32, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return [x]


# ============================================================
# 2. 体积反投影模块 (VBP) - 使用真旋转方法
# ============================================================
# ============================================================
# 2. 体积反投影模块 (VBP) - 使用代码1的反投影方法
# ============================================================

class VolumetricBackprojection(nn.Module):
    def __init__(self, multi_scale_channels, out_channels, volume_shape=(64, 64, 64),
                 proj_type='ap', dropout=0.0, use_raw_output=False):
        super().__init__()
        self.multi_scale_channels = multi_scale_channels
        self.out_channels = out_channels
        self.volume_shape = volume_shape
        self.proj_type = proj_type
        self.num_scales = len(multi_scale_channels)
        self.use_raw_output = use_raw_output
        self.angle = None

        # 融合卷积
        self.fusion_convs = nn.ModuleList()
        for i, ch in enumerate(multi_scale_channels):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout)
                )
            )

        # 3D精炼卷积
        self.project_3d = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout)
        )

        # 位置编码：使用1x1x1卷积
        self.position_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True)

    def backproject_single_angle(self, f_2d, angle_deg, reverse_y=False):
        """
        与代码1完全一致的反投影方法
        使用PyTorch实现，获得GPU加速

        代码1的旋转: scipy.ndimage.rotate(axes=(1,2))
        -> 绕X轴旋转 (在 X,Y,Z 坐标下)
        -> X' = X, Y' = Y*cos - Z*sin, Z' = Y*sin + Z*cos

        PyTorch grid_sample 使用 (Z, Y, X) 坐标:
        -> 需要转换: Z' = Z*cos - Y*sin, Y' = Z*sin + Y*cos, X' = X
        """
        if f_2d.dim() == 4:
            B, C, X, Z = f_2d.shape
            has_batch = True
        else:
            C, X, Z = f_2d.shape
            has_batch = False
            B = 1
            f_2d = f_2d.unsqueeze(0)

        X_dim, Y_dim, Z_dim = self.volume_shape
        assert X == X_dim and Z == Z_dim

        # ============ 第一步：沿Y轴复制（与代码1完全一致） ============
        f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)
        if reverse_y:
            f_3d = torch.flip(f_3d, dims=[3])

        # ============ 第二步：绕X轴旋转（与代码1 axes=(1,2) 完全一致） ============
        if self.proj_type == 'lat' and angle_deg != 90:
            rotation_angle = -angle_deg

            B_total = B * C
            f_flat = f_3d.view(B_total, 1, X_dim, Y_dim, Z_dim)

            angle_rad = torch.tensor(rotation_angle * np.pi / 180.0, device=f_2d.device)
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)

            # 在grid坐标 (x, y, z) = (Z, Y, X) 下绕X轴旋转
            # 对应代码1的 axes=(1,2) 即绕X轴旋转
            theta = torch.tensor([
                [cos_a, -sin_a, 0.0, 0.0],  # Z' = Z*cos - Y*sin
                [sin_a, cos_a, 0.0, 0.0],  # Y' = Z*sin + Y*cos
                [0.0, 0.0, 1.0, 0.0]  # X' = X
            ], device=f_2d.device).float()

            theta_batch = theta.unsqueeze(0).expand(B_total, -1, -1)

            grid = F.affine_grid(
                theta_batch,
                size=(B_total, 1, X_dim, Y_dim, Z_dim),
                align_corners=False
            )

            f_rotated = F.grid_sample(
                f_flat,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            del grid, theta_batch, f_flat
            f_3d = f_rotated.view(B, C, X_dim, Y_dim, Z_dim)
            del f_rotated

        if not has_batch:
            f_3d = f_3d.squeeze(0)

        f_3d = torch.clamp(f_3d, min=1e-8, max=1e6)
        return f_3d

    def forward(self, f_multi_scale, angle=None, proj_type=None):
        B = f_multi_scale[0].shape[0]
        X_dim, Y_dim, Z_dim = self.volume_shape

        if proj_type is None:
            proj_type = self.proj_type

        if angle is not None:
            if isinstance(angle, torch.Tensor):
                if angle.numel() == 1:
                    self.angle = angle.item()
                    use_per_sample = False
                else:
                    self.angle = angle.cpu().tolist()
                    use_per_sample = True
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
            if f.shape[-2:] != (X_dim, Z_dim):
                f_up = F.interpolate(f, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)
            else:
                f_up = f
            f_proj = self.fusion_convs[i](f_up)
            fused_2d.append(f_proj)

        fused_2d = sum(fused_2d)

        # 反投影
        if use_per_sample:
            f_3d_list = []
            for b in range(B):
                f_single = fused_2d[b:b + 1]
                angle_deg = self.angle[b] if isinstance(self.angle, list) else self.angle
                reverse_y = (proj_type == 'lat')
                f_3d_single = self.backproject_single_angle(f_single, angle_deg, reverse_y)
                f_3d_list.append(f_3d_single)
            f_3d = torch.cat(f_3d_list, dim=0)
        else:
            angle_deg = self.angle
            if isinstance(angle_deg, torch.Tensor):
                angle_deg = angle_deg.item()
            reverse_y = (proj_type == 'lat')
            f_3d = self.backproject_single_angle(fused_2d, angle_deg, reverse_y)

        raw_backprojection = None
        if self.use_raw_output:
            raw_backprojection = f_3d.clone().detach()

        out = self.project_3d(f_3d)
        out = self.position_conv(out)
        out = torch.clamp(out, min=-10, max=10)

        if self.use_raw_output:
            return out, raw_backprojection
        return out


# ============================================================
# 3. 纯反投影模块 (与VBP使用完全相同的反投影方法)
# ============================================================

class PureBackprojection(nn.Module):
    """
    纯反投影模块：与VBP使用完全相同的反投影方法
    与代码1完全一致：绕X轴旋转 (axes=(1,2))
    """

    def __init__(self, volume_shape=(256, 256, 256)):
        super().__init__()
        self.volume_shape = volume_shape

    def backproject_single_angle(self, f_2d, angle_deg, proj_type='ap', reverse_y=False):
        """
        与VBP.backproject_single_angle 完全一致
        与代码1完全一致：绕X轴旋转
        """
        if f_2d.dim() == 4:
            B, C, X, Z = f_2d.shape
            has_batch = True
        else:
            C, X, Z = f_2d.shape
            has_batch = False
            B = 1
            f_2d = f_2d.unsqueeze(0)

        X_dim, Y_dim, Z_dim = self.volume_shape
        assert X == X_dim and Z == Z_dim

        # ============ 第一步：沿Y轴复制 ============
        f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)
        if reverse_y:
            f_3d = torch.flip(f_3d, dims=[3])

        # ============ 第二步：绕X轴旋转（与代码1 axes=(1,2) 完全一致） ============
        if proj_type == 'lat' and angle_deg != 90:
            rotation_angle = -angle_deg

            B_total = B * C
            f_flat = f_3d.view(B_total, 1, X_dim, Y_dim, Z_dim)

            angle_rad = torch.tensor(rotation_angle * np.pi / 180.0, device=f_2d.device)
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)

            # 绕X轴旋转 (对应代码1的 axes=(1,2))
            theta = torch.tensor([
                [cos_a, -sin_a, 0.0, 0.0],
                [sin_a, cos_a, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], device=f_2d.device).float()

            theta_batch = theta.unsqueeze(0).expand(B_total, -1, -1)

            grid = F.affine_grid(
                theta_batch,
                size=(B_total, 1, X_dim, Y_dim, Z_dim),
                align_corners=False
            )

            f_rotated = F.grid_sample(
                f_flat,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            del grid, theta_batch, f_flat
            f_3d = f_rotated.view(B, C, X_dim, Y_dim, Z_dim)
            del f_rotated

        if not has_batch:
            f_3d = f_3d.squeeze(0)

        f_3d = torch.clamp(f_3d, min=0, max=1e6)
        return f_3d

    def forward(self, x_ap, x_lat, angle):
        # AP反投影：reverse_y=False
        ap_volume = self.backproject_single_angle(x_ap, angle, proj_type='ap', reverse_y=False)
        # LAT反投影：reverse_y=True (与代码1一致)
        lat_volume = self.backproject_single_angle(x_lat, angle, proj_type='lat', reverse_y=True)
        return ap_volume, lat_volume


# ============================================================
# 4. 3D Encoder
# ============================================================

class Encoder3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout)
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
    def __init__(self, in_chans, channels=[32, 64, 128, 256], dropout=0.0):
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
# 5. Cross-Mamba 模块
# ============================================================

class CrossMamba(nn.Module):
    def __init__(self, dim, dropout=0.0, use_residual=True):
        super().__init__()
        self.dim = dim
        self.use_residual = use_residual

        self.norm_ap = nn.LayerNorm(dim)
        self.norm_lat = nn.LayerNorm(dim)
        self.mamba_ap = Mamba(d_model=dim, d_state=8, d_conv=4, expand=1)
        self.mamba_lat = Mamba(d_model=dim, d_state=8, d_conv=4, expand=1)
        self.gate_ap = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout))
        self.gate_lat = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout))

        if use_residual:
            self.residual_block = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout)
            )
            self.norm_res = nn.LayerNorm(dim)

    def forward(self, f_ap, f_lat):
        B, C, X_dim, Y_dim, Z_dim = f_ap.shape

        f_ap_flat = f_ap.flatten(2).transpose(1, 2)
        f_lat_flat = f_lat.flatten(2).transpose(1, 2)

        y1 = self.mamba_ap(self.norm_ap(f_ap_flat))
        g1 = self.gate_ap(self.norm_ap(f_lat_flat))
        out1 = (y1 * g1).transpose(1, 2).view(B, C, X_dim, Y_dim, Z_dim)

        y2 = self.mamba_lat(self.norm_lat(f_lat_flat))
        g2 = self.gate_lat(self.norm_lat(f_ap_flat))
        out2 = (y2 * g2).transpose(1, 2).view(B, C, X_dim, Y_dim, Z_dim)

        out = out1 + out2

        if self.use_residual:
            out_flat = out.flatten(2).transpose(1, 2)
            res_out = self.residual_block(self.norm_res(out_flat))
            out_flat = out_flat + res_out
            out = out_flat.transpose(1, 2).view(B, C, X_dim, Y_dim, Z_dim)

        return out


# ============================================================
# 6. 残差块
# ============================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 3, padding=1),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.InstanceNorm3d(dim),
            nn.Dropout3d(dropout)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

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
# 7. 分割头
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 8. 完整的 X2Shape 模型
# ============================================================

class X2Shape(nn.Module):
    def __init__(self, img_size=256, in_chans=1, num_classes=1,
                 encoder_channels=32, dims_3d=[32, 64, 128, 256],
                 vbp_output_channels=64, vbp_volume_size=64, dropout=0.0):
        super().__init__()
        self.img_size = img_size
        self.num_levels = len(dims_3d)

        vol_sizes = [vbp_volume_size, vbp_volume_size, vbp_volume_size]

        self.encoder_2d_ap = Simple2DEncoder(in_chans, encoder_channels, dropout=dropout)
        self.encoder_2d_lat = Simple2DEncoder(in_chans, encoder_channels, dropout=dropout)

        self.vbp_ap = VolumetricBackprojection(
            [encoder_channels], vbp_output_channels, vol_sizes, proj_type='ap', dropout=dropout
        )
        self.vbp_lat = VolumetricBackprojection(
            [encoder_channels], vbp_output_channels, vol_sizes, proj_type='lat', dropout=dropout
        )

        self.encoder_3d_ap = Encoder3D(vbp_output_channels, dims_3d, dropout=dropout)
        self.encoder_3d_lat = Encoder3D(vbp_output_channels, dims_3d, dropout=dropout)

        self.cross_mambas = nn.ModuleList([
            CrossMamba(dims_3d[3], dropout=dropout),
            CrossMamba(dims_3d[2], dropout=dropout),
            CrossMamba(dims_3d[1], dropout=dropout),
            CrossMamba(dims_3d[0], dropout=dropout)
        ])

        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose3d(dims_3d[3], dims_3d[2], kernel_size=2, stride=2),
            nn.ConvTranspose3d(dims_3d[2], dims_3d[1], kernel_size=2, stride=2),
            nn.ConvTranspose3d(dims_3d[1], dims_3d[0], kernel_size=2, stride=2)
        ])

        self.res_blocks = nn.ModuleList([
            ResidualBlock3D(dims_3d[2], dropout=dropout),
            ResidualBlock3D(dims_3d[1], dropout=dropout),
            ResidualBlock3D(dims_3d[0], dropout=dropout)
        ])

        self.final_res_block = ResidualBlock3D(dims_3d[0], dropout=dropout)
        self.seg_head = SegmentationHead(dims_3d[0], num_classes, dropout=dropout)
        self.vbp_jump_adjust = nn.Conv3d(vbp_output_channels, dims_3d[0], kernel_size=1)

    def forward(self, x_ap, x_lat, angle=None):
        f_ap_2d = self.encoder_2d_ap(x_ap)
        f_lat_2d = self.encoder_2d_lat(x_lat)

        f_ap_3d_base = self.vbp_ap(f_ap_2d, angle, proj_type='ap')
        f_lat_3d_base = self.vbp_lat(f_lat_2d, angle, proj_type='lat')

        vbp_jump_ap = f_ap_3d_base
        vbp_jump_lat = f_lat_3d_base

        f_ap_3d_multi = self.encoder_3d_ap(f_ap_3d_base)
        f_lat_3d_multi = self.encoder_3d_lat(f_lat_3d_base)

        cross_outputs = []
        for i in range(self.num_levels):
            cross_feat = self.cross_mambas[i](f_ap_3d_multi[i], f_lat_3d_multi[i])
            cross_outputs.append(cross_feat)

        current_feat = cross_outputs[0]

        for i in range(1, self.num_levels):
            upsampled = self.upsample_layers[i - 1](current_feat)
            current_cross = cross_outputs[i]

            if upsampled.shape[-3:] != current_cross.shape[-3:]:
                upsampled = F.interpolate(upsampled, size=current_cross.shape[-3:],
                                          mode='trilinear', align_corners=False)

            current_feat = self.res_blocks[i - 1](current_cross, upsampled)

        vbp_jump = vbp_jump_ap + vbp_jump_lat

        if vbp_jump.shape[1] != current_feat.shape[1]:
            vbp_jump = self.vbp_jump_adjust(vbp_jump)

        if current_feat.shape[-3:] != vbp_jump.shape[-3:]:
            current_feat = F.interpolate(current_feat, size=vbp_jump.shape[-3:],
                                         mode='trilinear', align_corners=False)

        final_feat = self.final_res_block(current_feat, vbp_jump)
        output = self.seg_head(final_feat)

        if output.shape[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size, self.img_size),
                                   mode='trilinear', align_corners=False)

        return output


# ============================================================
# 9. 工具函数
# ============================================================

def load_nifti_as_tensor(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def save_tensor_as_nifti(tensor, file_path, ref_nii_path=None):
    if tensor.dim() == 5:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()
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


def find_available_cases(base_path):
    available_cases = []

    if not os.path.exists(base_path):
        print(f"  错误: 路径不存在 {base_path}")
        return available_cases

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split('_')
        if len(parts) != 2:
            continue

        try:
            case_num = int(parts[0])
            angle = int(parts[1])
        except ValueError:
            continue

        ap_path = os.path.join(folder_path, "ap.nii.gz")
        lat_path = os.path.join(folder_path, "lat.nii.gz")
        mask_path = os.path.join(folder_path, "mask.nii.gz")

        if all([os.path.exists(p) for p in [ap_path, lat_path, mask_path]]):
            available_cases.append((case_num, angle, folder_name))
            print(f"  发现病例: {folder_name} (病例{case_num}, 角度{angle}°)")

    return available_cases


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


# ============================================================
# 10. 主程序
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("X2Shape 模型测试 - 完整模型 + 纯反投影验证")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")

    # 配置
    img_size = 256
    vbp_volume_size = 64
    encoder_channels = 32
    vbp_output_channels = 64
    dims_3d = [32, 64, 128, 256]

    print(f"\n配置:")
    print(f"  - 图像尺寸: {img_size}x{img_size}")
    print(f"  - VBP体积尺寸: {vbp_volume_size}³")
    print(f"  - 反投影方法: 真旋转 (grid_sample)")
    print(f"  - AP反投影: reverse_y=False")
    print(f"  - LAT反投影: reverse_y=True")

    # ============================================================
    # 第一部分：完整模型测试
    # ============================================================
    print("\n" + "=" * 70)
    print("第一部分: 完整模型前向传播测试")
    print("=" * 70)

    print("\n创建完整模型...")
    model = X2Shape(
        img_size=img_size,
        in_chans=1,
        num_classes=1,
        encoder_channels=encoder_channels,
        dims_3d=dims_3d,
        vbp_output_channels=vbp_output_channels,
        vbp_volume_size=vbp_volume_size,
        dropout=0.0
    ).to(device)
    model.eval()

    total_params, _ = count_parameters(model)
    print(f"\n模型总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 统计各模块参数量 - 修复：直接计算参数量
    print(f"\n各模块参数量:")


    def get_module_params(module):
        if isinstance(module, nn.ModuleList):
            return sum(p.numel() for m in module for p in m.parameters())
        else:
            return sum(p.numel() for p in module.parameters())


    module_names = [
        ('2D Encoder AP', get_module_params(model.encoder_2d_ap)),
        ('2D Encoder LAT', get_module_params(model.encoder_2d_lat)),
        ('VBP AP', get_module_params(model.vbp_ap)),
        ('VBP LAT', get_module_params(model.vbp_lat)),
        ('3D Encoder AP', get_module_params(model.encoder_3d_ap)),
        ('3D Encoder LAT', get_module_params(model.encoder_3d_lat)),
        ('Cross-Mamba (4个)', get_module_params(model.cross_mambas)),
        ('Upsample Layers', get_module_params(model.upsample_layers)),
        ('Res Blocks', get_module_params(model.res_blocks)),
        ('Final Res Block', get_module_params(model.final_res_block)),
        ('Seg Head', get_module_params(model.seg_head)),
        ('VBP Jump Adjust', get_module_params(model.vbp_jump_adjust)),
    ]

    for name, params in module_names:
        print(f"  - {name:20s}: {params:>10,} ({params / 1e6:.2f}M)")

    # 创建随机输入
    print("\n创建随机测试输入...")
    x_ap = torch.randn(1, 1, img_size, img_size).to(device)
    x_lat = torch.randn(1, 1, img_size, img_size).to(device)
    angle = 50

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 预热
    print("\n预热运行...")
    with torch.no_grad():
        _ = model(x_ap, x_lat, angle)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 正式测试
    print("\n正式测试完整模型前向传播...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    with torch.no_grad():
        output = model(x_ap, x_lat, angle)

    torch.cuda.synchronize()
    end_time = time.time()

    allocated, reserved = get_memory_usage()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

    print(f"\n前向传播完成!")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  - 推理时间: {(end_time - start_time) * 1000:.2f}ms")
    print(f"  - 峰值内存: {peak_memory:.2f}GB")

    # 清理
    del model, x_ap, x_lat, output
    torch.cuda.empty_cache()
    gc.collect()

    # ============================================================
    # 第二部分：纯反投影测试
    # ============================================================
    print("\n" + "=" * 70)
    print("第二部分: 纯反投影测试 (无可学习参数)")
    print("=" * 70)

    # 配置路径
    base_path = r"/mnt/d/med_data/biron/data1/test_vbp"
    output_base = r"/mnt/d/med_data/biron/data1/VBP_pure"

    print(f"\n数据路径: {base_path}")
    print(f"输出路径: {output_base}")

    # 创建纯反投影模块
    print("\n创建纯反投影模块...")
    backprojector = PureBackprojection(volume_shape=(img_size, img_size, img_size)).to(device)
    backprojector.eval()

    pure_params = sum(p.numel() for p in backprojector.parameters())
    print(f"  - 纯反投影参数量: {pure_params:,} (应=0，表示无可学习参数)")

    print("\n正在扫描可用病例...")
    print("-" * 50)
    available_cases = find_available_cases(base_path)

    if not available_cases:
        print("\n错误: 未找到任何有效的病例文件夹!")
        exit(1)

    print("-" * 50)
    print(f"\n找到 {len(available_cases)} 个可用病例")

    print("\n开始处理...")
    print("=" * 70)

    total_start_time = time.time()

    with torch.no_grad():
        for idx, (case_num, angle, folder_name) in enumerate(available_cases, 1):
            print(f"\n{'=' * 60}")
            print(f"[{idx}/{len(available_cases)}] 处理: {folder_name} (病例 {case_num}, 角度 {angle}°)")
            print(f"{'=' * 60}")

            folder_path = os.path.join(base_path, folder_name)
            ap_path = os.path.join(folder_path, "ap.nii.gz")
            lat_path = os.path.join(folder_path, "lat.nii.gz")
            mask_path = os.path.join(folder_path, "mask.nii.gz")

            try:
                print(f"  加载AP: {os.path.basename(ap_path)}")
                ap_tensor = load_nifti_as_tensor(ap_path).to(device)

                print(f"  加载LAT: {os.path.basename(lat_path)}")
                lat_tensor = load_nifti_as_tensor(lat_path).to(device)

                print(f"  加载Mask: {os.path.basename(mask_path)}")
                mask_tensor = load_nifti_as_tensor(mask_path).to(device)

                print(f"  AP形状: {ap_tensor.shape}")
                print(f"  LAT形状: {lat_tensor.shape}")
                print(f"  Mask形状: {mask_tensor.shape}")

                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()

                print(f"\n  执行纯反投影 (角度={angle}°)...")
                ap_volume, lat_volume = backprojector(ap_tensor, lat_tensor, angle)

                torch.cuda.synchronize()
                end_time = time.time()
                peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

                print(f"\n  反投影完成!")
                print(f"  - 处理时间: {(end_time - start_time) * 1000:.2f}ms")
                print(f"  - 峰值内存: {peak_memory:.2f}GB")
                print(f"  - AP体积形状: {ap_volume.shape}")
                print(f"  - LAT体积形状: {lat_volume.shape}")
                print(f"  - AP体积范围: [{ap_volume.min():.4f}, {ap_volume.max():.4f}]")
                print(f"  - LAT体积范围: [{lat_volume.min():.4f}, {lat_volume.max():.4f}]")
                print(f"  - AP非零体素: {torch.sum(ap_volume > 0).item()}")
                print(f"  - LAT非零体素: {torch.sum(lat_volume > 0).item()}")

                output_folder = os.path.join(output_base, folder_name)
                os.makedirs(output_folder, exist_ok=True)

                print(f"\n  保存结果到: {output_folder}")

                # 1. 保存AP反投影体积
                ap_path_out = os.path.join(output_folder, "ap_backprojection.nii.gz")
                save_tensor_as_nifti(ap_volume, ap_path_out, mask_path)

                # 2. 保存LAT反投影体积
                lat_path_out = os.path.join(output_folder, "lat_backprojection.nii.gz")
                save_tensor_as_nifti(lat_volume, lat_path_out, mask_path)

                # 3. 保存AP + LAT (叠加)
                combined_volume = ap_volume + lat_volume
                combined_path = os.path.join(output_folder, "combined_backprojection.nii.gz")
                save_tensor_as_nifti(combined_volume, combined_path, mask_path)

                # 4. 保存AP + LAT + Mask (叠加)
                # 确保mask与体积尺寸一致 (都是256³)
                if mask_tensor.shape[-3:] == combined_volume.shape[-3:]:
                    combined_mask = combined_volume + mask_tensor
                else:
                    # 如果mask尺寸不一致，先上采样
                    mask_resized = F.interpolate(mask_tensor, size=combined_volume.shape[-3:],
                                                 mode='trilinear', align_corners=False)
                    combined_mask = combined_volume + mask_resized

                combined_mask_path = os.path.join(output_folder, "combined_with_mask.nii.gz")
                save_tensor_as_nifti(combined_mask, combined_mask_path, mask_path)

                # 清理内存
                del ap_tensor, lat_tensor, mask_tensor
                del ap_volume, lat_volume, combined_volume, combined_mask
                torch.cuda.empty_cache()

                print(f"\n  ✓ 处理完成!")
                print(f"    保存的文件:")
                print(f"    - ap_backprojection.nii.gz (AP反投影)")
                print(f"    - lat_backprojection.nii.gz (LAT反投影)")
                print(f"    - combined_backprojection.nii.gz (AP+LAT)")
                print(f"    - combined_with_mask.nii.gz (AP+LAT+Mask)")

            except Exception as e:
                print(f"  ✗ 处理失败: {str(e)}")
                import traceback

                traceback.print_exc()
                torch.cuda.empty_cache()

    total_end_time = time.time()

    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"  - 成功处理: {len(available_cases)} 个病例")
    print(f"  - 总耗时: {(total_end_time - total_start_time):.2f}s")
    print(f"  - 平均每病例: {(total_end_time - total_start_time) / len(available_cases):.2f}s")
    print("=" * 70)
