"""
X2Shape: CT-free 3D multi-organ reconstruction with biplanar X-rays
适配版 - 支持通道减半和160尺寸
输入: 160x160, 输出: 160x160x160 单通道
支持自定义角度反投影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import nibabel as nib
import os
import glob

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
    输入: 160x160
    输出尺度: 80x80, 40x40, 20x20, 10x10
    通道: [dim0, dim1, dim2, dim3]
    """

    def __init__(self, in_chans=1, dims=[32, 64, 128, 256], depths=[1, 1, 2, 1]):
        super().__init__()
        self.dims = dims
        self.num_scales = len(dims)

        # Stem: 初始下采样 (160 -> 80)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )

        # 构建多阶段编码器
        self.stages = nn.ModuleList()
        current_dim = dims[0]

        for i in range(self.num_scales):
            stage_blocks = nn.Sequential()
            for _ in range(depths[i]):
                stage_blocks.append(MambaVisionBlock(current_dim))
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
        x = self.stem(x)  # (B, dims[0], 80, 80)
        features = []

        for layer in self.stages:
            x = layer(x)
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                if not isinstance(layer[0], nn.Conv2d):
                    features.append(x)

        return features  # [dim0@80², dim1@40², dim2@20², dim3@10²]


# ============================================================
# 2. 体积反投影模块 (VBP) - 支持角度控制
# ============================================================

class VolumetricBackprojection(nn.Module):
    """
    VBP: 接收多尺度 2D 特征 (4个尺度)，输出单一尺度 3D 特征
    支持角度控制，实现与投影过程一致的反投影
    """

    def __init__(self, multi_scale_channels, out_channels, volume_shape=(40, 40, 40)):
        super().__init__()
        self.multi_scale_channels = multi_scale_channels
        self.out_channels = out_channels
        self.volume_shape = volume_shape
        self.num_scales = len(multi_scale_channels)

        # 角度参数（在forward中设置）
        self.angle = None

        # 多尺度特征融合: 将不同尺度的 2D 特征转换到统一通道
        self.fusion_convs = nn.ModuleList()
        for i, ch in enumerate(multi_scale_channels):
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 可学习的位置权重
        self.position_weights = nn.Parameter(torch.ones(1, 1, volume_shape[0], volume_shape[1], volume_shape[2]))

    def set_angle(self, angle):
        """设置当前处理的角度"""
        self.angle = angle

    def backproject_with_angle(self, f_2d, angle_deg):
        """
        根据角度进行反投影
        角度控制层数复制的方向和范围
        angle_deg: 角度值（30, 40, 60, 70, 90）
        """
        B, C, H, W = f_2d.shape
        D, H_vol, W_vol = self.volume_shape

        # 根据角度计算投影的层数范围和方向
        if angle_deg == 90:
            # 正交投影：沿Z轴均匀复制
            f_3d = f_2d[:, :, None, :, :].expand(-1, -1, D, -1, -1)
        else:
            # 非正交投影：根据角度计算投影方向
            # 将角度转换为弧度
            angle_rad = torch.tensor(angle_deg * np.pi / 180.0, device=f_2d.device)

            # 计算在Z方向上的偏移量
            # 对于给定的角度，投影在Z轴上的范围是 D * tan(angle)
            max_offset = int(D * torch.tan(angle_rad) / 2)
            max_offset = min(max_offset, D // 2)  # 限制最大偏移

            # 创建初始3D体积
            f_3d = torch.zeros(B, C, D, H, W, device=f_2d.device)

            # 对每个Z层，根据角度计算对应的2D特征位置
            for z in range(D):
                # 计算该层在投影方向上的偏移
                # 使用线性映射：中心层偏移为0，边缘层偏移最大
                z_center = D / 2
                z_norm = (z - z_center) / z_center  # 范围[-1, 1]

                # 根据角度计算偏移量（在Y方向）
                offset_y = int(z_norm * max_offset)

                # 对2D特征进行平移（在Y方向）
                if offset_y != 0:
                    # 使用循环平移或填充
                    if offset_y > 0:
                        shifted = torch.roll(f_2d, shifts=offset_y, dims=-2)
                        # 填充边界
                        shifted[:, :, :offset_y, :] = 0
                    else:
                        shifted = torch.roll(f_2d, shifts=offset_y, dims=-2)
                        shifted[:, :, offset_y:, :] = 0
                else:
                    shifted = f_2d

                # 分配到对应的Z层
                f_3d[:, :, z, :, :] = shifted

            # 对非投影区域进行平滑处理
            # 使用高斯权重
            z_weights = torch.zeros(D, device=f_2d.device)
            for z in range(D):
                z_norm = (z - D / 2) / (D / 2)
                z_weights[z] = torch.exp(-z_norm ** 2 * 2)  # 高斯权重

            z_weights = z_weights.view(1, 1, D, 1, 1)
            f_3d = f_3d * z_weights

        return f_3d

    def forward(self, f_multi_scale, angle=None):
        """
        前向传播

        参数:
            f_multi_scale: 多尺度2D特征列表
            angle: 角度值（可选），如果不提供则使用self.angle
        """
        B = f_multi_scale[0].shape[0]
        D, H_vol, W_vol = self.volume_shape
        target_h = H_vol
        target_w = W_vol

        # 获取角度
        if angle is not None:
            self.angle = angle
        elif self.angle is None:
            self.angle = 90  # 默认使用正交投影

        fused_2d = []
        for i, f in enumerate(f_multi_scale):
            if f.shape[-2:] != (target_h, target_w):
                f_up = F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
            else:
                f_up = f
            f_proj = self.fusion_convs[i](f_up)
            fused_2d.append(f_proj)

        fused_2d = sum(fused_2d)

        # 使用角度控制的反投影
        f_3d = self.backproject_with_angle(fused_2d, self.angle)

        # 3D卷积处理
        out = self.project_3d(f_3d)

        # 应用位置权重
        pos_weight = self.position_weights.expand(B, -1, -1, -1, -1)
        out = out * torch.sigmoid(pos_weight)

        return out


# ============================================================
# 3. 3D Encoder (4个尺度特征金字塔)
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
    3D Encoder: 4个尺度
    输入: (B, in_chans, 40, 40, 40)
    输出尺度: 5³, 10³, 20³, 40³
    通道: [channels[0], channels[1], channels[2], channels[3]]
    """

    def __init__(self, in_chans, channels=[32, 64, 128, 256]):
        super().__init__()
        self.channels = channels

        self.blocks = nn.ModuleList()
        curr_ch = in_chans

        for i, out_ch in enumerate(channels):
            downsample = (i < len(channels) - 1)
            self.blocks.append(Encoder3DBlock(curr_ch, out_ch, downsample))
            curr_ch = out_ch

    def forward(self, x):
        features = []
        for block in self.blocks:
            x, feat = block(x)
            features.append(feat)
        # 返回从深到浅: [channels[3]@5³, channels[2]@10³, channels[1]@20³, channels[0]@40³]
        return features[::-1]


# ============================================================
# 4. Cross-Mamba 模块 (4个) - 修正维度
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
# 6. 分割头 (单通道输出)
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# 7. 完整的 X2Shape 模型 (适配通道减半和160尺寸)
# ============================================================

class X2Shape(nn.Module):
    def __init__(self, img_size=160, in_chans=1, num_classes=1,
                 dims_2d=[32, 64, 128, 256], depths_2d=[1, 1, 2, 1],
                 dims_3d=[32, 64, 128, 256], vbp_output_channels=64):
        super().__init__()
        self.img_size = img_size
        self.num_levels = len(dims_3d)  # 4层

        # 体积尺寸: 40 -> 20 -> 10 -> 5 (for img_size=160)
        vol_sizes = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        # ========== 1. 两个独立的 2D 编码器 ==========
        self.encoder_2d_ap = MambaVisionEncoder2D(in_chans, dims_2d, depths_2d)
        self.encoder_2d_lat = MambaVisionEncoder2D(in_chans, dims_2d, depths_2d)

        # ========== 2. VBP 模块 ==========
        self.vbp_ap = VolumetricBackprojection(dims_2d, vbp_output_channels, (vol_sizes[0], vol_sizes[0], vol_sizes[0]))
        self.vbp_lat = VolumetricBackprojection(dims_2d, vbp_output_channels,
                                                (vol_sizes[0], vol_sizes[0], vol_sizes[0]))

        # ========== 3. 两个独立的 3D Encoder ==========
        self.encoder_3d_ap = Encoder3D(vbp_output_channels, dims_3d)
        self.encoder_3d_lat = Encoder3D(vbp_output_channels, dims_3d)

        # ========== 4. Cross-Mamba 模块 (4个) - 修正维度匹配 ==========
        self.cross_mambas = nn.ModuleList([
            CrossMamba(dims_3d[3]),  # 最深尺度: 256 (5³)
            CrossMamba(dims_3d[2]),  # 次深尺度: 128 (10³)
            CrossMamba(dims_3d[1]),  # 次浅尺度: 64 (20³)
            CrossMamba(dims_3d[0])  # 最浅尺度: 32 (40³)
        ])

        # ========== 5. 上采样层 (3个) - 修正通道匹配 ==========
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose3d(dims_3d[3], dims_3d[2], kernel_size=2, stride=2),  # 256 -> 128 (5³ -> 10³)
            nn.ConvTranspose3d(dims_3d[2], dims_3d[1], kernel_size=2, stride=2),  # 128 -> 64 (10³ -> 20³)
            nn.ConvTranspose3d(dims_3d[1], dims_3d[0], kernel_size=2, stride=2)  # 64 -> 32 (20³ -> 40³)
        ])

        # ========== 6. 残差块 (3个) - 修正维度 ==========
        self.res_blocks = nn.ModuleList([
            ResidualBlock3D(dims_3d[2]),  # 128
            ResidualBlock3D(dims_3d[1]),  # 64
            ResidualBlock3D(dims_3d[0])  # 32
        ])

        # ========== 7. 最终残差块 ==========
        self.final_res_block = ResidualBlock3D(dims_3d[0])  # 32

        # ========== 8. 分割头 ==========
        self.seg_head = SegmentationHead(dims_3d[0], num_classes)

        # ========== 9. 原始 VBP 跳跃连接通道调整 ==========
        self.vbp_jump_adjust = nn.Conv3d(vbp_output_channels, dims_3d[0], kernel_size=1)

    def forward(self, x_ap, x_lat, angle=None):
        """
        前向传播

        参数:
            x_ap: AP投影图
            x_lat: LAT投影图
            angle: 角度值（用于VBP）
        """
        # Step 1: 2D 编码 (4个尺度)
        f_ap_2d = self.encoder_2d_ap(x_ap)  # [32@80², 64@40², 128@20², 256@10²]
        f_lat_2d = self.encoder_2d_lat(x_lat)

        # Step 2: VBP (传入角度)
        f_ap_3d_base = self.vbp_ap(f_ap_2d, angle)  # (B, 64, 40, 40, 40)
        f_lat_3d_base = self.vbp_lat(f_lat_2d, angle)  # (B, 64, 40, 40, 40)

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

        # Step 5: U-Net 升采样路径
        current_feat = cross_outputs[0]  # (B, 256, 5, 5, 5)

        for i in range(1, self.num_levels):
            upsampled = self.upsample_layers[i - 1](current_feat)
            current_cross = cross_outputs[i]

            if upsampled.shape[-3:] != current_cross.shape[-3:]:
                upsampled = F.interpolate(upsampled, size=current_cross.shape[-3:],
                                          mode='trilinear', align_corners=False)

            current_feat = self.res_blocks[i - 1](current_cross, upsampled)

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

        if output.shape[-1] != self.img_size:
            output = F.interpolate(output, size=(self.img_size, self.img_size, self.img_size),
                                   mode='trilinear', align_corners=False)

        return output


# ============================================================
# 测试代码 - 读取实际数据并测试
# ============================================================

def load_nifti_as_tensor(file_path):
    """加载NIFTI文件并转换为PyTorch张量"""
    if not os.path.exists(file_path):
        return None
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    # 添加batch和channel维度
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def save_tensor_as_nifti(tensor, file_path, ref_nii_path=None):
    """将PyTorch张量保存为NIFTI文件"""
    # 移除batch和channel维度
    if tensor.dim() == 5:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif tensor.dim() == 4:
        data = tensor.squeeze(0).cpu().numpy()
    else:
        data = tensor.cpu().numpy()

    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 获取参考affine
    if ref_nii_path and os.path.exists(ref_nii_path):
        ref_nii = nib.load(ref_nii_path)
        affine = ref_nii.affine
    else:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)
    print(f"  已保存: {file_path}")


def find_available_cases(base_path):
    """查找可用的病例和角度"""
    available_cases = []

    # 查找所有子文件夹
    if not os.path.exists(base_path):
        print(f"错误: 路径不存在 {base_path}")
        return available_cases

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # 检查是否包含所需的文件
        ap_path = os.path.join(folder_path, "ap.nii.gz")
        lat_path = os.path.join(folder_path, "lat.nii.gz")
        mask_path = os.path.join(folder_path, "mask.nii.gz")

        if all([os.path.exists(p) for p in [ap_path, lat_path, mask_path]]):
            # 解析文件夹名称，期望格式为 "病例号_角度"
            parts = folder.split('_')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                case_num = int(parts[0])
                angle = int(parts[1])
                available_cases.append((case_num, angle, folder_path))
                print(f"  找到: 病例 {case_num}, 角度 {angle}°")
            else:
                print(f"  跳过: 文件夹名称格式不正确 '{folder}'")
        else:
            print(f"  跳过: {folder} (缺少文件)")

    return available_cases


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("X2Shape 模型测试 - 支持角度控制的反投影")
    print("=" * 60)

    # 创建模型
    model = X2Shape(
        img_size=256,
        in_chans=1,
        num_classes=1,
        dims_2d=[32, 64, 128, 256],
        depths_2d=[1, 1, 2, 1],
        dims_3d=[32, 64, 128, 256],
        vbp_output_channels=64
    ).to(device)

    model.eval()

    base_path = r"D:\med_data\biron\data1\VBP\test"
    output_base = r"D:\med_data\biron\data1\VBP"

    print(f"\n查找数据路径: {base_path}")
    print("-" * 50)

    # 查找所有可用的病例
    available_cases = find_available_cases(base_path)

    print(f"\n找到 {len(available_cases)} 个可用病例")
    print(f"输出路径: {output_base}")
    print(f"设备: {device}")

    if len(available_cases) == 0:
        print("\n错误: 没有找到可用的数据！")
        print("请检查:")
        print(f"  1. 路径是否正确: {base_path}")
        print("  2. 文件夹名称格式是否为 '病例号_角度' (例如 '0_30')")
        print("  3. 每个文件夹是否包含 ap.nii.gz, lat.nii.gz, mask.nii.gz")
        exit(1)

    with torch.no_grad():
        for case_num, angle, case_path in available_cases:
            print(f"\n{'=' * 50}")
            print(f"处理病例 {case_num}，角度 {angle}°")
            print(f"路径: {case_path}")
            print(f"{'=' * 50}")

            try:
                # 构建文件路径
                ap_path = os.path.join(case_path, "ap.nii.gz")
                lat_path = os.path.join(case_path, "lat.nii.gz")
                mask_path = os.path.join(case_path, "mask.nii.gz")

                # 加载数据
                print(f"  加载AP: {ap_path}")
                ap_tensor = load_nifti_as_tensor(ap_path)
                if ap_tensor is None:
                    print(f"  ✗ 加载AP失败")
                    continue
                ap_tensor = ap_tensor.to(device)

                print(f"  加载LAT: {lat_path}")
                lat_tensor = load_nifti_as_tensor(lat_path)
                if lat_tensor is None:
                    print(f"  ✗ 加载LAT失败")
                    continue
                lat_tensor = lat_tensor.to(device)

                print(f"  加载Mask: {mask_path}")
                mask_tensor = load_nifti_as_tensor(mask_path)
                if mask_tensor is None:
                    print(f"  ✗ 加载Mask失败")
                    continue
                mask_tensor = mask_tensor.to(device)

                print(f"  AP形状: {ap_tensor.shape}")
                print(f"  LAT形状: {lat_tensor.shape}")
                print(f"  Mask形状: {mask_tensor.shape}")

                # 模型前向传播（传入角度）
                output = model(ap_tensor, lat_tensor, angle=angle)

                print(f"  输出形状: {output.shape}")
                print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

                # 保存输出
                output_folder = os.path.join(output_base, f"{case_num}_{angle}")
                output_path = os.path.join(output_folder, "vbp_output.nii.gz")

                # 将输出与mask相加
                # 确保尺寸匹配
                if mask_tensor.shape[-3:] != output.shape[-3:]:
                    mask_resized = F.interpolate(mask_tensor, size=output.shape[-3:],
                                                 mode='trilinear', align_corners=False)
                else:
                    mask_resized = mask_tensor

                # 相加
                combined = output + mask_resized
                combined_path = os.path.join(output_folder, "combined_output.nii.gz")

                # 保存
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