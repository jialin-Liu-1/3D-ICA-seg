"""
模型1: 无参数VBP反投影 + PixelUnshuffle子图提取
输入: AP和LAT 2D图像 (256x256)
输出: 8个子图 (每个128x128x128)
无训练参数，纯函数式处理
支持从文件读取并保存结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import os
import time
from scipy.ndimage import rotate
from einops import rearrange


# ============================================================
# 1. 核心模块定义
# ============================================================

class VBPBackprojector(nn.Module):
    """
    无参数VBP反投影模块
    只进行反投影操作，不包含任何可学习参数
    """
    def __init__(self, volume_shape=(256, 256, 256)):
        super().__init__()
        self.volume_shape = volume_shape  # (X, Y, Z)
        self.X_dim, self.Y_dim, self.Z_dim = volume_shape

    def backproject_ap(self, f_2d, reverse_y=False):
        """
        AP反投影：沿Y轴复制
        f_2d: (B, C, X, Z) 或 (B, C, H, W)
        返回: (B, C, X, Y, Z)
        """
        B, C, H, W = f_2d.shape
        X_dim, Y_dim, Z_dim = self.volume_shape

        # 确保尺寸匹配 (H=X, W=Z)
        if H != X_dim or W != Z_dim:
            f_2d = F.interpolate(f_2d, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)

        # 沿Y轴复制 (B, C, X, Z) -> (B, C, X, Y, Z)
        f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)

        if reverse_y:
            f_3d = torch.flip(f_3d, dims=[3])

        return f_3d

    def backproject_lat(self, f_2d, angle_deg=90, reverse_y=False):
        """
        LAT反投影：沿Y轴复制 + 反向旋转
        f_2d: (B, C, X, Z)
        volume_shape: (X, Y, Z)
        angle_deg: 角度（度）
        reverse_y: 是否沿Y轴反向复制
        返回: (B, C, X, Y, Z)
        """
        B, C, H, W = f_2d.shape
        X_dim, Y_dim, Z_dim = self.volume_shape

        if H != X_dim or W != Z_dim:
            f_2d = F.interpolate(f_2d, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)

        # 如果角度为90度，与AP相同
        if abs(angle_deg - 90) < 1e-6:
            f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)
            if reverse_y:
                f_3d = torch.flip(f_3d, dims=[3])
            return f_3d

        # 第一步：沿Y轴复制
        f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)

        if reverse_y:
            f_3d = torch.flip(f_3d, dims=[3])

        # 第二步：反向旋转（绕X轴，即旋转Y-Z平面）
        rotation_angle = -angle_deg
        f_3d_np = f_3d.cpu().numpy()

        f_3d_rotated = np.zeros_like(f_3d_np)
        for b in range(B):
            for c in range(C):
                f_3d_rotated[b, c] = rotate(
                    f_3d_np[b, c],
                    rotation_angle,
                    axes=(1, 2),
                    reshape=False,
                    order=1,
                    cval=0
                )

        f_3d_result = torch.from_numpy(f_3d_rotated).to(f_3d.device)
        return f_3d_result

    def forward(self, x_ap, x_lat, angle=90, reverse_y=False):
        """
        前向传播

        Args:
            x_ap: AP图像 (B, C, H, W)
            x_lat: LAT图像 (B, C, H, W)
            angle: 投影角度（度）
            reverse_y: 是否沿Y轴反向

        Returns:
            fused_volume: 融合后的3D体积 (B, C, X, Y, Z)
        """
        # AP反投影
        ap_volume = self.backproject_ap(x_ap, reverse_y)

        # LAT反投影
        lat_volume = self.backproject_lat(x_lat, angle, reverse_y)

        # 融合 (相加)
        fused_volume = ap_volume + lat_volume

        return fused_volume, ap_volume, lat_volume


class SubvolumeExtractor(nn.Module):
    """
    子图提取模块：使用PixelUnshuffle将大体积切成8个子图
    输入: (B, C, X, Y, Z) 其中 X=Y=Z=256
    输出: (B*8, C, X/2, Y/2, Z/2) 其中每个子图尺寸为128x128x128
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, volume):
        """
        volume: (B, C, X, Y, Z)
        返回: (B * scale_factor^3, C, X/scale, Y/scale, Z/scale)
        """
        B, C, X, Y, Z = volume.shape
        s = self.scale_factor
        X_s, Y_s, Z_s = X // s, Y // s, Z // s

        # 使用rearrange进行子图提取
        subvolumes = rearrange(
            volume,
            'b c (x xp) (y yp) (z zp) -> (b xp yp zp) c x y z',
            x=X_s, y=Y_s, z=Z_s, xp=s, yp=s, zp=s
        )

        return subvolumes


class SubvolumeReconstructor(nn.Module):
    """
    子图重组模块：将8个子图合并回完整体积
    输入: (B*8, C, 128, 128, 128)
    输出: (B, C, 256, 256, 256)
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, subvolumes):
        """
        subvolumes: (B*8, C, 128, 128, 128)
        返回: (B, C, 256, 256, 256)
        """
        B_sub, C, X, Y, Z = subvolumes.shape
        B = B_sub // (self.scale_factor ** 3)

        s = self.scale_factor
        X_s, Y_s, Z_s = X * s, Y * s, Z * s

        volume = rearrange(
            subvolumes,
            '(b xp yp zp) c x y z -> b c (x xp) (y yp) (z zp)',
            b=B, xp=s, yp=s, zp=s
        )

        return volume


class VBPSubvolumeExtractor(nn.Module):
    """
    组合模块：VBP反投影 + 子图提取
    无训练参数，纯预处理模块
    """
    def __init__(self, volume_shape=(256, 256, 256), scale_factor=2):
        super().__init__()
        self.volume_shape = volume_shape
        self.scale_factor = scale_factor

        self.backprojector = VBPBackprojector(volume_shape)
        self.extractor = SubvolumeExtractor(scale_factor)
        self.reconstructor = SubvolumeReconstructor(scale_factor)

        # 计算输出尺寸
        X, Y, Z = volume_shape
        self.subvolume_shape = (X // scale_factor, Y // scale_factor, Z // scale_factor)
        self.num_subvolumes = scale_factor ** 3  # 8

    def forward(self, x_ap, x_lat, angle=90, reverse_y=False, return_all=False):
        """
        前向传播

        Args:
            x_ap: AP图像 (B, C, H, W)
            x_lat: LAT图像 (B, C, H, W)
            angle: 投影角度
            reverse_y: 是否沿Y轴反向
            return_all: 是否返回所有中间结果

        Returns:
            subvolumes: (B * 8, C, 128, 128, 128)
            如果 return_all=True: (subvolumes, ap_volume, lat_volume, fused_volume)
        """
        # 1. VBP反投影
        fused_volume, ap_volume, lat_volume = self.backprojector(x_ap, x_lat, angle, reverse_y)

        # 2. 子图提取
        subvolumes = self.extractor(fused_volume)

        if return_all:
            return subvolumes, ap_volume, lat_volume, fused_volume
        return subvolumes


# ============================================================
# 2. 辅助函数：加载和保存
# ============================================================

def load_nifti_as_tensor(file_path):
    """加载NIFTI文件并转换为PyTorch张量"""
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
    """将PyTorch张量保存为NIFTI文件"""
    if tensor.dim() == 5:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif tensor.dim() == 4:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()
    else:
        data = tensor.cpu().numpy()

    if data.ndim == 2:
        data = data[None, ...]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if ref_nii_path and os.path.exists(ref_nii_path):
        ref_nii = nib.load(ref_nii_path)
        affine = ref_nii.affine
    else:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)
    print(f"  已保存: {file_path}")


def save_subvolumes_as_nifti(subvolumes, output_dir, prefix="subvolume", ref_nii_path=None):
    """
    将子图批量保存为单独的NIFTI文件

    Args:
        subvolumes: (B*8, C, X, Y, Z)
        output_dir: 输出目录
        prefix: 文件名前缀
        ref_nii_path: 参考NIFTI文件路径
    """
    num_subvolumes = subvolumes.shape[0]
    for i in range(num_subvolumes):
        subvol = subvolumes[i:i+1]  # 保持维度 (1, C, X, Y, Z)
        file_path = os.path.join(output_dir, f"{prefix}_{i:02d}.nii.gz")
        save_tensor_as_nifti(subvol, file_path, ref_nii_path)


# ============================================================
# 3. 主处理函数
# ============================================================

def process_images(ap_path, lat_path, mask_path, output_dir,
                   volume_shape=(256, 256, 256), scale_factor=2,
                   angle=90, reverse_y=False, verbose=True):
    """
    处理单组图像：VBP反投影 + 子图提取 + 保存所有结果

    Args:
        ap_path: AP图像路径
        lat_path: LAT图像路径
        mask_path: Mask图像路径（用于参考）
        output_dir: 输出目录
        volume_shape: 三维体积形状 (X, Y, Z)
        scale_factor: 子图缩放因子
        angle: LAT投影角度（度）
        reverse_y: 是否沿Y轴反向复制
        verbose: 是否打印详细信息
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 时间统计 ==========
    timings = {
        'load_ap': 0.0,
        'load_lat': 0.0,
        'load_mask': 0.0,
        'resize': 0.0,
        'backproject': 0.0,
        'extract_subvolumes': 0.0,
        'save': 0.0,
        'total': 0.0
    }

    start_total = time.time()

    direction_str = "反向Y轴" if reverse_y else "正向Y轴"
    if verbose:
        print(f"  ========================================")
        print(f"  VBP反投影 + 子图提取")
        print(f"  角度: {angle}° (Y轴方向: {direction_str})")
        print(f"  缩放因子: {scale_factor} (子图尺寸: {volume_shape[0]//scale_factor}³)")
        print(f"  ========================================")

    # ========== 创建模块 ==========
    extractor = VBPSubvolumeExtractor(
        volume_shape=volume_shape,
        scale_factor=scale_factor
    ).to(device)

    # ========== 加载图像 ==========
    if verbose:
        print(f"  加载AP: {ap_path}")
    start = time.time()
    ap_tensor = load_nifti_as_tensor(ap_path).to(device)
    timings['load_ap'] = time.time() - start

    if verbose:
        print(f"  加载LAT: {lat_path}")
    start = time.time()
    lat_tensor = load_nifti_as_tensor(lat_path).to(device)
    timings['load_lat'] = time.time() - start

    if verbose:
        print(f"  加载Mask: {mask_path}")
    start = time.time()
    mask_tensor = load_nifti_as_tensor(mask_path).to(device)
    timings['load_mask'] = time.time() - start

    if verbose:
        print(f"  AP形状: {ap_tensor.shape}")
        print(f"  LAT形状: {lat_tensor.shape}")
        print(f"  Mask形状: {mask_tensor.shape}")

    # ========== 调整尺寸 ==========
    start = time.time()
    X_dim, Y_dim, Z_dim = volume_shape
    _, _, H, W = ap_tensor.shape

    if H != X_dim or W != Z_dim:
        if verbose:
            print(f"  调整图像尺寸: {H}x{W} -> {X_dim}x{Z_dim}")
        ap_tensor = F.interpolate(ap_tensor, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)
        lat_tensor = F.interpolate(lat_tensor, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)

    if mask_tensor.shape[-3:] != (X_dim, Y_dim, Z_dim):
        if verbose:
            print(f"  调整Mask尺寸: {mask_tensor.shape[-3:]} -> {(X_dim, Y_dim, Z_dim)}")
        mask_tensor = F.interpolate(mask_tensor, size=(X_dim, Y_dim, Z_dim),
                                    mode='trilinear', align_corners=False)
    timings['resize'] = time.time() - start

    # ========== VBP反投影 + 子图提取 ==========
    if verbose:
        print("\n  [VBP反投影 + 子图提取]...")
    start = time.time()
    with torch.no_grad():
        subvolumes, ap_volume, lat_volume, fused_volume = extractor(
            ap_tensor, lat_tensor, angle, reverse_y, return_all=True
        )
    timings['backproject'] = time.time() - start

    if verbose:
        print(f"  融合体积形状: {fused_volume.shape}")
        print(f"  子图数量: {subvolumes.shape[0]}")
        print(f"  每个子图尺寸: {subvolumes.shape[2:]}")

    # ========== 创建输出目录 ==========
    os.makedirs(output_dir, exist_ok=True)
    subvolume_dir = os.path.join(output_dir, "subvolumes")
    os.makedirs(subvolume_dir, exist_ok=True)

    dir_suffix = "reverse_y" if reverse_y else "forward_y"

    # ========== 保存结果 ==========
    if verbose:
        print("\n  [保存结果]...")
    start = time.time()

    # 1. 保存AP体积
    ap_output_path = os.path.join(output_dir, f"ap_volume.nii.gz")
    save_tensor_as_nifti(ap_volume, ap_output_path, mask_path)

    # 2. 保存LAT体积
    lat_output_path = os.path.join(output_dir, f"lat_volume_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(lat_volume, lat_output_path, mask_path)

    # 3. 保存融合体积
    fused_output_path = os.path.join(output_dir, f"fused_volume_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(fused_volume, fused_output_path, mask_path)

    # 4. 保存融合体积 + Mask
    fused_with_mask = fused_volume + mask_tensor
    fused_mask_path = os.path.join(output_dir, f"fused_with_mask_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(fused_with_mask, fused_mask_path, mask_path)

    # 5. 保存所有子图
    subvolume_prefix = f"subvolume_{angle}deg_{dir_suffix}"
    save_subvolumes_as_nifti(subvolumes, subvolume_dir, subvolume_prefix, mask_path)

    timings['save'] = time.time() - start
    timings['total'] = time.time() - start_total

    # ========== 打印时间统计 ==========
    print(f"\n  ⏱️  时间统计:")
    print(f"    - 加载AP: {timings['load_ap']:.4f}s")
    print(f"    - 加载LAT: {timings['load_lat']:.4f}s")
    print(f"    - 加载Mask: {timings['load_mask']:.4f}s")
    print(f"    - 调整尺寸: {timings['resize']:.4f}s")
    print(f"    - 反投影+子图提取: {timings['backproject']:.4f}s")
    print(f"    - 保存文件: {timings['save']:.4f}s")
    print(f"    - 总耗时: {timings['total']:.4f}s")

    if verbose:
        print(f"\n  ✓ 成功保存到: {output_dir}")
        print(f"  - AP体积: ap_volume.nii.gz")
        print(f"  - LAT体积: lat_volume_{angle}deg_{dir_suffix}.nii.gz")
        print(f"  - 融合体积: fused_volume_{angle}deg_{dir_suffix}.nii.gz")
        print(f"  - 融合+Mask: fused_with_mask_{angle}deg_{dir_suffix}.nii.gz")
        print(f"  - 子图: subvolumes/ (共{subvolumes.shape[0]}个子图)")

    return subvolumes, ap_volume, lat_volume, fused_volume, timings


def find_available_cases(base_path):
    """
    自动查找所有可用的病例文件夹
    返回: list of (case_num, angle, folder_name) tuples
    """
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
        else:
            missing = []
            if not os.path.exists(ap_path): missing.append("ap.nii.gz")
            if not os.path.exists(lat_path): missing.append("lat.nii.gz")
            if not os.path.exists(mask_path): missing.append("mask.nii.gz")
            print(f"  跳过文件夹 {folder_name}: 缺少 {', '.join(missing)}")

    return available_cases


# ============================================================
# 4. 测试代码
# ============================================================

def test_with_random_data():
    """使用随机数据测试"""
    print("=" * 60)
    print("测试: VBP反投影 + 子图提取模块 (随机数据)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = VBPSubvolumeExtractor(
        volume_shape=(256, 256, 256),
        scale_factor=2
    ).to(device)

    batch_size = 1
    x_ap = torch.randn(batch_size, 1, 256, 256).to(device)
    x_lat = torch.randn(batch_size, 1, 256, 256).to(device)

    print(f"\n输入 AP: {x_ap.shape}")
    print(f"输入 LAT: {x_lat.shape}")
    print(f"参数量: 0 (无训练参数)")

    with torch.no_grad():
        subvolumes = extractor(x_ap, x_lat, angle=90, reverse_y=False)

    print(f"\n输出子图: {subvolumes.shape}")
    print(f"  子图数量: {subvolumes.shape[0]}")
    print(f"  每个子图尺寸: {subvolumes.shape[2:]}")

    expected = (batch_size * 8, 1, 128, 128, 128)
    if subvolumes.shape == expected:
        print(f"\n✅ 测试通过！")
    else:
        print(f"\n❌ 形状错误: 期望 {expected}, 实际 {subvolumes.shape}")

    return extractor


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VBP反投影 + PixelUnshuffle子图提取")
    print("AP + LAT 反投影融合 → 8个子图 (128³)")
    print("=" * 60)

    # ============================================================
    # 🔧 配置路径
    # ============================================================
    base_path = r"/mnt/d/med_data/biron/data1/test_vbp"
    output_base = r"/mnt/d/med_data/biron/data1/VBP_subvolumes"
    volume_shape = (256, 256, 256)
    scale_factor = 2  # 256/2 = 128

    # ========== 参数配置 ==========
    ANGLE = 90  # 投影角度
    REVERSE_Y = True  # 是否沿Y轴反向
    SHOW_TIMING = True  # 是否显示详细时间统计

    print(f"\n数据路径: {base_path}")
    print(f"输出路径: {output_base}")
    print(f"设备: {device}")
    print(f"体积形状: X={volume_shape[0]}, Y={volume_shape[1]}, Z={volume_shape[2]}")
    print(f"子图尺寸: {volume_shape[0]//scale_factor}³ (共{scale_factor**3}个子图)")
    print(f"Y轴投影方向: {'反向Y轴' if REVERSE_Y else '正向Y轴'}")
    print(f"投影角度: {ANGLE}°")

    print("\n正在扫描可用病例...")
    print("-" * 50)
    available_cases = find_available_cases(base_path)

    if not available_cases:
        print("\n错误: 未找到任何有效的病例文件夹!")
        # 如果没有文件，运行随机数据测试
        print("\n运行随机数据测试...")
        test_with_random_data()
        exit(0)

    print("-" * 50)
    print(f"\n找到 {len(available_cases)} 个可用病例:")
    for case_num, angle, folder_name in available_cases:
        print(f"  - {folder_name} (病例 {case_num}, 角度 {angle}°)")

    print("\n开始处理...")
    print("=" * 60)

    processed_count = 0
    failed_cases = []

    # ========== 全局时间统计 ==========
    total_timings = {
        'total_time': 0.0,
        'num_cases': 0,
        'per_case_times': []
    }
    overall_start = time.time()

    for case_num, angle, folder_name in available_cases:
        print(f"\n{'=' * 60}")
        print(f"处理: {folder_name}")
        print(f"病例编号: {case_num}, 角度: {angle}°")
        print(f"{'=' * 60}")

        folder_path = os.path.join(base_path, folder_name)
        ap_path = os.path.join(folder_path, "ap.nii.gz")
        lat_path = os.path.join(folder_path, "lat.nii.gz")
        mask_path = os.path.join(folder_path, "mask.nii.gz")

        try:
            output_dir = os.path.join(output_base, folder_name)
            result = process_images(
                ap_path, lat_path, mask_path, output_dir,
                volume_shape, scale_factor,
                angle=angle,  # 使用文件夹名中的角度
                reverse_y=REVERSE_Y,
                verbose=SHOW_TIMING
            )
            if SHOW_TIMING:
                total_timings['per_case_times'].append({
                    'case': folder_name,
                    'angle': angle,
                    'timings': result[4]  # timings
                })
            processed_count += 1

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_cases.append(folder_name)

    total_timings['total_time'] = time.time() - overall_start
    total_timings['num_cases'] = processed_count

    # ========== 打印总结时间统计 ==========
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"  成功: {processed_count}/{len(available_cases)} 个病例")
    if failed_cases:
        print(f"  失败: {len(failed_cases)} 个病例")
        print(f"  失败的病例: {', '.join(failed_cases)}")

    if SHOW_TIMING and processed_count > 0:
        print("\n" + "-" * 60)
        print("⏱️  总体时间统计:")
        print("-" * 60)

        avg_times = {}
        for case_data in total_timings['per_case_times']:
            for key, value in case_data['timings'].items():
                if key not in avg_times:
                    avg_times[key] = []
                avg_times[key].append(value)

        print(f"\n  总处理病例数: {total_timings['num_cases']}")
        print(f"  总耗时: {total_timings['total_time']:.4f}s")
        print(f"  平均每病例耗时: {total_timings['total_time'] / total_timings['num_cases']:.4f}s")

        print("\n  各阶段平均耗时:")
        for key, values in avg_times.items():
            avg = sum(values) / len(values)
            print(f"    - {key}: {avg:.4f}s")

        if len(total_timings['per_case_times']) > 1:
            sorted_cases = sorted(
                total_timings['per_case_times'],
                key=lambda x: x['timings']['total']
            )
            print(f"\n  最快病例: {sorted_cases[0]['case']} ({sorted_cases[0]['timings']['total']:.4f}s)")
            print(f"  最慢病例: {sorted_cases[-1]['case']} ({sorted_cases[-1]['timings']['total']:.4f}s)")

    print("=" * 60)