import numpy as np
import nibabel as nib
import os
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom, map_coordinates
import warnings
import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage

warnings.filterwarnings('ignore')

# ============================================================
# 设置全局精度
# ============================================================
COMPUTE_DTYPE = np.float32
SAVE_DTYPE = np.float32

# ============================================================
# 参数配置
# ============================================================

# 基础路径
RAW_NII_BASE = r"D:\med_data\biron\data1\raw_nii"
TRAIN_BASE = r"D:\med_data\biron\data1\train"

# ============================================================
# 分辨率参数
# ============================================================
TARGET_3D_SIZE = 256
TARGET_2D_SIZE = 256

# ============================================================
# DSA图像参数
# ============================================================
VASCULAR_ATTENUATION = np.float32(0.05)
TISSUE_ATTENUATION = np.float32(0.03)
BACKGROUND_ATTENUATION = np.float32(0.02)
MAP_TO_REAL_ATTENUATION = True

# ============================================================
# 旋转参数
# ============================================================
START_ANGLE = 0
END_ANGLE = 90
STEP_ANGLE = 10

# ============================================================
# 扇形束几何参数（用于重排算法）
# ============================================================
SOURCE_TO_ISOCENTER = 500.0  # 源到等中心距离 (mm)
SOURCE_TO_DETECTOR = 1000.0  # 源到探测器距离 (mm)
DETECTOR_WIDTH = 400.0  # 探测器宽度 (mm)
NUM_DETECTOR_CHANNELS = 256  # 探测器通道数

# 计算派生几何参数
detector_channel_spacing = DETECTOR_WIDTH / NUM_DETECTOR_CHANNELS
gamma_max = np.arcsin(DETECTOR_WIDTH / 2 / SOURCE_TO_DETECTOR)
gamma_values = np.linspace(-gamma_max, gamma_max, NUM_DETECTOR_CHANNELS)

# 平行束参数
R = SOURCE_TO_ISOCENTER
num_parallel = 256
t_max = R * np.sin(gamma_max)
t_parallel = np.linspace(-t_max, t_max, num_parallel)

# ============================================================
# 扇形束正弦图采样角度（用于重排，覆盖180°）
# ============================================================
REBINNING_ANGLES = np.arange(0, 180, 2)  # 每2度一个投影，共90个角度

# ============================================================
# 蒙泰卡罗物理参数
# ============================================================
XRAY_TUBE_VOLTAGE = 80.0
USE_SCATTER = False
SCATTER_FRACTION = np.float32(0.3)

# ============================================================
# 二值化参数
# ============================================================
BINARY_THRESHOLD = np.float32(0.1)

# ============================================================
# 投影参数
# ============================================================
PROJECTION_SCALE_FACTOR = 100.0

# ============================================================
# 保存选项
# ============================================================
INVERT_PROJECTION = True  # True: 反转像素值后保存 (血管变暗，背景变亮)

# ============================================================
# 重叠图参数
# ============================================================
AP_ALPHA = 0.7
LAT_ALPHA = 0.7
MASK_ALPHA = 0.4

# ============================================================
# 投影轴定义
# ============================================================
# Y轴 (axis=1): 正面投影 (AP)
# Z轴 (axis=2): 侧面投影 (LAT)

# ============================================================
# 运行模式
# ============================================================
RUN_MODE = "batch"  # "batch" 或 "test"
TEST_CASE_NUM = 0
TEST_ANGLE_IDX = 0

# ============================================================
# 计算派生参数
# ============================================================
ANGLES = list(range(START_ANGLE, END_ANGLE, STEP_ANGLE))
NUM_ANGLES = len(ANGLES)

print("=" * 80)
print("DSA扇形束蒙泰卡罗模拟 + 重排为平行束投影系统 (GPU加速版)")
print("=" * 80)
print(f"计算精度: {COMPUTE_DTYPE}, 保存精度: {SAVE_DTYPE}")
print(f"GPU加速: 已启用 (CuPy)")
print(f"投影反转保存: {'是 (血管变暗)' if INVERT_PROJECTION else '否 (血管变亮)'}")
print(f"投影轴定义:")
print(f"  - 正面投影 (AP): Y轴 (axis=1)")
print(f"  - 侧面投影 (LAT): Z轴 (axis=2)")
print(f"DSA图像尺寸: {TARGET_3D_SIZE}³, 投影尺寸: {TARGET_2D_SIZE}²")
print(f"映射到真实衰减: {MAP_TO_REAL_ATTENUATION}")
if MAP_TO_REAL_ATTENUATION:
    print(f"  血管衰减: {VASCULAR_ATTENUATION} mm^-1")
    print(f"  组织衰减: {TISSUE_ATTENUATION} mm^-1")
    print(f"  背景衰减: {BACKGROUND_ATTENUATION} mm^-1")
print(f"扫描角度: {START_ANGLE}°~{END_ANGLE - STEP_ANGLE}°, 步长{STEP_ANGLE}°, 共{NUM_ANGLES}个")
print(f"扇形束几何: 源到等中心={SOURCE_TO_ISOCENTER}mm, 源到探测器={SOURCE_TO_DETECTOR}mm")
print(f"探测器: 宽度={DETECTOR_WIDTH}mm, 通道数={NUM_DETECTOR_CHANNELS}, 扇角={np.degrees(2 * gamma_max):.1f}°")
print(f"重排算法: 已启用 (扇形束→平行束), 采样角度={len(REBINNING_ANGLES)}个")
print("=" * 80)


# ============================================================
# 辅助函数
# ============================================================

def normalize_image(image):
    """将图像归一化到0-1范围"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-8:
        return image
    normalized = (image - min_val) / (max_val - min_val)
    return normalized


def save_overlay_image(proj_image, mask_image, output_path, title1="Projection", title2="Mask",
                       cmap1='gray', cmap2='hot', alpha1=0.7, alpha2=0.4):
    """保存投影和mask的重叠图"""
    img1_norm = normalize_image(proj_image.astype(np.float32))
    img2_norm = normalize_image(mask_image.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1_norm, cmap=cmap1, interpolation='nearest', origin='lower')
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=10)
    axes[0].set_ylabel('Y', fontsize=10)
    plt.colorbar(axes[0].images[0], ax=axes[0], shrink=0.8)

    axes[1].imshow(img2_norm, cmap=cmap2, interpolation='nearest', origin='lower')
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X', fontsize=10)
    axes[1].set_ylabel('Y', fontsize=10)
    plt.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)

    cmap1_func = plt.cm.get_cmap(cmap1)
    cmap2_func = plt.cm.get_cmap(cmap2)

    img1_rgb = cmap1_func(img1_norm)[:, :, :3]
    img2_rgb = cmap2_func(img2_norm)[:, :, :3]

    overlay = img1_rgb * alpha1 + img2_rgb * alpha2
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay, interpolation='nearest', origin='lower')
    axes[2].set_title(f'Overlay (Alpha: {alpha1}/{alpha2})', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X', fontsize=10)
    axes[2].set_ylabel('Y', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_image(ap_proj, lat_proj, mask_ap, mask_lat,
                          ap_overlay, lat_overlay, output_path, angle_deg):
    """保存综合对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(ap_proj, cmap='gray', interpolation='nearest', origin='lower')
    axes[0, 0].set_title(f'AP Projection (Angle: {angle_deg}°)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_ap, cmap='hot', interpolation='nearest', origin='lower')
    axes[0, 1].set_title(f'AP Mask Projection', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(ap_overlay, interpolation='nearest', origin='lower')
    axes[0, 2].set_title(f'AP Overlay', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(lat_proj, cmap='gray', interpolation='nearest', origin='lower')
    axes[1, 0].set_title(f'LAT Projection (Angle: {angle_deg}°)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_lat, cmap='hot', interpolation='nearest', origin='lower')
    axes[1, 1].set_title(f'LAT Mask Projection', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(lat_overlay, interpolation='nearest', origin='lower')
    axes[1, 2].set_title(f'LAT Overlay', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle(f'DSA Projections Comparison - Case - Angle {angle_deg}°',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# DSA图像预处理函数
# ============================================================

def dsa_to_attenuation(dsa_volume, threshold=0.1):
    """将DSA图像转换为真实的线性衰减系数"""
    dsa_float32 = dsa_volume.astype(np.float32)
    threshold_f32 = float(threshold)

    base_attenuation = np.float32(0.01)
    extra_attenuation = np.float32(0.06)

    attenuation_volume = base_attenuation + dsa_float32 * extra_attenuation

    vessel_mask = dsa_float32 >= threshold_f32
    attenuation_volume[vessel_mask] = base_attenuation + extra_attenuation * 1.5

    print(
        f"    衰减系数统计: min={attenuation_volume.min():.4f}, max={attenuation_volume.max():.4f}, mean={attenuation_volume.mean():.4f}")

    return attenuation_volume.astype(COMPUTE_DTYPE)


def process_dsa_volume(dsa_volume, target_size=256):
    """处理DSA体积"""
    current_size = dsa_volume.shape[0]

    if current_size != target_size:
        print(f"    尺寸调整: {current_size} -> {target_size}")
        zoom_factor = target_size / current_size
        processed_float32 = zoom(dsa_volume.astype(np.float32), zoom_factor, order=1)
        processed = processed_float32.astype(COMPUTE_DTYPE)
    else:
        print(f"    尺寸匹配: {current_size}，无需调整")
        processed = dsa_volume.astype(COMPUTE_DTYPE)

    if MAP_TO_REAL_ATTENUATION:
        print(f"    映射DSA值到真实衰减系数...")
        attenuation_volume = dsa_to_attenuation(processed, threshold=BINARY_THRESHOLD)
        return attenuation_volume
    else:
        print(f"    警告: 直接使用DSA值")
        return processed


# ============================================================
# GPU加速的空白背景修复函数
# ============================================================

def fix_rotation_artifacts_gpu(projection_gpu):
    """
    GPU版本：修复旋转产生的伪影
    将第一大像素值的像素替换为第二大像素值，再进行归一化
    """
    unique_vals = cp.unique(projection_gpu)

    if len(unique_vals) >= 2:
        max_val = unique_vals[-1]
        second_max_val = unique_vals[-2]

        max_mask = (projection_gpu >= max_val - 1e-6) & (projection_gpu <= max_val + 1e-6)
        max_count = cp.sum(max_mask)
        total_pixels = projection_gpu.size

        if float(max_count) / total_pixels < 0.3:
            print(f"    修复旋转伪影: 将 {max_count} 个最大值像素 ({max_val:.4f}) 替换为 {second_max_val:.4f}")
            projection_gpu[max_mask] = second_max_val

    return projection_gpu


# ============================================================
# 扇形束到平行束重排算法（核心）
# ============================================================

def generate_fan_beam_sinogram_gpu(attenuation_volume_gpu, axis, beta_angles_deg):
    """
    GPU加速：生成完整的扇形束正弦图

    输入：
        attenuation_volume_gpu: 3D衰减体积
        axis: 投影轴 (1=AP, 2=LAT)
        beta_angles_deg: 扇形束投影角度（度）

    返回：
        fan_sinogram: 扇形束正弦图 (num_angles, num_detector_channels)
    """

    num_angles = len(beta_angles_deg)
    fan_sinogram = np.zeros((num_angles, NUM_DETECTOR_CHANNELS), dtype=np.float32)

    voxel_spacing = 0.5

    for i, angle_deg in enumerate(beta_angles_deg):
        # 旋转体积
        rotated_gpu = cp_ndimage.rotate(attenuation_volume_gpu, angle_deg,
                                        axes=(1, 2), reshape=False, order=1)

        if axis == 1:  # AP投影（沿Y轴）
            # 沿Y轴求和，得到线积分
            line_integral = cp.sum(rotated_gpu, axis=1) * voxel_spacing  # (Z, X)

            # 沿X方向采样到探测器通道
            x_coords = cp.linspace(0, line_integral.shape[1] - 1, NUM_DETECTOR_CHANNELS)

            # 对每个Z层采样，然后平均
            sampled = cp.zeros((line_integral.shape[0], NUM_DETECTOR_CHANNELS), dtype=cp.float32)
            for ch in range(NUM_DETECTOR_CHANNELS):
                x_idx = x_coords[ch]
                x0 = int(cp.floor(x_idx))
                x1 = min(x0 + 1, line_integral.shape[1] - 1)
                dx = float(x_idx - x0)

                # 线性插值
                sampled[:, ch] = (1 - dx) * line_integral[:, x0] + dx * line_integral[:, x1]

            # 沿Z方向平均，得到最终的投影值
            fan_sinogram[i, :] = cp.asnumpy(cp.mean(sampled, axis=0))

        elif axis == 2:  # LAT投影（沿Z轴）
            # 沿Z轴求和
            line_integral = cp.sum(rotated_gpu, axis=0) * voxel_spacing  # (Y, X)

            x_coords = cp.linspace(0, line_integral.shape[1] - 1, NUM_DETECTOR_CHANNELS)

            sampled = cp.zeros((line_integral.shape[0], NUM_DETECTOR_CHANNELS), dtype=cp.float32)
            for ch in range(NUM_DETECTOR_CHANNELS):
                x_idx = x_coords[ch]
                x0 = int(cp.floor(x_idx))
                x1 = min(x0 + 1, line_integral.shape[1] - 1)
                dx = float(x_idx - x0)

                sampled[:, ch] = (1 - dx) * line_integral[:, x0] + dx * line_integral[:, x1]

            fan_sinogram[i, :] = cp.asnumpy(cp.mean(sampled, axis=0))

    return fan_sinogram


def fan_to_parallel_rebinning(fan_sinogram, beta_angles_deg):
    """
    扇形束到平行束的重排算法（核心！）

    重排公式：
        p_parallel(θ, t) = p_fan(β, γ)
        其中 θ = β + γ, t = R × sin(γ)

    参数：
        fan_sinogram: 扇形束正弦图 (num_beta, num_gamma)
        beta_angles_deg: 扇形束投影角度（度）

    返回：
        parallel_sinogram: 平行束正弦图 (num_theta, num_t)
        theta_angles_deg: 平行束角度（度）
        t_values: 平行束偏移值
    """

    num_beta = len(beta_angles_deg)
    num_gamma = fan_sinogram.shape[1]

    # 转换为弧度
    beta_rad = np.deg2rad(beta_angles_deg)

    # 计算平行束角度范围
    theta_min = beta_rad[0] + gamma_values[0]
    theta_max = beta_rad[-1] + gamma_values[-1]
    num_theta = num_beta
    theta_rad = np.linspace(theta_min, theta_max, num_theta)
    theta_deg = np.rad2deg(theta_rad)

    # 平行束偏移范围
    t_max = R * np.sin(gamma_max)
    t_values = np.linspace(-t_max, t_max, num_gamma)

    # 创建平行束正弦图
    parallel_sinogram = np.zeros((num_theta, num_gamma), dtype=np.float32)

    # 创建权重数组用于插值
    weight_sum = np.zeros((num_theta, num_gamma), dtype=np.float32)

    # 重排：将扇形束数据映射到平行束网格
    for beta_idx, beta in enumerate(beta_rad):
        for gamma_idx, gamma in enumerate(gamma_values):
            # 计算平行束坐标
            theta = beta + gamma
            t = R * np.sin(gamma)

            # 找到对应的平行束索引
            theta_idx = np.argmin(np.abs(theta_rad - theta))
            t_idx = np.argmin(np.abs(t_values - t))

            # 获取扇形束投影值
            fan_value = fan_sinogram[beta_idx, gamma_idx]

            # 累加到平行束正弦图
            parallel_sinogram[theta_idx, t_idx] += fan_value
            weight_sum[theta_idx, t_idx] += 1

    # 归一化（处理多个扇形束射线映射到同一个平行束射线的情况）
    weight_sum[weight_sum == 0] = 1
    parallel_sinogram = parallel_sinogram / weight_sum

    return parallel_sinogram, theta_deg, t_values


def extract_parallel_projection(parallel_sinogram, theta_deg, target_angle_deg):
    """
    从平行束正弦图中提取指定角度的投影
    """
    target_idx = np.argmin(np.abs(theta_deg - target_angle_deg))
    return parallel_sinogram[target_idx, :]


def fan_beam_with_rebinning_gpu(attenuation_volume_gpu, axis, target_angle_deg):
    """
    完整的扇形束投影 + 重排矫正（GPU加速版）

    流程：
    1. 生成多个角度的扇形束正弦图
    2. 重排为平行束正弦图
    3. 提取目标角度的平行束投影

    这样得到的投影与真正的平行束投影结构完全一致！
    """

    print(f"    生成扇形束正弦图 ({len(REBINNING_ANGLES)}个角度)...")

    # 1. 生成扇形束正弦图（需要多个角度才能完成重排）
    fan_sinogram = generate_fan_beam_sinogram_gpu(attenuation_volume_gpu, axis, REBINNING_ANGLES)

    print(f"    重排为平行束正弦图...")

    # 2. 重排为平行束
    parallel_sinogram, theta_deg, t_values = fan_to_parallel_rebinning(fan_sinogram, REBINNING_ANGLES)

    print(f"    提取目标角度 {target_angle_deg}° 投影...")

    # 3. 提取目标角度的投影
    parallel_projection = extract_parallel_projection(parallel_sinogram, theta_deg, target_angle_deg)

    # 4. 转换为GPU数组并缩放到目标尺寸
    projection_gpu = cp.asarray(parallel_projection, dtype=cp.float32)

    # 缩放到目标尺寸
    from cupyx.scipy.ndimage import zoom as zoom_gpu
    zoom_factor = TARGET_2D_SIZE / len(parallel_projection)
    projection_resized_gpu = zoom_gpu(projection_gpu, zoom_factor, order=1)

    # 确保是2D
    if projection_resized_gpu.ndim == 1:
        projection_resized_gpu = projection_resized_gpu[:, cp.newaxis]

    return projection_resized_gpu


# ============================================================
# GPU加速的投影模拟函数（使用重排算法）
# ============================================================

def simulate_projection_gpu(attenuation_volume_gpu, axis, angle_deg):
    """
    GPU加速的投影模拟（使用扇形束+重排算法）
    axis: 1-Y轴(正面AP), 2-Z轴(侧面LAT)
    angle_deg: 目标投影角度
    """

    print(f"    使用扇形束+重排算法生成投影...")

    # 使用完整的扇形束+重排算法
    projection_gpu = fan_beam_with_rebinning_gpu(attenuation_volume_gpu, axis, angle_deg)

    # 修复旋转伪影（背景消除功能）
    projection_gpu = fix_rotation_artifacts_gpu(projection_gpu)

    # 归一化到0-1范围
    proj_min = cp.min(projection_gpu)
    proj_max = cp.max(projection_gpu)
    if proj_max - proj_min > 1e-8:
        projection_gpu = (projection_gpu - proj_min) / (proj_max - proj_min)
    else:
        projection_gpu = cp.ones_like(projection_gpu) * 0.5

    return projection_gpu


def rotate_volume_gpu(volume_gpu, angle_deg):
    """
    GPU加速的3D体积旋转
    在Y-Z平面旋转 (axes=(1, 2))
    """
    return cp_ndimage.rotate(volume_gpu, angle_deg, axes=(1, 2), reshape=False, order=1)


def resize_2d_image_gpu(image_gpu, target_size):
    """
    GPU加速的2D图像缩放
    """
    current_size = image_gpu.shape[0]
    if current_size == target_size:
        return image_gpu

    zoom_factor = target_size / current_size

    x = cp.linspace(0, current_size - 1, target_size)
    y = cp.linspace(0, current_size - 1, target_size)
    xx, yy = cp.meshgrid(x, y)
    coords = cp.array([xx.ravel(), yy.ravel()])
    from cupyx.scipy.ndimage import map_coordinates
    resized = map_coordinates(image_gpu, coords, order=1).reshape(target_size, target_size)

    return resized


def save_as_nifti(data, output_path):
    """保存为NIfTI格式"""
    data_save = data.astype(SAVE_DTYPE)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii_img = nib.Nifti1Image(data_save, affine)
    nib.save(nii_img, output_path)


# ============================================================
# 核心处理函数（GPU加速版 + 重排算法）
# ============================================================

def process_single_angle_gpu(case_num, attenuation_volume, original_dsa_volume, angle_idx, output_base,
                             generate_overlay=False):
    """处理单个角度（GPU加速版 + 扇形束到平行束重排）"""
    angle_deg = ANGLES[angle_idx]

    # 直接使用 "caseNum_angleIdx" 格式作为文件夹名
    output_dir = os.path.join(output_base, f"{case_num}_{angle_idx}")
    os.makedirs(output_dir, exist_ok=True)

    ap_path = os.path.join(output_dir, "ap.nii.gz")  # 正面投影 (Y轴)
    lat_path = os.path.join(output_dir, "lat.nii.gz")  # 侧面投影 (Z轴)
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    try:
        print(f"  处理角度 {angle_deg}°...")

        # 1. 将数据移到GPU
        atten_gpu = cp.asarray(attenuation_volume.astype(np.float32))

        # 2. GPU投影计算（使用重排算法，不需要预先旋转！）
        # 重排算法内部会处理多角度采样，直接传入目标角度
        ap_projection_gpu = simulate_projection_gpu(atten_gpu, axis=1, angle_deg=angle_deg)
        lat_projection_gpu = simulate_projection_gpu(atten_gpu, axis=2, angle_deg=angle_deg)

        # 3. 转回CPU
        ap_projection_resized = cp.asnumpy(ap_projection_gpu)
        lat_projection_resized = cp.asnumpy(lat_projection_gpu)

        # 4. 根据选项决定是否反转投影（血管变暗）
        if INVERT_PROJECTION:
            ap_projection_resized = 1.0 - ap_projection_resized
            lat_projection_resized = 1.0 - lat_projection_resized
            print(f"    已反转像素值 (血管变暗)")

        # 5. 生成mask投影（CPU，使用简单旋转）
        dsa_f32 = original_dsa_volume.astype(np.float32)
        rotated_dsa_f32 = rotate(dsa_f32, angle_deg, axes=(1, 2), reshape=False, order=1)
        binary_mask = (rotated_dsa_f32 >= float(BINARY_THRESHOLD)).astype(np.float32)

        # mask投影：Y轴正面、Z轴侧面
        mask_ap_projection = np.max(binary_mask, axis=1)  # Y轴正面
        mask_lat_projection = np.max(binary_mask, axis=2)  # Z轴侧面

        # 缩放mask投影（CPU）
        from scipy.ndimage import zoom as zoom_cpu
        zoom_factor = TARGET_2D_SIZE / mask_ap_projection.shape[0]
        mask_ap_resized = zoom_cpu(mask_ap_projection, zoom_factor, order=1)
        mask_lat_resized = zoom_cpu(mask_lat_projection, zoom_factor, order=1)

        # 确保投影是2D的
        if ap_projection_resized.ndim == 1:
            ap_projection_resized = ap_projection_resized[:, np.newaxis]
        if lat_projection_resized.ndim == 1:
            lat_projection_resized = lat_projection_resized[:, np.newaxis]

        # 6. 保存文件
        save_as_nifti(ap_projection_resized, ap_path)
        save_as_nifti(lat_projection_resized, lat_path)
        save_as_nifti(binary_mask, mask_path)

        # 7. 生成可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ap_display = ap_projection_resized.astype(np.float32)
        im1 = axes[0, 0].imshow(ap_display, cmap='gray')
        axes[0, 0].set_title(f'AP Projection (Angle: {angle_deg}°)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        lat_display = lat_projection_resized.astype(np.float32)
        im2 = axes[0, 1].imshow(lat_display, cmap='gray')
        axes[0, 1].set_title(f'LAT Projection (Angle: {angle_deg}°)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        mid_slice = binary_mask.shape[2] // 2
        axes[1, 0].imshow(binary_mask[:, :, mid_slice], cmap='gray')
        axes[1, 0].set_title(f'Mask Mid Slice (Z={mid_slice})')
        axes[1, 0].axis('off')

        atten_mid = attenuation_volume.shape[2] // 2
        atten_display = attenuation_volume[:, :, atten_mid].astype(np.float32)
        im4 = axes[1, 1].imshow(atten_display, cmap='hot')
        axes[1, 1].set_title(f'Attenuation Map (Z={atten_mid})')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

        plt.tight_layout()
        vis_path = os.path.join(output_dir, f'visualization_{angle_deg}deg.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 8. 生成叠加图（仅前两个角度）
        if generate_overlay:
            overlay_dir = os.path.join(output_dir, "overlays")
            os.makedirs(overlay_dir, exist_ok=True)

            ap_overlay_path = os.path.join(overlay_dir, "AP_overlay.png")
            save_overlay_image(ap_projection_resized, mask_ap_resized, ap_overlay_path,
                               title1="AP Projection (Rebinned)", title2="AP Mask",
                               cmap1='gray', cmap2='hot', alpha1=AP_ALPHA, alpha2=MASK_ALPHA)

            lat_overlay_path = os.path.join(overlay_dir, "LAT_overlay.png")
            save_overlay_image(lat_projection_resized, mask_lat_resized, lat_overlay_path,
                               title1="LAT Projection (Rebinned)", title2="LAT Mask",
                               cmap1='gray', cmap2='hot', alpha1=LAT_ALPHA, alpha2=MASK_ALPHA)

            ap_overlay_img = plt.imread(ap_overlay_path)
            lat_overlay_img = plt.imread(lat_overlay_path)

            comparison_path = os.path.join(overlay_dir, f"comparison_{angle_deg}deg.png")
            save_comparison_image(
                ap_projection_resized.astype(np.float32),
                lat_projection_resized.astype(np.float32),
                mask_ap_resized.astype(np.float32),
                mask_lat_resized.astype(np.float32),
                ap_overlay_img, lat_overlay_img,
                comparison_path, angle_deg
            )

            print(f"    叠加图已保存到: {overlay_dir}")

        # 文件大小信息
        ap_size = os.path.getsize(ap_path) / 1024
        lat_size = os.path.getsize(lat_path) / 1024

        print(f"    已保存到: {output_dir}")
        print(
            f"    AP投影(重排后): {ap_projection_resized.shape}, 范围 [{ap_projection_resized.min():.4f}, {ap_projection_resized.max():.4f}], 大小: {ap_size:.1f}KB")
        print(
            f"    LAT投影(重排后): {lat_projection_resized.shape}, 范围 [{lat_projection_resized.min():.4f}, {lat_projection_resized.max():.4f}], 大小: {lat_size:.1f}KB")

        return True

    except Exception as e:
        print(f"    ✗ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_single_case_gpu(case_num, output_base, max_angles_with_overlay=2):
    """处理单个病例（GPU加速版 + 重排算法）"""
    input_file = os.path.join(RAW_NII_BASE, f"{case_num}.nii.gz")
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return 0, 0

    nii_img = nib.load(input_file)
    original_dsa = nii_img.get_fdata().astype(np.float32)

    print(f"\n病例 {case_num}:")
    print(f"  原始形状: {original_dsa.shape}")
    print(f"  像素范围: [{original_dsa.min():.4f}, {original_dsa.max():.4f}]")
    print(f"  非零像素比例: {np.sum(original_dsa > 0) / original_dsa.size * 100:.2f}%")

    if np.max(original_dsa) < 0.01:
        print(f"  警告: DSA图像最大值过小，可能没有血管结构")

    print(f"  预处理DSA体积...")
    attenuation_volume = process_dsa_volume(original_dsa, target_size=TARGET_3D_SIZE)

    original_dsa_resized = original_dsa
    if original_dsa.shape[0] != TARGET_3D_SIZE:
        zoom_factor = TARGET_3D_SIZE / original_dsa.shape[0]
        original_dsa_resized = zoom(original_dsa, zoom_factor, order=1).astype(np.float32)

    success_count = 0
    for angle_idx in range(NUM_ANGLES):
        generate_overlay = (angle_idx < max_angles_with_overlay)
        if process_single_angle_gpu(case_num, attenuation_volume, original_dsa_resized,
                                    angle_idx, output_base, generate_overlay):
            success_count += 1

    return success_count, NUM_ANGLES


def test_mode_gpu():
    """测试模式（GPU加速版 + 重排算法）"""
    print("\n" + "=" * 80)
    print("测试模式 - DSA扇形束蒙泰卡罗模拟 (GPU加速版 + 重排算法)")
    print("=" * 80)
    print(f"测试病例: {TEST_CASE_NUM}")
    print(f"测试角度索引: {TEST_ANGLE_IDX} (对应角度: {ANGLES[TEST_ANGLE_IDX]}°)")
    print("=" * 80)

    input_file = os.path.join(RAW_NII_BASE, f"{TEST_CASE_NUM}.nii.gz")
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return

    nii_img = nib.load(input_file)
    original_dsa = nii_img.get_fdata().astype(np.float32)

    print(f"\n原始DSA信息:")
    print(f"  形状: {original_dsa.shape}")
    print(f"  像素范围: [{original_dsa.min():.4f}, {original_dsa.max():.4f}]")
    print(f"  非零像素比例: {np.sum(original_dsa > 0) / original_dsa.size * 100:.2f}%")

    print(f"\n预处理DSA体积...")
    attenuation_volume = process_dsa_volume(original_dsa, target_size=TARGET_3D_SIZE)

    original_dsa_resized = original_dsa
    if original_dsa.shape[0] != TARGET_3D_SIZE:
        zoom_factor = TARGET_3D_SIZE / original_dsa.shape[0]
        original_dsa_resized = zoom(original_dsa, zoom_factor, order=1).astype(np.float32)

    # 直接使用 TRAIN_BASE 作为输出目录
    os.makedirs(TRAIN_BASE, exist_ok=True)

    print(f"\n处理角度索引 {TEST_ANGLE_IDX} ({ANGLES[TEST_ANGLE_IDX]}°)...")
    success = process_single_angle_gpu(TEST_CASE_NUM, attenuation_volume, original_dsa_resized,
                                       TEST_ANGLE_IDX, TRAIN_BASE, generate_overlay=True)

    if success:
        output_dir = os.path.join(TRAIN_BASE, f"{TEST_CASE_NUM}_{TEST_ANGLE_IDX}")
        print(f"\n✓ 成功保存到: {output_dir}")

        ap_test = nib.load(os.path.join(output_dir, "ap.nii.gz")).get_fdata()
        print(f"\n验证结果:")
        print(f"  AP投影数据类型: {ap_test.dtype}, 范围: [{ap_test.min():.6f}, {ap_test.max():.6f}]")
        print(f"  AP投影非零像素: {np.sum(ap_test > 0)} / {ap_test.size}")
        print(f"  投影反转: {'是' if INVERT_PROJECTION else '否'}")
    else:
        print(f"\n✗ 处理失败")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def batch_mode_gpu():
    """批量模式（GPU加速版 + 重排算法）"""
    print("\n" + "=" * 80)
    print("批量处理模式 (GPU加速版 + 扇形束到平行束重排)")
    print("=" * 80)
    print(f"输出目录: {TRAIN_BASE}")
    print(f"投影反转保存: {'是' if INVERT_PROJECTION else '否'}")
    print(f"投影轴定义:")
    print(f"  - 正面投影 (AP): Y轴 (axis=1)")
    print(f"  - 侧面投影 (LAT): Z轴 (axis=2)")
    print(f"  - 每个病例前2个角度生成叠加图")
    print(f"文件夹命名格式: {{病例号}}_{{角度索引}}")
    print(f"重排算法: 使用 {len(REBINNING_ANGLES)} 个采样角度进行扇形束→平行束转换")
    print("=" * 80)

    nii_files = glob.glob(os.path.join(RAW_NII_BASE, "*.nii.gz"))
    cases = []
    for f in nii_files:
        match = re.search(r"(\d+)\.nii\.gz$", os.path.basename(f))
        if match:
            cases.append(int(match.group(1)))
    cases = sorted(cases)

    if len(cases) == 0:
        print("错误：没有找到可用的病例")
        return

    print(f"\n找到 {len(cases)} 个病例")
    os.makedirs(TRAIN_BASE, exist_ok=True)

    total_success = 0
    total_angles = 0

    for case_num in tqdm(cases, desc="处理病例"):
        success, total = process_single_case_gpu(case_num, TRAIN_BASE, max_angles_with_overlay=2)
        total_success += success
        total_angles += total

    print("\n" + "=" * 80)
    print("批量处理完成！")
    print("=" * 80)
    print(f"处理病例数: {len(cases)}")
    print(f"成功处理角度: {total_success}/{total_angles}")
    print("=" * 80)


def main():
    if RUN_MODE == "test":
        test_mode_gpu()
    elif RUN_MODE == "batch":
        batch_mode_gpu()
    else:
        print(f"错误: 未知的运行模式 '{RUN_MODE}'")


if __name__ == "__main__":
    main()