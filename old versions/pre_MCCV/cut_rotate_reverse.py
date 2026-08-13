import numpy as np
import nibabel as nib
import os
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom, binary_closing
from scipy import ndimage as ndi
from skimage import measure
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 设置全局精度
# ============================================================
COMPUTE_DTYPE = np.float32
SAVE_DTYPE = np.float32

# ============================================================
# 参数配置
# ============================================================

RAW_NII_BASE = r"D:\med_data\biron\data1\raw_nii"
TRAIN_BASE = r"D:\med_data\biron\data1\train1"

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
# 扇形束几何参数
# ============================================================
SOURCE_TO_ISOCENTER = 500.0
SOURCE_TO_DETECTOR = 1000.0
DETECTOR_WIDTH = 400.0
NUM_DETECTOR_CHANNELS = 256

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
# 运行模式
# ============================================================
RUN_MODE = "test"  # "batch" 或 "test"
TEST_CASE_NUM = 0
TEST_ANGLE_IDX = 0

# ============================================================
# 血管剪枝参数（连通域分析 + 形态学操作）
# ============================================================
ENABLE_VESSEL_PRUNING = True  # 是否启用血管剪枝
VOLUME_THRESHOLD = 6000  # 体积阈值（体素），小于此值的连通域被删除
CLOSING_ITERATIONS = 1  # 闭运算迭代次数，用于平滑边界

# ============================================================
# 计算派生参数
# ============================================================
ANGLES = list(range(START_ANGLE, END_ANGLE, STEP_ANGLE))
NUM_ANGLES = len(ANGLES)

print("=" * 80)
print("DSA扇形束蒙泰卡罗模拟 + 快速血管剪枝系统")
print("=" * 80)
print(f"计算精度: {COMPUTE_DTYPE}, 保存精度: {SAVE_DTYPE}")
print(f"投影反转保存: {'是 (血管变暗)' if INVERT_PROJECTION else '否 (血管变亮)'}")
print(f"快速剪枝: {'启用' if ENABLE_VESSEL_PRUNING else '禁用'}")
if ENABLE_VESSEL_PRUNING:
    print(f"  体积阈值: {VOLUME_THRESHOLD} 体素")
    print(f"  闭运算迭代: {CLOSING_ITERATIONS}")
print(f"DSA图像尺寸: {TARGET_3D_SIZE}³, 投影尺寸: {TARGET_2D_SIZE}²")
print(f"扫描角度: {START_ANGLE}°~{END_ANGLE - STEP_ANGLE}°, 步长{STEP_ANGLE}°, 共{NUM_ANGLES}个")
print("=" * 80)


# ============================================================
# 辅助函数
# ============================================================

def ensure_dir(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"  创建目录: {path}")


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
# 快速血管剪枝函数（连通域分析 + 形态学操作）
# ============================================================

def prune_vessel_fast(mask, volume_threshold=800, closing_iterations=1):
    """
    快速剪枝：只删除体积小的连通域
    速度：<2秒（256³）

    参数:
        mask: 二值血管mask
        volume_threshold: 体积阈值，小于此值的连通域被删除
        closing_iterations: 闭运算迭代次数，用于平滑边界
    """
    mask_bool = mask.astype(bool)
    original_voxels = np.sum(mask_bool)

    # 1. 连通域分析
    labeled_mask, num_features = measure.label(mask_bool, connectivity=2, return_num=True)

    # 2. 找出所有大于阈值的连通域
    keep_mask = np.zeros_like(mask_bool)
    kept_count = 0
    removed_voxels = 0

    for comp_id in range(1, num_features + 1):
        comp_size = np.sum(labeled_mask == comp_id)
        if comp_size >= volume_threshold:
            keep_mask[labeled_mask == comp_id] = True
            kept_count += 1
        else:
            removed_voxels += comp_size

    # 3. 可选：形态学平滑
    if closing_iterations > 0:
        struct_elem = np.ones((3, 3, 3), dtype=bool)
        keep_mask = binary_closing(keep_mask, struct_elem, iterations=closing_iterations)

    final_voxels = np.sum(keep_mask)

    print(
        f"    快速剪枝: {original_voxels} -> {final_voxels} 体素 (删除 {removed_voxels} 体素, {removed_voxels / original_voxels * 100:.1f}%)")
    print(f"    保留连通域: {kept_count} 个, 删除 {num_features - kept_count} 个")

    return keep_mask.astype(np.uint8)


def generate_binary_mask_from_dsa(dsa_volume, threshold):
    """从DSA体积生成二值mask"""
    return (dsa_volume >= threshold).astype(np.uint8)


# ============================================================
# DSA图像预处理函数
# ============================================================

def dsa_to_attenuation(dsa_volume, threshold=0.1):
    """
    将DSA图像转换为真实的线性衰减系数
    """
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
# 空白背景修复函数
# ============================================================

def fix_rotation_artifacts(projection):
    """
    修复旋转产生的伪影：将最大值（旋转空白区域）替换为第二大的像素值
    """
    unique_vals = np.unique(projection)

    if len(unique_vals) >= 2:
        max_val = unique_vals[-1]
        second_max_val = unique_vals[-2]

        max_mask = (projection >= max_val - 1e-6) & (projection <= max_val + 1e-6)
        max_count = np.sum(max_mask)
        total_pixels = projection.size

        if max_count / total_pixels < 0.3:
            print(f"    修复旋转伪影: 将 {max_count} 个最大值像素 ({max_val:.4f}) 替换为 {second_max_val:.4f}")
            projection[max_mask] = second_max_val

    return projection


# ============================================================
# 投影模拟函数
# ============================================================

def simulate_projection_fixed(attenuation_volume, axis):
    """
    改进的投影模拟（基于Beer-Lambert定律，带归一化和伪影修复）
    """
    atten_f32 = attenuation_volume.astype(np.float32)
    voxel_spacing = np.float32(0.5)

    line_integral = np.sum(atten_f32, axis=axis) * voxel_spacing

    if np.random.random() < 0.01:
        print(
            f"    线积分统计: min={line_integral.min():.4f}, max={line_integral.max():.4f}, mean={line_integral.mean():.4f}")

    projection = np.exp(-line_integral)

    if np.max(projection) < 1e-6:
        print(f"    警告: 投影接近全零！")
        projection = 1.0 - (line_integral - line_integral.min()) / (line_integral.max() - line_integral.min() + 1e-8)

    projection = fix_rotation_artifacts(projection)

    proj_min = projection.min()
    proj_max = projection.max()
    if proj_max - proj_min > 1e-8:
        projection = (projection - proj_min) / (proj_max - proj_min)
    else:
        projection = np.ones_like(projection) * 0.5

    return projection.astype(COMPUTE_DTYPE)


def resize_2d_image(image, target_size):
    """将2D图像缩放到目标尺寸"""
    current_size = image.shape[0]
    if current_size == target_size:
        return image

    zoom_factor = target_size / current_size
    image_f32 = image.astype(np.float32)
    resized_f32 = zoom(image_f32, zoom_factor, order=1)
    return resized_f32.astype(COMPUTE_DTYPE)


def save_as_nifti(data, output_path):
    """保存为NIfTI格式，自动创建目录"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    data_save = data.astype(SAVE_DTYPE)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii_img = nib.Nifti1Image(data_save, affine)
    nib.save(nii_img, output_path)


# ============================================================
# 核心处理函数
# ============================================================

def process_single_angle(case_num, attenuation_volume, original_dsa_volume, binary_mask, angle_idx, output_dir_parent,
                         generate_overlay=False):
    """处理单个角度"""
    angle_deg = ANGLES[angle_idx]

    # 病例文件夹：如 0_0
    case_folder = f"{case_num}_{angle_idx}"
    output_dir = os.path.join(output_dir_parent, case_folder)
    ensure_dir(output_dir)

    ap_path = os.path.join(output_dir, "ap.nii.gz")
    lat_path = os.path.join(output_dir, "lat.nii.gz")
    mask_path = os.path.join(output_dir, "mask.nii.gz")
    mask1_path = os.path.join(output_dir, "mask1.nii.gz")

    try:
        print(f"  处理角度 {angle_deg}°...")

        # 1. 旋转体积
        atten_f32 = attenuation_volume.astype(np.float32)
        rotated_atten_f32 = rotate(atten_f32, angle_deg, axes=(1, 2), reshape=False, order=1)
        rotated_attenuation = rotated_atten_f32.astype(COMPUTE_DTYPE)

        # 2. 投影计算
        ap_projection = simulate_projection_fixed(rotated_attenuation, axis=1)
        lat_projection = simulate_projection_fixed(rotated_attenuation, axis=0)

        # 3. 缩放投影
        ap_projection_resized = resize_2d_image(ap_projection, TARGET_2D_SIZE)
        lat_projection_resized = resize_2d_image(lat_projection, TARGET_2D_SIZE)

        # 4. 像素值反转（血管变暗，背景变亮）
        if INVERT_PROJECTION:
            ap_projection_resized = 1.0 - ap_projection_resized
            lat_projection_resized = 1.0 - lat_projection_resized
            print(f"    已反转像素值 (血管变暗)")

        # 5. 生成mask投影（使用剪枝后的mask）
        rotated_binary_mask = rotate(binary_mask.astype(np.float32), angle_deg, axes=(1, 2), reshape=False, order=0)
        rotated_binary_mask = (rotated_binary_mask >= 0.5).astype(np.float32)

        mask_ap_projection = np.max(rotated_binary_mask, axis=1)
        mask_lat_projection = np.max(rotated_binary_mask, axis=0)
        mask_ap_resized = resize_2d_image(mask_ap_projection, TARGET_2D_SIZE)
        mask_lat_resized = resize_2d_image(mask_lat_projection, TARGET_2D_SIZE)

        # 6. 保存文件（mask1在第一次调用时已保存，这里保存mask）
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

        # 8. 生成叠加图
        if generate_overlay:
            overlay_dir = os.path.join(output_dir, "overlays")
            ensure_dir(overlay_dir)

            ap_overlay_path = os.path.join(overlay_dir, "AP_overlay.png")
            save_overlay_image(ap_projection_resized, mask_ap_resized, ap_overlay_path,
                               title1="AP Projection", title2="AP Mask",
                               cmap1='gray', cmap2='hot', alpha1=AP_ALPHA, alpha2=MASK_ALPHA)

            lat_overlay_path = os.path.join(overlay_dir, "LAT_overlay.png")
            save_overlay_image(lat_projection_resized, mask_lat_resized, lat_overlay_path,
                               title1="LAT Projection", title2="LAT Mask",
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

        ap_size = os.path.getsize(ap_path) / 1024
        lat_size = os.path.getsize(lat_path) / 1024

        print(f"    已保存到: {output_dir}")
        print(
            f"    AP投影: {ap_projection_resized.shape}, 范围 [{ap_projection_resized.min():.4f}, {ap_projection_resized.max():.4f}], 大小: {ap_size:.1f}KB")
        print(
            f"    LAT投影: {lat_projection_resized.shape}, 范围 [{lat_projection_resized.min():.4f}, {lat_projection_resized.max():.4f}], 大小: {lat_size:.1f}KB")

        return True

    except Exception as e:
        print(f"    ✗ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_single_case(case_num, output_base, max_angles_with_overlay=2):
    """处理单个病例"""
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

    # 生成原始mask (mask1)
    print(f"  生成原始血管mask (mask1)...")
    mask1 = generate_binary_mask_from_dsa(original_dsa, BINARY_THRESHOLD)
    print(f"  原始mask1体素数: {np.sum(mask1)}")

    # 快速剪枝
    if ENABLE_VESSEL_PRUNING:
        print(f"  开始快速血管剪枝...")
        pruned_mask = prune_vessel_fast(
            mask1,
            volume_threshold=VOLUME_THRESHOLD,
            closing_iterations=CLOSING_ITERATIONS
        )
        final_mask = pruned_mask
    else:
        print(f"  血管剪枝已禁用，使用原始mask")
        final_mask = mask1

    binary_mask = final_mask.astype(np.uint8)

    print(f"  预处理DSA体积...")
    attenuation_volume = process_dsa_volume(original_dsa, target_size=TARGET_3D_SIZE)

    original_dsa_resized = original_dsa
    if original_dsa.shape[0] != TARGET_3D_SIZE:
        zoom_factor = TARGET_3D_SIZE / original_dsa.shape[0]
        original_dsa_resized = zoom(original_dsa, zoom_factor, order=1).astype(np.float32)

        # 同时调整mask尺寸
        mask1_resized = zoom(mask1.astype(np.float32), zoom_factor, order=0)
        mask1 = (mask1_resized >= 0.5).astype(np.uint8)

        binary_mask_resized = zoom(binary_mask.astype(np.float32), zoom_factor, order=0)
        binary_mask = (binary_mask_resized >= 0.5).astype(np.uint8)

    success_count = 0
    for angle_idx in range(NUM_ANGLES):
        generate_overlay = (angle_idx < max_angles_with_overlay)
        if process_single_angle(case_num, attenuation_volume, original_dsa_resized,
                                binary_mask, angle_idx, output_base, generate_overlay):
            success_count += 1

        # 每个角度文件夹内保存对应的mask1（只在第一个角度保存一次）
        if angle_idx == 0:
            first_angle_folder = os.path.join(output_base, f"{case_num}_{angle_idx}")
            mask1_path = os.path.join(first_angle_folder, "mask1.nii.gz")
            save_as_nifti(mask1, mask1_path)
            print(f"  原始mask (mask1) 已保存到: {mask1_path}")

    return success_count, NUM_ANGLES


def test_mode():
    """测试模式"""
    print("\n" + "=" * 80)
    print("测试模式 - DSA扇形束蒙泰卡罗模拟 + 快速剪枝")
    print("=" * 80)
    print(f"测试病例: {TEST_CASE_NUM}")
    print(f"测试角度索引: {TEST_ANGLE_IDX} (对应角度: {ANGLES[TEST_ANGLE_IDX]}°)")
    print(f"投影反转: {'是 (血管变暗)' if INVERT_PROJECTION else '否'}")
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

    ensure_dir(TRAIN_BASE)

    # 处理单个病例（会生成多个角度的文件夹，每个文件夹内有ap、lat、mask、mask1）
    process_single_case(TEST_CASE_NUM, TRAIN_BASE, max_angles_with_overlay=2)

    # 验证结果
    output_dir = os.path.join(TRAIN_BASE, f"{TEST_CASE_NUM}_{TEST_ANGLE_IDX}")
    if os.path.exists(output_dir):
        print(f"\n✓ 成功保存到: {output_dir}")

        # 检查文件
        ap_test = nib.load(os.path.join(output_dir, "ap.nii.gz")).get_fdata()
        mask_test = nib.load(os.path.join(output_dir, "mask.nii.gz")).get_fdata()
        mask1_test = nib.load(os.path.join(output_dir, "mask1.nii.gz")).get_fdata()

        print(f"\n验证结果:")
        print(f"  AP投影数据类型: {ap_test.dtype}, 范围: [{ap_test.min():.6f}, {ap_test.max():.6f}]")
        print(f"  mask体素数: {np.sum(mask_test)}")
        print(f"  mask1体素数: {np.sum(mask1_test)}")
        print(f"  投影反转: {'是' if INVERT_PROJECTION else '否'}")
    else:
        print(f"\n✗ 处理失败")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def batch_mode():
    """批量模式"""
    print("\n" + "=" * 80)
    print("批量处理模式 (DSA + 快速血管剪枝)")
    print("=" * 80)
    print(f"输出目录: {TRAIN_BASE}")
    print(f"投影反转: {'是 (血管变暗)' if INVERT_PROJECTION else '否'}")
    print(f"快速剪枝: {'启用' if ENABLE_VESSEL_PRUNING else '禁用'}")
    if ENABLE_VESSEL_PRUNING:
        print(f"  体积阈值: {VOLUME_THRESHOLD} 体素")
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
    ensure_dir(TRAIN_BASE)

    total_success = 0
    total_angles = 0

    for case_num in tqdm(cases, desc="处理病例"):
        success, total = process_single_case(case_num, TRAIN_BASE, max_angles_with_overlay=2)
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
        test_mode()
    elif RUN_MODE == "batch":
        batch_mode()
    else:
        print(f"错误: 未知的运行模式 '{RUN_MODE}'")


if __name__ == "__main__":
    main()