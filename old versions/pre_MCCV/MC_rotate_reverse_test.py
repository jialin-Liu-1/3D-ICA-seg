import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
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
RAW_NII_BASE = r"D:\med_data\biron\data1\raw_nii"
TEST_OUTPUT_BASE = r"D:\med_data\biron\data1\test"

# ============================================================
# 分辨率参数
# ============================================================
TARGET_3D_SIZE = 256
TARGET_2D_SIZE = 256

# ============================================================
# DSA图像参数
# ============================================================
MAP_TO_REAL_ATTENUATION = True
VASCULAR_ATTENUATION = np.float32(0.05)
TISSUE_ATTENUATION = np.float32(0.03)
BACKGROUND_ATTENUATION = np.float32(0.02)
BINARY_THRESHOLD = np.float32(0.1)

# ============================================================
# 旋转角度（测试用）
# ============================================================
TEST_ANGLES = [0, 30, 60, 90]

# ============================================================
# 重叠图参数
# ============================================================
PROJ_ALPHA = 0.7
MASK_ALPHA = 0.4

# ============================================================
# 投影轴定义和名称
# ============================================================
# axis=0: X轴投影 (左右方向) -> Y-Z平面
# axis=1: Y轴投影 (前后方向) -> X-Z平面
# axis=2: Z轴投影 (上下方向) -> X-Y平面
PROJECTION_AXES = {
    0: {"name": "X_axis", "title": "X-axis Projection (Left-Right)", "plane": "Y-Z Plane"},
    1: {"name": "Y_axis", "title": "Y-axis Projection (Anterior-Posterior)", "plane": "X-Z Plane"},
    2: {"name": "Z_axis", "title": "Z-axis Projection (Superior-Inferior)", "plane": "X-Y Plane"}
}

print("=" * 80)
print("三轴投影测试 - 保持原始旋转功能不变")
print("=" * 80)
print(f"测试病例: 0")
print(f"测试角度: {TEST_ANGLES}")
print(f"输出目录: {TEST_OUTPUT_BASE}")
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


def save_single_axis_overlay(proj_image, mask_image, output_path, axis_name, axis_title, plane_name, angle_deg,
                             cmap1='gray', cmap2='hot', alpha1=0.7, alpha2=0.4):
    """保存单个轴的投影和mask重叠图"""
    img1_norm = normalize_image(proj_image.astype(np.float32))
    img2_norm = normalize_image(mask_image.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始投影
    axes[0].imshow(img1_norm, cmap=cmap1, interpolation='nearest', origin='lower')
    axes[0].set_title(f'Projection: {axis_title}\nAngle: {angle_deg}°', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Pixel X', fontsize=10)
    axes[0].set_ylabel('Pixel Y', fontsize=10)
    plt.colorbar(axes[0].images[0], ax=axes[0], shrink=0.8)

    # Mask投影
    axes[1].imshow(img2_norm, cmap=cmap2, interpolation='nearest', origin='lower')
    axes[1].set_title(f'Mask Projection\n{plane_name}', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Pixel X', fontsize=10)
    axes[1].set_ylabel('Pixel Y', fontsize=10)
    plt.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)

    # 重叠图
    cmap1_func = plt.cm.get_cmap(cmap1)
    cmap2_func = plt.cm.get_cmap(cmap2)

    img1_rgb = cmap1_func(img1_norm)[:, :, :3]
    img2_rgb = cmap2_func(img2_norm)[:, :, :3]

    overlay = img1_rgb * alpha1 + img2_rgb * alpha2
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay, interpolation='nearest', origin='lower')
    axes[2].set_title(f'Overlay (Alpha: {alpha1}/{alpha2})\n{axis_name}', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Pixel X', fontsize=10)
    axes[2].set_ylabel('Pixel Y', fontsize=10)

    plt.suptitle(f'Projection along {axis_name} at {angle_deg}°', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    已保存: {os.path.basename(output_path)}")


def save_all_axes_comparison(all_projections, all_masks, output_path, angle_deg):
    """保存所有三个轴的对比图"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    axis_list = list(PROJECTION_AXES.keys())

    for i, axis in enumerate(axis_list):
        axis_info = PROJECTION_AXES[axis]

        # 投影图像
        proj_norm = normalize_image(all_projections[axis].astype(np.float32))
        axes[i, 0].imshow(proj_norm, cmap='gray', interpolation='nearest', origin='lower')
        axes[i, 0].set_title(f'{axis_info["title"]}\n{axis_info["plane"]}', fontsize=10, fontweight='bold')
        axes[i, 0].set_xlabel('X', fontsize=9)
        axes[i, 0].set_ylabel('Y', fontsize=9)

        # Mask图像
        mask_norm = normalize_image(all_masks[axis].astype(np.float32))
        axes[i, 1].imshow(mask_norm, cmap='hot', interpolation='nearest', origin='lower')
        axes[i, 1].set_title(f'Mask ({axis_info["name"]})', fontsize=10, fontweight='bold')
        axes[i, 1].set_xlabel('X', fontsize=9)
        axes[i, 1].set_ylabel('Y', fontsize=9)

        # 重叠图
        cmap_gray = plt.cm.get_cmap('gray')
        cmap_hot = plt.cm.get_cmap('hot')
        img1_rgb = cmap_gray(proj_norm)[:, :, :3]
        img2_rgb = cmap_hot(mask_norm)[:, :, :3]
        overlay = img1_rgb * PROJ_ALPHA + img2_rgb * MASK_ALPHA
        overlay = np.clip(overlay, 0, 1)

        axes[i, 2].imshow(overlay, interpolation='nearest', origin='lower')
        axes[i, 2].set_title(f'Overlay ({axis_info["name"]})', fontsize=10, fontweight='bold')
        axes[i, 2].set_xlabel('X', fontsize=9)
        axes[i, 2].set_ylabel('Y', fontsize=9)

    plt.suptitle(f'Three-Axis Projections Comparison at {angle_deg}°', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存综合对比图: {os.path.basename(output_path)}")


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
# GPU加速的投影模拟函数
# ============================================================

def simulate_projection_gpu(attenuation_volume_gpu, axis):
    """
    GPU加速的投影模拟
    axis: 0-X轴, 1-Y轴, 2-Z轴
    """
    voxel_spacing = cp.float32(0.5)
    line_integral = cp.sum(attenuation_volume_gpu, axis=axis) * voxel_spacing
    projection = cp.exp(-line_integral)

    # 归一化到0-1范围
    proj_min = cp.min(projection)
    proj_max = cp.max(projection)
    if proj_max - proj_min > 1e-8:
        projection = (projection - proj_min) / (proj_max - proj_min)
    else:
        projection = cp.ones_like(projection) * 0.5

    return projection


def rotate_volume_gpu(volume_gpu, angle_deg):
    """
    GPU加速的3D体积旋转 - 保持原始功能不变
    旋转轴: Y-Z平面 (axes=(1, 2))
    """
    return cp_ndimage.rotate(volume_gpu, angle_deg, axes=(1, 2), reshape=False, order=1)


def resize_2d_image_gpu(image_gpu, target_size):
    """GPU加速的2D图像缩放"""
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


# ============================================================
# 核心测试函数
# ============================================================

def test_three_axis_projections():
    """测试三个轴的投影"""

    # 创建输出目录
    os.makedirs(TEST_OUTPUT_BASE, exist_ok=True)

    # 加载病例0
    input_file = os.path.join(RAW_NII_BASE, f"0.nii.gz")
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return

    print(f"\n加载病例: 0")
    nii_img = nib.load(input_file)
    original_dsa = nii_img.get_fdata().astype(np.float32)

    print(f"  原始形状: {original_dsa.shape}")
    print(f"  像素范围: [{original_dsa.min():.4f}, {original_dsa.max():.4f}]")

    # 预处理DSA体积
    print(f"\n预处理DSA体积...")
    attenuation_volume = process_dsa_volume(original_dsa, target_size=TARGET_3D_SIZE)

    # 调整原始DSA尺寸
    original_dsa_resized = original_dsa
    if original_dsa.shape[0] != TARGET_3D_SIZE:
        zoom_factor = TARGET_3D_SIZE / original_dsa.shape[0]
        original_dsa_resized = zoom(original_dsa, zoom_factor, order=1).astype(np.float32)

    # 处理每个角度
    for angle_deg in TEST_ANGLES:
        print(f"\n{'=' * 60}")
        print(f"处理角度: {angle_deg}°")
        print(f"{'=' * 60}")

        # 创建角度输出目录
        angle_dir = os.path.join(TEST_OUTPUT_BASE, f"angle_{angle_deg}")
        os.makedirs(angle_dir, exist_ok=True)

        # GPU旋转
        print(f"  执行3D旋转 (axes=(1,2), Y-Z平面)...")
        atten_gpu = cp.asarray(attenuation_volume.astype(np.float32))
        rotated_atten_gpu = rotate_volume_gpu(atten_gpu, angle_deg)

        # 计算三个轴的投影
        all_projections = {}
        all_masks = {}

        print(f"  计算三个轴投影...")
        for axis in PROJECTION_AXES.keys():
            axis_name = PROJECTION_AXES[axis]["name"]
            print(f"    - {axis_name}")

            # GPU投影计算
            proj_gpu = simulate_projection_gpu(rotated_atten_gpu, axis)
            proj_resized_gpu = resize_2d_image_gpu(proj_gpu, TARGET_2D_SIZE)
            projection = cp.asnumpy(proj_resized_gpu)

            # 反转投影（血管变暗）
            projection = 1.0 - projection

            all_projections[axis] = projection

            # 生成mask投影（CPU）
            dsa_f32 = original_dsa_resized.astype(np.float32)
            rotated_dsa_f32 = rotate(dsa_f32, angle_deg, axes=(1, 2), reshape=False, order=1)
            binary_mask = (rotated_dsa_f32 >= float(BINARY_THRESHOLD)).astype(np.float32)

            mask_proj = np.max(binary_mask, axis=axis)
            zoom_factor = TARGET_2D_SIZE / mask_proj.shape[0]
            from scipy.ndimage import zoom as zoom_cpu
            mask_resized = zoom_cpu(mask_proj, zoom_factor, order=1)

            all_masks[axis] = mask_resized

            # 保存单个轴的叠加图
            overlay_path = os.path.join(angle_dir, f"overlay_{axis_name}.png")
            save_single_axis_overlay(
                projection, mask_resized, overlay_path,
                axis_name, PROJECTION_AXES[axis]["title"],
                PROJECTION_AXES[axis]["plane"], angle_deg,
                alpha1=PROJ_ALPHA, alpha2=MASK_ALPHA
            )

        # 保存所有三个轴的综合对比图
        comparison_path = os.path.join(angle_dir, "comparison_all_axes.png")
        save_all_axes_comparison(all_projections, all_masks, comparison_path, angle_deg)

        print(f"\n  ✓ 角度 {angle_deg}° 处理完成，保存到: {angle_dir}")

    print("\n" + "=" * 80)
    print("所有测试完成！")
    print(f"结果保存在: {TEST_OUTPUT_BASE}")
    print("=" * 80)


def main():
    test_three_axis_projections()


if __name__ == "__main__":
    main()