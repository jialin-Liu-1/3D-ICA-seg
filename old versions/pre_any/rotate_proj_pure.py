import numpy as np
import nibabel as nib
import os
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
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
MASK_TEST_BASE = r"D:\med_data\biron\data1\slicer"
TRAIN_BASE = r"D:\med_data\biron\data2\train_any"

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
#ROTATION_ANGLES = list(range(0, 46, 45))
ROTATION_ANGLES = [0, 45]
PROJECTION_ANGLES = [50, 60, 70, 80, 85, 90]

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
# 二值化参数（已废弃，保留仅为兼容性）
# ============================================================
BINARY_THRESHOLD = np.float32(0.2)

# ============================================================
# 投影参数
# ============================================================
PROJECTION_SCALE_FACTOR = 100.0

# ============================================================
# 保存选项
# ============================================================
INVERT_PROJECTION = True

# ============================================================
# 重叠图参数
# ============================================================
AP_ALPHA = 0.7
LAT_ALPHA = 0.7
MASK_ALPHA = 0.4

# ============================================================
# 投影轴定义
# ============================================================
PROJECTION_AXIS = 1
LAT_ROTATION_AXES = (1, 2)

# ============================================================
# 运行模式
# ============================================================
RUN_MODE = "batch"
TEST_CASE_NUM = 0
TEST_ROTATION_IDX = 0
TEST_PROJ_IDX = 0

# ============================================================
# 血管剪枝参数（已废弃，保留仅为兼容性）
# ============================================================
ENABLE_KEEP_LARGEST = False
VOLUME_THRESHOLD = 500

print("=" * 80)
print("DSA扇形束蒙泰卡罗模拟 + 自定义夹角双平面投影系统")
print("=" * 80)
print(f"计算精度: {COMPUTE_DTYPE}, 保存精度: {SAVE_DTYPE}")
print(f"投影反转保存: {'是 (血管变暗)' if INVERT_PROJECTION else '否 (血管变亮)'}")
print(f"投影轴定义:")
print(f"  - AP投影轴: axis={PROJECTION_AXIS} (沿Y轴)")
print(f"  - LAT旋转轴: axis={LAT_ROTATION_AXES} (绕X轴旋转后沿Y轴投影)")
print(f"  - AP和LAT夹角: 自定义 ({PROJECTION_ANGLES}°)")
print(f"Mask来源: {MASK_TEST_BASE} (预生成，直接加载)")
print(f"DSA图像尺寸: {TARGET_3D_SIZE}³, 投影尺寸: {TARGET_2D_SIZE}²")
print(f"整体旋转角度: {ROTATION_ANGLES}")
print(f"投影夹角: {PROJECTION_ANGLES}")
print(f"总样本数: {len(ROTATION_ANGLES)} × {len(PROJECTION_ANGLES)} = {len(ROTATION_ANGLES) * len(PROJECTION_ANGLES)}")
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
    """处理DSA体积，转换为衰减系数"""
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
# 投影模拟函数
# ============================================================

def compute_projection(attenuation_volume, axis):
    """
    计算投影（不进行归一化，保留物理值）
    返回: 投影图像 (透射率值，范围0-1)
    """
    atten_f32 = attenuation_volume.astype(np.float32)
    voxel_spacing = np.float32(0.5)

    line_integral = np.sum(atten_f32, axis=axis) * voxel_spacing
    projection = np.exp(-line_integral)

    # 裁剪极端值（防止数值溢出）
    projection = np.clip(projection, 0, 1)

    return projection.astype(COMPUTE_DTYPE)


def normalize_projections(ap_proj, lat_proj):
    """
    使用统一的归一化范围对AP和LAT进行归一化
    """
    # 计算全局最小值和最大值
    global_min = min(ap_proj.min(), lat_proj.min())
    global_max = max(ap_proj.max(), lat_proj.max())

    if global_max - global_min > 1e-8:
        ap_normalized = (ap_proj - global_min) / (global_max - global_min)
        lat_normalized = (lat_proj - global_min) / (global_max - global_min)
    else:
        ap_normalized = np.ones_like(ap_proj) * 0.5
        lat_normalized = np.ones_like(lat_proj) * 0.5

    return ap_normalized.astype(COMPUTE_DTYPE), lat_normalized.astype(COMPUTE_DTYPE)


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

def process_single_combination(case_num, attenuation_volume, mask_pruned,
                               rot_angle_deg, proj_angle_deg,
                               rot_idx, proj_idx, output_dir_parent,
                               generate_overlay=False):
    """
    处理单个旋转角度和投影夹角的组合
    """
    # 文件夹命名
    folder_name = f"{case_num}{rot_idx:1d}_{proj_angle_deg}"
    output_dir = os.path.join(output_dir_parent, folder_name)
    ensure_dir(output_dir)

    ap_path = os.path.join(output_dir, "ap.nii.gz")
    lat_path = os.path.join(output_dir, "lat.nii.gz")
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    try:
        print(f"  整体旋转 {rot_angle_deg}°, 投影夹角 {proj_angle_deg}°...")

        # ========== 1. 获取衰减体积的最小值（用于填充空白区域） ==========
        atten_f32 = attenuation_volume.astype(np.float32)
        fill_value = np.min(atten_f32)
        print(f"    旋转填充值: {fill_value:.4f}")

        # ========== 2. 整体旋转体积（使用最小值填充空白区域） ==========
        rotated_atten_f32 = rotate(
            atten_f32,
            rot_angle_deg,
            axes=(1, 2),
            reshape=False,
            order=1,
            cval=fill_value
        )
        rotated_attenuation = rotated_atten_f32.astype(COMPUTE_DTYPE)

        # ========== 3. AP投影 ==========
        ap_projection = compute_projection(rotated_attenuation, axis=PROJECTION_AXIS)

        # ========== 4. LAT投影 ==========
        lat_atten_f32 = rotated_attenuation.astype(np.float32)
        lat_rotated_f32 = rotate(
            lat_atten_f32,
            proj_angle_deg,
            axes=LAT_ROTATION_AXES,
            reshape=False,
            order=1,
            cval=fill_value
        )
        lat_rotated_attenuation = lat_rotated_f32.astype(COMPUTE_DTYPE)
        lat_projection = compute_projection(lat_rotated_attenuation, axis=PROJECTION_AXIS)

        # ========== 5. 使用统一的归一化范围 ==========
        ap_normalized, lat_normalized = normalize_projections(ap_projection, lat_projection)

        # ========== 6. 缩放投影 ==========
        ap_projection_resized = resize_2d_image(ap_normalized, TARGET_2D_SIZE)
        lat_projection_resized = resize_2d_image(lat_normalized, TARGET_2D_SIZE)

        # ========== 7. 像素值反转 ==========
        if INVERT_PROJECTION:
            ap_projection_resized = 1.0 - ap_projection_resized
            lat_projection_resized = 1.0 - lat_projection_resized
            print(f"    已反转像素值 (血管变暗)")

        # ========== 8. Mask的投影 ==========
        rotated_mask = rotate(
            mask_pruned.astype(np.float32),
            rot_angle_deg,
            axes=(1, 2),
            reshape=False,
            order=0,
            cval=0
        )
        rotated_mask = (rotated_mask >= 0.5).astype(np.float32)

        mask_ap_projection = np.max(rotated_mask, axis=PROJECTION_AXIS)

        lat_rotated_mask = rotate(
            rotated_mask,
            proj_angle_deg,
            axes=LAT_ROTATION_AXES,
            reshape=False,
            order=0,
            cval=0
        )
        lat_rotated_mask = (lat_rotated_mask >= 0.5).astype(np.float32)
        mask_lat_projection = np.max(lat_rotated_mask, axis=PROJECTION_AXIS)

        mask_ap_resized = resize_2d_image(mask_ap_projection, TARGET_2D_SIZE)
        mask_lat_resized = resize_2d_image(mask_lat_projection, TARGET_2D_SIZE)

        # ========== 9. 保存文件 ==========
        save_as_nifti(ap_projection_resized, ap_path)
        save_as_nifti(lat_projection_resized, lat_path)
        save_as_nifti(mask_pruned, mask_path)

        # ========== 10. 生成叠加图（仅前两个组合） ==========
        if generate_overlay:
            overlay_dir = os.path.join(output_dir, "overlays")
            ensure_dir(overlay_dir)

            ap_overlay_path = os.path.join(overlay_dir, "AP_overlay.png")
            save_overlay_image(ap_projection_resized, mask_ap_resized, ap_overlay_path,
                               title1="AP Projection", title2="AP Mask (Loaded)",
                               cmap1='gray', cmap2='hot', alpha1=AP_ALPHA, alpha2=MASK_ALPHA)

            lat_overlay_path = os.path.join(overlay_dir, "LAT_overlay.png")
            save_overlay_image(lat_projection_resized, mask_lat_resized, lat_overlay_path,
                               title1="LAT Projection", title2="LAT Mask (Loaded)",
                               cmap1='gray', cmap2='hot', alpha1=LAT_ALPHA, alpha2=MASK_ALPHA)

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


def process_single_case(case_num, output_base, max_combinations_with_overlay=2):
    """处理单个病例"""

    # ========== 1. 加载DSA图像 ==========
    dsa_file = os.path.join(RAW_NII_BASE, f"{case_num}.nii.gz")
    if not os.path.exists(dsa_file):
        print(f"错误: DSA文件不存在 {dsa_file}")
        return 0, 0

    nii_img = nib.load(dsa_file)
    original_dsa = nii_img.get_fdata().astype(np.float32)

    print(f"\n病例 {case_num}:")
    print(f"  DSA形状: {original_dsa.shape}")
    print(f"  DSA像素范围: [{original_dsa.min():.4f}, {original_dsa.max():.4f}]")

    # ========== 2. 加载预生成mask ==========
    mask_file = os.path.join(MASK_TEST_BASE, f"{case_num}.nii.gz")
    if not os.path.exists(mask_file):
        print(f"错误: Mask文件不存在 {mask_file}")
        return 0, 0

    mask_img = nib.load(mask_file)
    mask_pruned = mask_img.get_fdata().astype(np.float32)
    mask_pruned = (mask_pruned >= 0.5).astype(np.uint8)

    print(f"  加载mask形状: {mask_pruned.shape}")
    print(f"  mask体素数: {np.sum(mask_pruned)}")
    print(f"  mask像素范围: [{mask_pruned.min():.0f}, {mask_pruned.max():.0f}]")

    # ========== 3. 处理DSA体积 ==========
    print(f"  预处理DSA体积（转换为衰减系数）...")
    attenuation_volume = process_dsa_volume(original_dsa, target_size=TARGET_3D_SIZE)

    # ========== 4. 调整尺寸 ==========
    if mask_pruned.shape[0] != TARGET_3D_SIZE:
        print(f"  调整mask尺寸: {mask_pruned.shape[0]} -> {TARGET_3D_SIZE}")
        zoom_factor = TARGET_3D_SIZE / mask_pruned.shape[0]
        mask_pruned_resized = zoom(mask_pruned.astype(np.float32), zoom_factor, order=0)
        mask_pruned = (mask_pruned_resized >= 0.5).astype(np.uint8)
        print(f"  调整后mask体素数: {np.sum(mask_pruned)}")

    if original_dsa.shape[0] != TARGET_3D_SIZE:
        zoom_factor = TARGET_3D_SIZE / original_dsa.shape[0]
        attenuation_volume = zoom(attenuation_volume.astype(np.float32), zoom_factor, order=1).astype(COMPUTE_DTYPE)

    # ========== 5. 处理所有组合 ==========
    success_count = 0
    total_combinations = len(ROTATION_ANGLES) * len(PROJECTION_ANGLES)
    combo_counter = 0

    for rot_idx, rot_angle in enumerate(ROTATION_ANGLES):
        for proj_idx, proj_angle in enumerate(PROJECTION_ANGLES):
            generate_overlay = (combo_counter < max_combinations_with_overlay)

            if process_single_combination(case_num, attenuation_volume, mask_pruned,
                                          rot_angle, proj_angle,
                                          rot_idx, proj_idx,
                                          output_base, generate_overlay):
                success_count += 1

            combo_counter += 1

    return success_count, total_combinations


def test_mode():
    """测试模式"""
    print("\n" + "=" * 80)
    print("测试模式 - DSA扇形束蒙泰卡罗模拟 + 自定义夹角双平面投影")
    print("=" * 80)
    print(f"测试病例: {TEST_CASE_NUM}")
    print(f"整体旋转索引: {TEST_ROTATION_IDX} (对应角度: {ROTATION_ANGLES[TEST_ROTATION_IDX]}°)")
    print(f"投影夹角索引: {TEST_PROJ_IDX} (对应角度: {PROJECTION_ANGLES[TEST_PROJ_IDX]}°)")
    print(f"投影反转: {'是 (血管变暗)' if INVERT_PROJECTION else '否'}")
    print(f"Mask来源: {MASK_TEST_BASE}")
    print("=" * 80)

    ensure_dir(TRAIN_BASE)

    process_single_case(TEST_CASE_NUM, TRAIN_BASE, max_combinations_with_overlay=2)

    folder_name = f"{TEST_ROTATION_IDX:02d}_{PROJECTION_ANGLES[TEST_PROJ_IDX]}"
    output_dir = os.path.join(TRAIN_BASE, folder_name)
    if os.path.exists(output_dir):
        print(f"\n✓ 成功保存到: {output_dir}")

        ap_test = nib.load(os.path.join(output_dir, "ap.nii.gz")).get_fdata()
        lat_test = nib.load(os.path.join(output_dir, "lat.nii.gz")).get_fdata()
        mask_test = nib.load(os.path.join(output_dir, "mask.nii.gz")).get_fdata()

        print(f"\n验证结果:")
        print(f"  AP投影形状: {ap_test.shape}, 范围: [{ap_test.min():.6f}, {ap_test.max():.6f}]")
        print(f"  LAT投影形状: {lat_test.shape}, 范围: [{lat_test.min():.6f}, {lat_test.max():.6f}]")
        print(f"  mask体素数: {np.sum(mask_test)}")
        print(f"  投影反转: {'是' if INVERT_PROJECTION else '否'}")
    else:
        print(f"\n✗ 处理失败")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def batch_mode():
    """批量模式"""
    print("\n" + "=" * 80)
    print("批量处理模式 (DSA + 自定义夹角双平面投影)")
    print("=" * 80)
    print(f"输出目录: {TRAIN_BASE}")
    print(f"投影反转: {'是 (血管变暗)' if INVERT_PROJECTION else '否'}")
    print(f"投影轴定义:")
    print(f"  - AP投影轴: axis={PROJECTION_AXIS} (沿Y轴)")
    print(f"  - LAT旋转轴: axis={LAT_ROTATION_AXES} (绕X轴旋转后沿Y轴投影)")
    print(f"  - AP和LAT夹角: 自定义 ({PROJECTION_ANGLES}°)")
    print(f"Mask来源: {MASK_TEST_BASE}")
    print(f"整体旋转角度: {ROTATION_ANGLES}")
    print(f"投影夹角: {PROJECTION_ANGLES}")
    print(
        f"总样本数: {len(ROTATION_ANGLES)} × {len(PROJECTION_ANGLES)} = {len(ROTATION_ANGLES) * len(PROJECTION_ANGLES)}")
    print("=" * 80)

    nii_files = glob.glob(os.path.join(RAW_NII_BASE, "*.nii.gz"))
    cases = []
    for f in nii_files:
        match = re.search(r"(\d+)\.nii\.gz$", os.path.basename(f))
        if match:
            case_num = int(match.group(1))
            mask_file = os.path.join(MASK_TEST_BASE, f"{case_num}.nii.gz")
            if os.path.exists(mask_file):
                cases.append(case_num)
            else:
                print(f"警告: 病例 {case_num} 的mask不存在，跳过")

    cases = sorted(cases)

    if len(cases) == 0:
        print("错误：没有找到可用的病例（同时需要DSA和mask）")
        return

    print(f"\n找到 {len(cases)} 个有效病例")
    ensure_dir(TRAIN_BASE)

    total_success = 0
    total_combinations = 0

    for case_num in tqdm(cases, desc="处理病例"):
        success, total = process_single_case(case_num, TRAIN_BASE, max_combinations_with_overlay=2)
        total_success += success
        total_combinations += total

    print("\n" + "=" * 80)
    print("批量处理完成！")
    print("=" * 80)
    print(f"处理病例数: {len(cases)}")
    print(f"成功处理组合: {total_success}/{total_combinations}")
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