import numpy as np
import nibabel as nib
import os
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
import warnings
import cupy as cp

warnings.filterwarnings('ignore')

# ============================================================
# 设置全局精度（内部计算用float32，保持精度）
# ============================================================
COMPUTE_DTYPE = np.float32  # 改用float32保证精度
SAVE_DTYPE = np.float32  # 保存精度

# ============================================================
# 参数配置
# ============================================================

# 基础路径
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
# 真实DSA中，衰减系数需要调整到合理范围
# 对于DSA减影图像，血管与背景的衰减差异是关键
VASCULAR_ATTENUATION = np.float32(0.05)  # 降低血管衰减系数
TISSUE_ATTENUATION = np.float32(0.03)  # 降低组织衰减系数
BACKGROUND_ATTENUATION = np.float32(0.02)  # 降低背景衰减系数

# 是否将DSA值映射到真实衰减系数
MAP_TO_REAL_ATTENUATION = True

# ============================================================
# 旋转参数
# ============================================================
START_ANGLE = 0
END_ANGLE = 90
STEP_ANGLE = 30

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
BINARY_THRESHOLD = np.float32(0.2)

# ============================================================
# 投影参数
# ============================================================
# 添加一个缩放因子来控制投影对比度
PROJECTION_SCALE_FACTOR = 100.0  # 缩放因子，调整投影对比度

# ============================================================
# 重叠图参数
# ============================================================
AP_ALPHA = 0.7
LAT_ALPHA = 0.7
MASK_ALPHA = 0.4

# ============================================================
# 运行模式
# ============================================================
RUN_MODE = "batch"
TEST_CASE_NUM = 0
TEST_ANGLE_IDX = 0

# ============================================================
# 计算派生参数
# ============================================================
ANGLES = list(range(START_ANGLE, END_ANGLE, STEP_ANGLE))
NUM_ANGLES = len(ANGLES)

print("=" * 80)
print("DSA扇形束蒙泰卡罗模拟 + 重排为平行束投影系统")
print("=" * 80)
print(f"计算精度: {COMPUTE_DTYPE}, 保存精度: {SAVE_DTYPE}")
print(f"DSA图像尺寸: {TARGET_3D_SIZE}³, 投影尺寸: {TARGET_2D_SIZE}²")
print(f"映射到真实衰减: {MAP_TO_REAL_ATTENUATION}")
if MAP_TO_REAL_ATTENUATION:
    print(f"  血管衰减: {VASCULAR_ATTENUATION} mm^-1")
    print(f"  组织衰减: {TISSUE_ATTENUATION} mm^-1")
    print(f"  背景衰减: {BACKGROUND_ATTENUATION} mm^-1")
print(f"扫描角度: {START_ANGLE}°~{END_ANGLE - STEP_ANGLE}°, 步长{STEP_ANGLE}°, 共{NUM_ANGLES}个")
print(f"投影缩放因子: {PROJECTION_SCALE_FACTOR}")
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
    """
    将DSA图像（0-1范围）转换为真实的线性衰减系数

    关键修改：DSA是减影图像，背景应该接近0，血管有正对比度
    """
    dsa_float32 = dsa_volume.astype(np.float32)
    threshold_f32 = float(threshold)

    # DSA减影图像：背景≈0，血管>0
    # 衰减系数 = 基础衰减 + DSA值 × 额外衰减
    base_attenuation = np.float32(0.01)  # 基础衰减（背景）
    extra_attenuation = np.float32(0.06)  # 额外衰减（血管贡献）

    attenuation_volume = base_attenuation + dsa_float32 * extra_attenuation

    # 确保血管区域有足够高的对比度
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
# 空白背景修复函数（新增）
# ============================================================

def fix_rotation_artifacts(projection):
    """
    修复旋转产生的伪影：将最大值（旋转空白区域）替换为第二大的像素值

    参数:
        projection: 输入投影图像

    返回:
        fixed_projection: 修复后的投影图像
    """
    # 获取唯一的像素值
    unique_vals = np.unique(projection)

    if len(unique_vals) >= 2:
        # 找到最大值和第二大值
        max_val = unique_vals[-1]
        second_max_val = unique_vals[-2]

        # 创建mask标记最大值像素
        max_mask = (projection >= max_val - 1e-6) & (projection <= max_val + 1e-6)

        # 统计最大值像素的数量
        max_count = np.sum(max_mask)
        total_pixels = projection.size

        # 如果最大值像素占比小于30%（避免误伤正常的高亮区域）
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
    # 转换为float32进行计算
    atten_f32 = attenuation_volume.astype(np.float32)

    # 体素间距（mm）
    voxel_spacing = np.float32(0.5)

    # 计算线积分（沿指定轴求和）
    line_integral = np.sum(atten_f32, axis=axis) * voxel_spacing

    # 打印调试信息
    if np.random.random() < 0.01:
        print(
            f"    线积分统计: min={line_integral.min():.4f}, max={line_integral.max():.4f}, mean={line_integral.mean():.4f}")

    # 使用Beer-Lambert定律
    projection = np.exp(-line_integral)

    # 检查投影是否全零
    if np.max(projection) < 1e-6:
        print(f"    警告: 投影接近全零！线积分过大，需要调整衰减系数")
        projection = 1.0 - (line_integral - line_integral.min()) / (line_integral.max() - line_integral.min() + 1e-8)

    # ===== 修复旋转伪影（在归一化之前） =====
    projection = fix_rotation_artifacts(projection)

    # 归一化到0-1范围
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
    """保存为NIfTI格式"""
    data_save = data.astype(SAVE_DTYPE)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii_img = nib.Nifti1Image(data_save, affine)
    nib.save(nii_img, output_path)


# ============================================================
# 核心处理函数
# ============================================================

def process_single_angle(case_num, attenuation_volume, original_dsa_volume, angle_idx, output_base,
                         generate_overlay=False):
    """处理单个角度"""
    angle_deg = ANGLES[angle_idx]

    output_dir = os.path.join(output_base, f"{case_num}_{angle_idx}")
    os.makedirs(output_dir, exist_ok=True)

    ap_path = os.path.join(output_dir, "ap.nii.gz")
    lat_path = os.path.join(output_dir, "lat.nii.gz")
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    try:
        print(f"  处理角度 {angle_deg}°...")

        # 1. 旋转体积
        atten_f32 = attenuation_volume.astype(np.float32)
        rotated_atten_f32 = rotate(atten_f32, angle_deg, axes=(1, 2), reshape=False, order=1)
        rotated_attenuation = rotated_atten_f32.astype(COMPUTE_DTYPE)

        # 2. 使用改进的投影函数
        ap_projection = simulate_projection_fixed(rotated_attenuation, axis=1)
        lat_projection = simulate_projection_fixed(rotated_attenuation, axis=0)

        # 3. 缩放投影
        ap_projection_resized = resize_2d_image(ap_projection, TARGET_2D_SIZE)
        lat_projection_resized = resize_2d_image(lat_projection, TARGET_2D_SIZE)

        # 4. 生成mask投影
        dsa_f32 = original_dsa_volume.astype(np.float32)
        rotated_dsa_f32 = rotate(dsa_f32, angle_deg, axes=(1, 2), reshape=False, order=1)
        binary_mask = (rotated_dsa_f32 >= float(BINARY_THRESHOLD)).astype(np.float32)

        mask_ap_projection = np.max(binary_mask, axis=1)
        mask_lat_projection = np.max(binary_mask, axis=0)
        mask_ap_resized = resize_2d_image(mask_ap_projection, TARGET_2D_SIZE)
        mask_lat_resized = resize_2d_image(mask_lat_projection, TARGET_2D_SIZE)

        # 5. 保存文件
        save_as_nifti(ap_projection_resized, ap_path)
        save_as_nifti(lat_projection_resized, lat_path)
        save_as_nifti(binary_mask, mask_path)

        # 6. 生成可视化
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

        # 7. 生成叠加图
        if generate_overlay:
            overlay_dir = os.path.join(output_dir, "overlays")
            os.makedirs(overlay_dir, exist_ok=True)

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

        # 文件大小信息
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

    print(f"\n病例{case_num}:")
    print(f"  原始形状: {original_dsa.shape}")
    print(f"  像素范围: [{original_dsa.min():.4f}, {original_dsa.max():.4f}]")
    print(f"  非零像素比例: {np.sum(original_dsa > 0) / original_dsa.size * 100:.2f}%")

    # 检查DSA是否有效
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
        if process_single_angle(case_num, attenuation_volume, original_dsa_resized,
                                angle_idx, output_base, generate_overlay):
            success_count += 1

    return success_count, NUM_ANGLES


def test_mode():
    """测试模式"""
    print("\n" + "=" * 80)
    print("测试模式 - DSA扇形束蒙泰卡罗模拟")
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

    output_base = os.path.join(TRAIN_BASE, f"case{TEST_CASE_NUM}")
    os.makedirs(output_base, exist_ok=True)

    print(f"\n处理角度索引 {TEST_ANGLE_IDX} ({ANGLES[TEST_ANGLE_IDX]}°)...")
    success = process_single_angle(TEST_CASE_NUM, attenuation_volume, original_dsa_resized,
                                   TEST_ANGLE_IDX, output_base, generate_overlay=True)

    if success:
        output_dir = os.path.join(output_base, f"{TEST_CASE_NUM}_{TEST_ANGLE_IDX}")
        print(f"\n✓ 成功保存到: {output_dir}")

        ap_test = nib.load(os.path.join(output_dir, "ap.nii.gz")).get_fdata()
        print(f"\n验证结果:")
        print(f"  AP投影数据类型: {ap_test.dtype}, 范围: [{ap_test.min():.6f}, {ap_test.max():.6f}]")
        print(f"  AP投影非零像素: {np.sum(ap_test > 0)} / {ap_test.size}")
    else:
        print(f"\n✗ 处理失败")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def batch_mode():
    """批量模式"""
    print("\n" + "=" * 80)
    print("批量处理模式")
    print("=" * 80)
    print(f"输出目录: {TRAIN_BASE}")
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