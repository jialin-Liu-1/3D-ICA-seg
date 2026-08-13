import numpy as np
import nibabel as nib
import os
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 尝试导入CuPy进行GPU加速
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage

    USE_GPU = True
    print("GPU加速已启用 (CuPy)")
except ImportError:
    import scipy.ndimage as ndimage

    USE_GPU = False
    print("GPU加速未启用，使用CPU模式 (安装CuPy可加速: pip install cupy)")

# ============================================================
# 参数配置（在这里修改所有参数）
# ============================================================

# 基础路径
RAW_NII_BASE = r"D:\med_data\biron\data1\raw_nii"  # 原始三维图像路径
TRAIN_BASE = r"D:\med_data\biron\data1\train3"  # 输出路径

# ============================================================
# 分辨率参数
# ============================================================
TARGET_3D_SIZE = 192  # 三维mask目标尺寸 (128×128×128)
TARGET_2D_SIZE = 192  # 二维投影目标尺寸 (128×128)

# ============================================================
# 旋转参数
# ============================================================
START_ANGLE = 0  # 起始角度（度）
END_ANGLE = 180  # 结束角度（度）
STEP_ANGLE = 10  # 角度步长（度）
# 注意：旋转方向固定为绕X轴（vertical方向）

# ============================================================
# 二值化参数
# ============================================================
BINARY_THRESHOLD = 0.1  # 二值化阈值（大于等于此值设为1，小于设为0）

# ============================================================
# 重叠图像参数（仅前两个角度生成对比图）
# ============================================================
AP_ALPHA = 0.7  # AP投影不透明度
LAT_ALPHA = 0.7  # LAT投影不透明度
MASK_ALPHA = 0.3  # Mask投影不透明度

# ============================================================
# 处理选项
# ============================================================
NUM_COMPARE_IMAGES = 1  # 生成对比图的角度数量（前N个角度生成对比图）
SKIP_EXISTING = False  # 是否跳过已存在的文件
BATCH_SIZE = 4  # 批次处理大小（同时处理的病例数量）

# ============================================================
# 运行模式
# ============================================================
# "test"   : 测试单个病例的单个角度
# "batch"  : 批量处理所有病例的所有角度
RUN_MODE = "batch"  # 修改这里选择运行模式

# ============================================================
# 测试模式参数（仅当RUN_MODE="test"时生效）
# ============================================================
TEST_CASE_NUM = 0  # 测试用的病例编号
TEST_ANGLE_IDX = 10  # 测试用的旋转次数（0=0°, 10=20°, 20=40°等）

# ============================================================
# 计算派生参数
# ============================================================
ANGLES = list(range(START_ANGLE, END_ANGLE, STEP_ANGLE))
NUM_ANGLES = len(ANGLES)

print("=" * 80)
print("三维血管投影生成系统（GPU加速版 + 批次处理）")
print("=" * 80)
print(f"分辨率: 三维mask={TARGET_3D_SIZE}³, 二维投影={TARGET_2D_SIZE}²")
print(f"旋转角度范围: {START_ANGLE}° ~ {END_ANGLE - STEP_ANGLE}°, 步长{STEP_ANGLE}°, 共{NUM_ANGLES}个角度")
print(f"二值化阈值: {BINARY_THRESHOLD}")
print(f"生成对比图: 前{NUM_COMPARE_IMAGES}个角度")
print(f"批次大小: {BATCH_SIZE}")
print(f"运行模式: {RUN_MODE}")
print("=" * 80)


# ============================================================
# 核心处理函数（支持GPU）
# ============================================================

def get_ndimage_module():
    """获取适合当前设备的ndimage模块"""
    if USE_GPU:
        return cp_ndimage
    else:
        return ndimage


def to_device(data):
    """将数据转移到GPU（如果可用）"""
    if USE_GPU:
        return cp.asarray(data)
    return data


def to_cpu(data):
    """将数据从GPU转移回CPU"""
    if USE_GPU and hasattr(data, 'get'):
        return data.get()
    return data


def resize_volume(volume, target_size):
    """
    将三维体积缩放到目标尺寸

    参数:
        volume: 3D numpy数组
        target_size: 目标尺寸（整数）

    返回:
        缩放后的3D体积
    """
    current_size = volume.shape[0]
    if current_size == target_size:
        return volume

    zoom_factor = target_size / current_size
    ndimage_module = get_ndimage_module()

    volume_device = to_device(volume)
    resized = ndimage_module.zoom(volume_device, zoom_factor, order=1)

    return to_cpu(resized)


def resize_2d_image(image, target_size):
    """
    将2D图像缩放到目标尺寸

    参数:
        image: 2D numpy数组
        target_size: 目标尺寸（整数）

    返回:
        缩放后的2D图像
    """
    current_size = image.shape[0]
    if current_size == target_size:
        return image

    zoom_factor = target_size / current_size
    ndimage_module = get_ndimage_module()

    image_device = to_device(image)
    resized = ndimage_module.zoom(image_device, zoom_factor, order=1)

    return to_cpu(resized)


def binarize_volume(volume, threshold=0.1):
    """
    将三维体积二值化

    参数:
        volume: 3D numpy数组
        threshold: 阈值（大于等于此值设为1，小于设为0）

    返回:
        二值化的3D体积
    """
    binary = (volume >= threshold).astype(np.float32)
    return binary


def rotate_volume_vertical(volume, angle_deg, reshape=False):
    """
    绕X轴旋转3D体积（vertical方向）
    对应axes=(1,2)，在YZ平面旋转

    参数:
        volume: 3D numpy数组
        angle_deg: 旋转角度（度）
        reshape: 是否调整形状

    返回:
        旋转后的3D体积
    """
    ndimage_module = get_ndimage_module()

    # 将数据转移到GPU（如果需要）
    volume_device = to_device(volume)

    # 执行旋转
    rotated = ndimage_module.rotate(volume_device, angle_deg, axes=(1, 2), reshape=reshape, order=1)

    # 转回CPU
    return to_cpu(rotated)


def normalize_image(image):
    """将图像归一化到0-1范围"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-8:
        return image
    normalized = (image - min_val) / (max_val - min_val)
    return normalized


def get_projections(volume):
    """
    获取三维体积的三个正交投影

    参数:
        volume: 3D numpy数组

    返回:
        ap_projection: Y轴投影（冠状面）- 对应AP视图
        lat_projection: Z轴投影（横断面）- 对应LAT视图
        sagittal_projection: X轴投影（矢状面）- 备用
    """
    # Y轴投影（沿Y轴方向投影，得到XZ平面）- 作为AP投影
    ap_projection = np.max(volume, axis=1)

    # Z轴投影（沿Z轴方向投影，得到XY平面）- 作为LAT投影
    lat_projection = np.max(volume, axis=2)

    # X轴投影（沿X轴方向投影，得到YZ平面）- 备用
    sagittal_projection = np.max(volume, axis=0)

    return ap_projection, lat_projection, sagittal_projection


def save_as_nifti(data, output_path, description=""):
    """保存为NIfTI格式"""
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii_img = nib.Nifti1Image(data, affine)
    if description:
        nii_img.header['descrip'] = description[:80]
    nib.save(nii_img, output_path)


def save_overlay_image(image1, image2, output_path, title1="Image 1", title2="Image 2",
                       cmap1='viridis', cmap2='hot', alpha1=0.7, alpha2=0.3):
    """保存两张图像的重叠图"""
    img1_norm = normalize_image(image1)
    img2_norm = normalize_image(image2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图像1
    axes[0].imshow(img1_norm, cmap=cmap1, interpolation='nearest', origin='lower')
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=10)
    axes[0].set_ylabel('Y', fontsize=10)
    plt.colorbar(axes[0].images[0], ax=axes[0], shrink=0.8)

    # 图像2
    axes[1].imshow(img2_norm, cmap=cmap2, interpolation='nearest', origin='lower')
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X', fontsize=10)
    axes[1].set_ylabel('Y', fontsize=10)
    plt.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)

    # 重叠图
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


def save_comparison_image(ap_proj, lat_proj, mask_ap_proj, mask_lat_proj,
                          output_path, angle_deg):
    """保存综合对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行：AP相关
    axes[0, 0].imshow(ap_proj, cmap='gray', interpolation='nearest', origin='lower')
    axes[0, 0].set_title(f'AP Projection (from volume)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Z')

    axes[0, 1].imshow(mask_ap_proj, cmap='gray', interpolation='nearest', origin='lower')
    axes[0, 1].set_title(f'Mask AP Projection', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')

    axes[0, 2].imshow(ap_proj, cmap='viridis', interpolation='nearest', origin='lower', alpha=AP_ALPHA)
    axes[0, 2].imshow(mask_ap_proj, cmap='hot', interpolation='nearest', origin='lower', alpha=MASK_ALPHA)
    axes[0, 2].set_title(f'Overlay: AP + Mask AP', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Z')

    # 第二行：LAT相关
    axes[1, 0].imshow(lat_proj, cmap='gray', interpolation='nearest', origin='lower')
    axes[1, 0].set_title(f'LAT Projection (from volume)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')

    axes[1, 1].imshow(mask_lat_proj, cmap='gray', interpolation='nearest', origin='lower')
    axes[1, 1].set_title(f'Mask LAT Projection', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')

    axes[1, 2].imshow(lat_proj, cmap='viridis', interpolation='nearest', origin='lower', alpha=LAT_ALPHA)
    axes[1, 2].imshow(mask_lat_proj, cmap='hot', interpolation='nearest', origin='lower', alpha=MASK_ALPHA)
    axes[1, 2].set_title(f'Overlay: LAT + Mask LAT', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')

    plt.suptitle(f'Angle: {angle_deg}° (Rotated around X-axis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_angle(case_num, original_volume_resized, angle_idx, output_base, generate_compare=False):
    """
    处理单个角度的数据

    参数:
        case_num: 病例编号
        original_volume_resized: 已缩放的原始三维体积
        angle_idx: 角度索引（0,1,2,...）
        output_base: 输出基路径
        generate_compare: 是否生成对比图

    返回:
        成功标志
    """
    angle_deg = ANGLES[angle_idx]

    # 创建输出文件夹
    output_dir = os.path.join(output_base, f"{case_num}_{angle_idx}")
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否已存在
    ap_path = os.path.join(output_dir, "ap.nii.gz")
    lat_path = os.path.join(output_dir, "lat.nii.gz")
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    if SKIP_EXISTING and all([os.path.exists(ap_path), os.path.exists(lat_path), os.path.exists(mask_path)]):
        return True

    try:
        # 1. 旋转体积
        rotated_volume = rotate_volume_vertical(original_volume_resized, angle_deg, reshape=False)

        # 2. 获取旋转后体积的投影
        ap_projection, lat_projection, _ = get_projections(rotated_volume)

        # 3. 缩放投影到目标尺寸
        ap_projection_resized = resize_2d_image(ap_projection, TARGET_2D_SIZE)
        lat_projection_resized = resize_2d_image(lat_projection, TARGET_2D_SIZE)

        # 4. 对旋转后的体积进行二值化
        binary_mask = binarize_volume(rotated_volume, threshold=BINARY_THRESHOLD)

        # 5. 保存NIfTI文件
        save_as_nifti(ap_projection_resized, ap_path, description=f"AP projection at {angle_deg} degrees")
        save_as_nifti(lat_projection_resized, lat_path, description=f"LAT projection at {angle_deg} degrees")
        save_as_nifti(binary_mask, mask_path, description=f"Binary mask at {angle_deg} degrees")

        # 6. 仅当需要时保存重叠对比图
        if generate_compare:
            compare_dir = os.path.join(output_dir, "compare")
            os.makedirs(compare_dir, exist_ok=True)

            # 获取二值化mask的投影
            mask_ap_projection, mask_lat_projection, _ = get_projections(binary_mask)

            # 缩放mask投影
            mask_ap_projection_resized = resize_2d_image(mask_ap_projection, TARGET_2D_SIZE)
            mask_lat_projection_resized = resize_2d_image(mask_lat_projection, TARGET_2D_SIZE)

            # 保存AP重叠图
            overlay_ap_path = os.path.join(compare_dir, "overlay_AP.png")
            save_overlay_image(ap_projection_resized, mask_ap_projection_resized, overlay_ap_path,
                               title1="AP Projection", title2="Mask AP Projection",
                               alpha1=AP_ALPHA, alpha2=MASK_ALPHA)

            # 保存LAT重叠图
            overlay_lat_path = os.path.join(compare_dir, "overlay_LAT.png")
            save_overlay_image(lat_projection_resized, mask_lat_projection_resized, overlay_lat_path,
                               title1="LAT Projection", title2="Mask LAT Projection",
                               alpha1=LAT_ALPHA, alpha2=MASK_ALPHA)

            # 保存综合对比图
            comparison_path = os.path.join(compare_dir, f"comparison_{angle_deg}deg.png")
            save_comparison_image(ap_projection_resized, lat_projection_resized,
                                  mask_ap_projection_resized, mask_lat_projection_resized,
                                  comparison_path, angle_deg)

        return True

    except Exception as e:
        print(f"  ✗ 处理失败 (idx={angle_idx}, angle={angle_deg}°): {str(e)}")
        return False


def process_single_case_all_angles(case_num, output_base):
    """
    处理单个病例的所有角度

    参数:
        case_num: 病例编号
        output_base: 输出基路径

    返回:
        (成功数量, 总角度数)
    """
    # 加载原始三维图像
    input_file = os.path.join(RAW_NII_BASE, f"{case_num}.nii.gz")
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return 0, 0

    nii_img = nib.load(input_file)
    original_volume = nii_img.get_fdata().astype(np.float32)

    # 缩放三维体积到目标尺寸
    original_volume_resized = resize_volume(original_volume, TARGET_3D_SIZE)

    print(f"\n病例{case_num}: 原始形状 {original_volume.shape} -> 缩放后 {original_volume_resized.shape}")

    # 处理每个角度
    success_count = 0
    for angle_idx in range(NUM_ANGLES):
        # 只有前NUM_COMPARE_IMAGES个角度生成对比图
        generate_compare = (angle_idx < NUM_COMPARE_IMAGES)

        if process_single_angle(case_num, original_volume_resized, angle_idx, output_base, generate_compare):
            success_count += 1

    return success_count, NUM_ANGLES


def process_case_batch(case_list, output_base):
    """
    批次处理多个病例

    参数:
        case_list: 病例编号列表
        output_base: 输出基路径

    返回:
        (成功总数, 总角度数)
    """
    total_success = 0
    total_angles = 0

    for case_num in case_list:
        success, total = process_single_case_all_angles(case_num, output_base)
        total_success += success
        total_angles += total
        print(f"  病例{case_num}: 成功 {success}/{total}")

    return total_success, total_angles


def find_all_cases():
    """查找所有可用的病例"""
    nii_files = glob.glob(os.path.join(RAW_NII_BASE, "*.nii.gz"))
    cases = []
    for f in nii_files:
        match = re.search(r"(\d+)\.nii\.gz$", os.path.basename(f))
        if match:
            cases.append(int(match.group(1)))
    return sorted(cases)


def test_mode():
    """测试模式：处理单个病例的单个角度"""
    print("\n" + "=" * 80)
    print("测试模式")
    print("=" * 80)
    print(f"测试病例: {TEST_CASE_NUM}")
    print(f"测试角度索引: {TEST_ANGLE_IDX} (对应角度: {ANGLES[TEST_ANGLE_IDX]}°)")
    print(f"目标分辨率: 三维={TARGET_3D_SIZE}³, 二维={TARGET_2D_SIZE}²")
    print("=" * 80)

    # 加载原始三维图像
    input_file = os.path.join(RAW_NII_BASE, f"{TEST_CASE_NUM}.nii.gz")
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return

    nii_img = nib.load(input_file)
    original_volume = nii_img.get_fdata().astype(np.float32)
    print(f"\n原始形状: {original_volume.shape}")

    # 缩放三维体积
    original_volume_resized = resize_volume(original_volume, TARGET_3D_SIZE)
    print(f"缩放后形状: {original_volume_resized.shape}")

    # 创建输出目录
    output_base = os.path.join(TRAIN_BASE, f"test_case{TEST_CASE_NUM}")
    os.makedirs(output_base, exist_ok=True)

    # 处理指定角度（测试模式总是生成对比图）
    print(f"\n处理角度索引 {TEST_ANGLE_IDX} ({ANGLES[TEST_ANGLE_IDX]}°)...")
    success = process_single_angle(TEST_CASE_NUM, original_volume_resized, TEST_ANGLE_IDX, output_base,
                                   generate_compare=True)

    if success:
        output_dir = os.path.join(output_base, f"{TEST_CASE_NUM}_{TEST_ANGLE_IDX}")
        print(f"\n✓ 成功保存到: {output_dir}")
        print(f"  文件列表:")
        print(f"    - ap.nii.gz (AP投影, {TARGET_2D_SIZE}×{TARGET_2D_SIZE})")
        print(f"    - lat.nii.gz (LAT投影, {TARGET_2D_SIZE}×{TARGET_2D_SIZE})")
        print(f"    - mask.nii.gz (二值化mask, {TARGET_3D_SIZE}³)")
        print(f"    - compare/ (重叠对比图)")
    else:
        print(f"\n✗ 处理失败")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def batch_mode():
    """批量模式：批次处理所有病例的所有角度"""
    print("\n" + "=" * 80)
    print("批量处理模式（批次处理）")
    print("=" * 80)
    print(f"目标分辨率: 三维={TARGET_3D_SIZE}³, 二维={TARGET_2D_SIZE}²")
    print(f"旋转角度: {START_ANGLE}° ~ {END_ANGLE - STEP_ANGLE}°, 步长{STEP_ANGLE}°, 共{NUM_ANGLES}个角度")
    print(f"生成对比图: 前{NUM_COMPARE_IMAGES}个角度")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"输出目录: {TRAIN_BASE}")
    if USE_GPU:
        print("GPU加速: 已启用")
    else:
        print("GPU加速: 未启用 (安装CuPy可加速: pip install cupy)")
    print("=" * 80)

    # 查找所有病例
    available_cases = find_all_cases()
    if len(available_cases) == 0:
        print("错误：没有找到可用的病例")
        return

    print(f"\n找到 {len(available_cases)} 个病例")

    # 创建输出基目录
    os.makedirs(TRAIN_BASE, exist_ok=True)

    # 将病例分成批次
    case_batches = [available_cases[i:i + BATCH_SIZE] for i in range(0, len(available_cases), BATCH_SIZE)]

    print(f"分为 {len(case_batches)} 个批次进行处理\n")

    # 统计信息
    total_success = 0
    total_angles = 0

    # 使用进度条处理批次
    with tqdm(total=len(available_cases), desc="处理病例") as pbar:
        for batch_idx, case_batch in enumerate(case_batches):
            print(f"\n处理批次 {batch_idx + 1}/{len(case_batches)} (病例: {case_batch})")

            # 处理当前批次
            success, total = process_case_batch(case_batch, TRAIN_BASE)
            total_success += success
            total_angles += total

            pbar.update(len(case_batch))

    print("\n" + "=" * 80)
    print("批量处理完成！")
    print("=" * 80)
    print(f"处理病例数: {len(available_cases)}")
    print(f"总角度数: {total_angles}")
    print(f"成功处理: {total_success}")
    print(f"输出目录: {TRAIN_BASE}")
    print("=" * 80)


def main():
    """主函数"""
    if RUN_MODE == "test":
        test_mode()
    elif RUN_MODE == "batch":
        batch_mode()
    else:
        print(f"错误: 未知的运行模式 '{RUN_MODE}'")
        print("请设置 RUN_MODE = 'test' 或 'batch'")


if __name__ == "__main__":
    main()