import numpy as np
import nibabel as nib
import pydicom
import os
import glob
import re
from tqdm import tqdm
import scipy.ndimage as ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 参数配置
# ============================================================
RAW_DCM_BASE = r"D:\med_data\biron\data1\raw_dcm"
MASK_BASE = r"D:\med_data\biron\data1\mask"
TRAIN_BASE = r"D:\med_data\biron\data1\train"

# 角度参数
ANGLE_MULTIPLIER = 2  # 投影编号乘以2得到角度（度）
ORTHOGONAL_ANGLE_DIFF = 90  # 正交投影的角度差（度）
PROJECTION_OFFSET = ORTHOGONAL_ANGLE_DIFF // ANGLE_MULTIPLIER  # 45

# 图像尺寸参数
ORIGINAL_SIZE = 512
TARGET_SIZE = 256

# 处理选项
SKIP_EXISTING = True

# 投影参数
SAVE_PROJECTIONS = True  # 是否保存三个轴的投影
PROJECTION_DIR = "projections"  # 投影保存的子文件夹名

print(f"角度倍数: 投影编号 × {ANGLE_MULTIPLIER} = 角度")
print(f"正交角度差: {ORTHOGONAL_ANGLE_DIFF}°")
print(f"正交编号偏移: {PROJECTION_OFFSET} (编号 + {PROJECTION_OFFSET})")


# ============================================================
# 投影和可视化函数
# ============================================================

def create_custom_colormap():
    """创建自定义colormap用于二值图像显示"""
    colors = ['black', 'white']
    cmap = LinearSegmentedColormap.from_list('custom_binary', colors, N=2)
    return cmap


def normalize_image(image):
    """
    将图像归一化到0-1范围（基于图像自身的最大值和最小值）
    """
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val < 1e-8:
        print(f"    警告: 图像像素值范围过小，跳过归一化")
        return image

    normalized = (image - min_val) / (max_val - min_val)
    return normalized


def save_3d_projections(volume, output_dir, prefix="mask", axis_names=['X', 'Y', 'Z']):
    """保存3D体积的三个轴的最大强度投影"""
    os.makedirs(output_dir, exist_ok=True)
    cmap = create_custom_colormap()

    for axis_idx, axis_name in enumerate(axis_names):
        if axis_idx == 0:  # X轴，投影到YZ平面
            projection = np.max(volume, axis=axis_idx)
            title = f"X-axis Projection (Sagittal View)"
            xlabel = "Y"
            ylabel = "Z"
        elif axis_idx == 1:  # Y轴，投影到XZ平面
            projection = np.max(volume, axis=axis_idx)
            title = f"Y-axis Projection (Coronal View)"
            xlabel = "X"
            ylabel = "Z"
        else:  # Z轴，投影到XY平面
            projection = np.max(volume, axis=axis_idx)
            title = f"Z-axis Projection (Axial View)"
            xlabel = "X"
            ylabel = "Y"

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(projection, cmap=cmap, interpolation='nearest', origin='lower')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Intensity', fontsize=10)
        ax.grid(False)

        output_path = os.path.join(output_dir, f"{prefix}_projection_{axis_name}axis.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_2d_image_as_png(image, output_path, title="2D Projection", cmap='viridis', normalize=True):
    """
    保存2D图像为PNG格式

    参数:
        image: 2D numpy数组
        output_path: 输出路径
        title: 图像标题
        cmap: colormap
        normalize: 是否归一化显示
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if normalize:
        # 归一化显示
        img_display = normalize_image(image)
    else:
        img_display = image

    im = ax.imshow(img_display, cmap=cmap, interpolation='nearest', origin='lower')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (columns)', fontsize=12)
    ax.set_ylabel('Y (rows)', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Intensity', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_overlay_image(image1, image2, output_path, title1="Image 1", title2="Image 2", cmap1='viridis', cmap2='hot',
                       alpha=0.5):
    """
    保存两张图像的重叠图

    参数:
        image1: 第一张图像（背景）
        image2: 第二张图像（前景）
        output_path: 输出路径
        title1: 第一张图像标题
        title2: 第二张图像标题
        cmap1: 第一张图像的colormap
        cmap2: 第二张图像的colormap
        alpha: 透明度
    """
    # 归一化两张图像
    img1_norm = normalize_image(image1)
    img2_norm = normalize_image(image2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图像1
    im1 = axes[0].imshow(img1_norm, cmap=cmap1, interpolation='nearest', origin='lower')
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=10)
    axes[0].set_ylabel('Y', fontsize=10)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # 图像2
    im2 = axes[1].imshow(img2_norm, cmap=cmap2, interpolation='nearest', origin='lower')
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X', fontsize=10)
    axes[1].set_ylabel('Y', fontsize=10)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # 重叠图
    # 创建RGB图像
    overlay = np.zeros((*img1_norm.shape, 3))

    # 使用colormap转换
    cmap1_func = plt.cm.get_cmap(cmap1)
    cmap2_func = plt.cm.get_cmap(cmap2)

    # 将灰度图像映射到RGB
    img1_rgb = cmap1_func(img1_norm)[:, :, :3]
    img2_rgb = cmap2_func(img2_norm)[:, :, :3]

    # 混合
    overlay = img1_rgb * (1 - alpha) + img2_rgb * alpha

    axes[2].imshow(overlay, interpolation='nearest', origin='lower')
    axes[2].set_title(f'Overlay: {title1} + {title2}', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X', fontsize=10)
    axes[2].set_ylabel('Y', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_with_overlay(original, transformed, output_path, title="Comparison"):
    """保存原始和变换后的对比图像"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    img1_norm = normalize_image(original)
    img2_norm = normalize_image(transformed)

    im1 = axes[0].imshow(img1_norm, cmap='gray', interpolation='nearest', origin='lower')
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(img2_norm, cmap='gray', interpolation='nearest', origin='lower')
    axes[1].set_title('Transformed', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_multi_slice_visualization(volume, output_path, num_slices=9, axis=2):
    """保存3D体积的多切片可视化"""
    if axis == 2:  # Z轴（横断面）
        slices = np.linspace(0, volume.shape[2] - 1, num_slices, dtype=int)
        title = "Axial Slices (Z-axis)"
    elif axis == 1:  # Y轴（冠状面）
        slices = np.linspace(0, volume.shape[1] - 1, num_slices, dtype=int)
        title = "Coronal Slices (Y-axis)"
    else:  # X轴（矢状面）
        slices = np.linspace(0, volume.shape[0] - 1, num_slices, dtype=int)
        title = "Sagittal Slices (X-axis)"

    grid_size = int(np.ceil(np.sqrt(num_slices)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slices):
        if idx < len(axes):
            if axis == 2:
                slice_data = volume[:, :, slice_idx]
            elif axis == 1:
                slice_data = volume[:, slice_idx, :]
            else:
                slice_data = volume[slice_idx, :, :]

            axes[idx].imshow(slice_data, cmap='gray', interpolation='nearest', origin='lower')
            axes[idx].set_title(f'Slice {slice_idx}', fontsize=8)
            axes[idx].axis('off')

    for idx in range(len(slices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 核心处理函数
# ============================================================

def extract_projection_number(dicom_path):
    """从DICOM文件路径提取投影编号"""
    filename = os.path.basename(dicom_path)
    match = re.search(r"dat_(\d+)\.dcm$", filename)
    if match:
        return int(match.group(1))
    return None


def read_dicom_projection(dicom_path):
    """读取DICOM投影图像"""
    try:
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array
        if image.dtype == np.uint16:
            image = image.astype(np.float32)
        return image
    except Exception as e:
        print(f"  读取DICOM失败 {dicom_path}: {str(e)}")
        return None


def resize_image(image, target_size=256):
    """将图像尺寸减半"""
    current_size = image.shape[0]
    if current_size == target_size:
        return image
    zoom_factor = target_size / current_size
    resized = ndimage.zoom(image, zoom_factor, order=1)
    return resized


def flip_image_horizontal(image):
    """水平翻转图像（左右翻转）"""
    return np.fliplr(image)


def rotate_3d_volume(volume, angle_deg, axis='horizontal'):
    """旋转3D体积数据"""
    if axis == 'horizontal':
        rotated = ndimage.rotate(volume, angle_deg, axes=(0, 2), reshape=False, order=1)
    elif axis == 'vertical':
        rotated = ndimage.rotate(volume, angle_deg, axes=(1, 2), reshape=False, order=1)
    else:
        rotated = volume
    return rotated


def save_as_nifti(data, output_path, affine=None, description=""):
    """保存为NIfTI格式"""
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii_img = nib.Nifti1Image(data, affine)
    nii_img.header['descrip'] = description[:80]
    nib.save(nii_img, output_path)


def find_orthogonal_pair(projection_files, base_number):
    """查找正交配对的投影文件"""
    current_file = None
    for f in projection_files:
        num = extract_projection_number(f)
        if num == base_number:
            current_file = f
            break

    if current_file is None:
        return None, None

    orthogonal_number = base_number + PROJECTION_OFFSET
    orthogonal_file = None
    for f in projection_files:
        num = extract_projection_number(f)
        if num == orthogonal_number:
            orthogonal_file = f
            break

    return current_file, orthogonal_file


def collect_projection_pairs(case_folder):
    """收集一个病例文件夹中的所有投影配对"""
    dicom_files = glob.glob(os.path.join(case_folder, "*.dcm"))
    if len(dicom_files) == 0:
        return []

    numbers_with_files = []
    for f in dicom_files:
        num = extract_projection_number(f)
        if num is not None:
            numbers_with_files.append((num, f))

    numbers_with_files.sort(key=lambda x: x[0])
    projection_files = [f for num, f in numbers_with_files]
    all_numbers = [num for num, f in numbers_with_files]

    projection_pairs = []
    processed_numbers = set()

    for num in all_numbers:
        if num in processed_numbers:
            continue

        current_file, ortho_file = find_orthogonal_pair(projection_files, num)

        if current_file and ortho_file:
            projection_pairs.append((num, current_file, ortho_file))
            processed_numbers.add(num)
            processed_numbers.add(num + PROJECTION_OFFSET)

    return projection_pairs


def process_case(case_num, projection_pairs, mask_volume, output_base, skip_existing=True):
    """处理单个病例的所有投影配对"""
    success_count = 0
    fail_count = 0

    for proj_num, current_path, ortho_path in projection_pairs:
        # 创建输出子文件夹
        output_dir = os.path.join(output_base, f"{case_num}_{proj_num}")
        os.makedirs(output_dir, exist_ok=True)

        projection_dir = os.path.join(output_dir, PROJECTION_DIR)
        os.makedirs(projection_dir, exist_ok=True)

        current_angle = proj_num * ANGLE_MULTIPLIER
        ortho_angle = (proj_num + PROJECTION_OFFSET) * ANGLE_MULTIPLIER

        # 确定哪个角度小（LAT），哪个角度大（AP）
        if current_angle < ortho_angle:
            lat_angle = current_angle
            ap_angle = ortho_angle
            lat_path = current_path
            ap_path = ortho_path
            lat_num = proj_num
            ap_num = proj_num + PROJECTION_OFFSET
        else:
            lat_angle = ortho_angle
            ap_angle = current_angle
            lat_path = ortho_path
            ap_path = current_path
            lat_num = proj_num + PROJECTION_OFFSET
            ap_num = proj_num

        current_output = os.path.join(output_dir, f"lat.nii.gz")
        ortho_output = os.path.join(output_dir, f"ap.nii.gz")
        mask_output = os.path.join(output_dir, f"mask.nii.gz")

        if skip_existing and all([os.path.exists(current_output),
                                  os.path.exists(ortho_output),
                                  os.path.exists(mask_output)]):
            print(f"  跳过已存在的配对: 病例{case_num}_投影{proj_num}")
            success_count += 1
            continue

        try:
            # 1. 读取两个投影图像
            lat_img = read_dicom_projection(lat_path)
            ap_img = read_dicom_projection(ap_path)

            if lat_img is None or ap_img is None:
                print(f"  读取投影失败: 病例{case_num}_投影{proj_num}")
                fail_count += 1
                continue

            # 2. 将投影图像尺寸减半
            lat_resized = resize_image(lat_img, TARGET_SIZE)
            ap_resized = resize_image(ap_img, TARGET_SIZE)

            # 3. 归一化处理
            lat_normalized = normalize_image(lat_resized)
            ap_normalized = normalize_image(ap_resized)

            # 4. 对LAT图像进行左右翻转
            lat_flipped = flip_image_horizontal(lat_normalized)

            print(f"\n  处理: 病例{case_num}")
            print(f"    LAT: 编号{lat_num} = {lat_angle}° (较小角度)")
            print(f"    AP:  编号{ap_num} = {ap_angle}° (较大角度)")
            print(f"    角度差: {ap_angle - lat_angle}°")
            print(f"    已对LAT图像进行左右翻转")

            # 5. 旋转3D mask（使用当前投影的角度）
            rotated_mask = rotate_3d_volume(mask_volume, current_angle, axis='horizontal')

            # 6. 获取旋转后mask的三个轴投影
            mask_z_projection = np.max(rotated_mask, axis=2)  # Z轴投影（横断面）
            mask_y_projection = np.max(rotated_mask, axis=1)  # Y轴投影（冠状面）
            mask_x_projection = np.max(rotated_mask, axis=0)  # X轴投影（矢状面）

            # 7. 生成重叠图像用于验证
            print(f"  生成验证重叠图像...")

            # LAT翻转图与Z轴投影重叠
            overlay_lat_z = os.path.join(projection_dir, f"overlay_LAT_flipped_vs_Zproj.png")
            save_overlay_image(lat_flipped, mask_z_projection, overlay_lat_z,
                               title1=f"LAT Flipped (at {lat_angle}°)",
                               title2=f"Mask Z-axis Projection",
                               cmap1='viridis', cmap2='hot', alpha=0.5)

            # AP图与Y轴投影重叠
            overlay_ap_y = os.path.join(projection_dir, f"overlay_AP_vs_Yproj.png")
            save_overlay_image(ap_normalized, mask_y_projection, overlay_ap_y,
                               title1=f"AP Projection (at {ap_angle}°)",
                               title2=f"Mask Y-axis Projection",
                               cmap1='viridis', cmap2='hot', alpha=0.5)

            # 8. 保存所有投影和可视化图像
            if SAVE_PROJECTIONS:
                print(f"  生成三个轴的投影图像...")
                save_3d_projections(rotated_mask, projection_dir,
                                    prefix=f"rotated_mask_{current_angle}deg",
                                    axis_names=['X', 'Y', 'Z'])

                save_multi_slice_visualization(rotated_mask,
                                               os.path.join(projection_dir, f"multi_slices_Zaxis.png"),
                                               axis=2, num_slices=9)

                print(f"  保存2D投影图像...")
                # 保存原始LAT和AP
                save_2d_image_as_png(lat_resized,
                                     os.path.join(projection_dir, f"LAT_{lat_num}_{lat_angle}deg_original.png"),
                                     title=f"LAT Original at {lat_angle} degrees")

                save_2d_image_as_png(lat_normalized,
                                     os.path.join(projection_dir, f"LAT_{lat_num}_{lat_angle}deg_normalized.png"),
                                     title=f"LAT Normalized at {lat_angle} degrees")

                save_2d_image_as_png(lat_flipped,
                                     os.path.join(projection_dir, f"LAT_{lat_num}_{lat_angle}deg_flipped.png"),
                                     title=f"LAT Flipped at {lat_angle} degrees")

                save_2d_image_as_png(ap_resized,
                                     os.path.join(projection_dir, f"AP_{ap_num}_{ap_angle}deg_original.png"),
                                     title=f"AP Original at {ap_angle} degrees")

                save_2d_image_as_png(ap_normalized,
                                     os.path.join(projection_dir, f"AP_{ap_num}_{ap_angle}deg_normalized.png"),
                                     title=f"AP Normalized at {ap_angle} degrees")

                # 保存mask的三个轴投影
                save_2d_image_as_png(mask_z_projection,
                                     os.path.join(projection_dir, f"mask_Zaxis_projection.png"),
                                     title=f"Mask Z-axis Projection", cmap='gray')

                save_2d_image_as_png(mask_y_projection,
                                     os.path.join(projection_dir, f"mask_Yaxis_projection.png"),
                                     title=f"Mask Y-axis Projection", cmap='gray')

                save_2d_image_as_png(mask_x_projection,
                                     os.path.join(projection_dir, f"mask_Xaxis_projection.png"),
                                     title=f"Mask X-axis Projection", cmap='gray')

                # 保存对比图
                save_comparison_with_overlay(lat_resized, lat_flipped,
                                             os.path.join(projection_dir, f"LAT_comparison.png"),
                                             title=f"LAT Original vs Flipped (at {lat_angle}°)")

            # 9. 保存NIfTI文件
            print(f"  保存NIfTI文件...")
            save_as_nifti(lat_normalized, current_output if current_angle == lat_angle else ortho_output,
                          description=f"Projection at {lat_angle} degrees")
            save_as_nifti(ap_normalized, ortho_output if ortho_angle == ap_angle else current_output,
                          description=f"Projection at {ap_angle} degrees")
            save_as_nifti(rotated_mask, mask_output,
                          description=f"Rotated mask at {current_angle} degrees")

            print(f"    ✓ 成功保存到: {output_dir}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ 处理失败 病例{case_num}_投影{proj_num}: {str(e)}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

    return success_count, fail_count


def find_all_cases():
    """查找所有可用的病例"""
    raw_dcm_cases = []
    if os.path.exists(RAW_DCM_BASE):
        for item in os.listdir(RAW_DCM_BASE):
            item_path = os.path.join(RAW_DCM_BASE, item)
            if os.path.isdir(item_path) and item.isdigit():
                raw_dcm_cases.append(int(item))

    mask_files = glob.glob(os.path.join(MASK_BASE, "*_mask.nii.gz"))
    mask_cases = []
    for f in mask_files:
        match = re.search(r"(\d+)_mask\.nii\.gz$", os.path.basename(f))
        if match:
            mask_cases.append(int(match.group(1)))

    common_cases = sorted(set(raw_dcm_cases) & set(mask_cases))
    return common_cases


def batch_process():
    """批量处理所有病例"""
    print("=" * 80)
    print("投影与Mask配对处理系统（带归一化、翻转和重叠验证）")
    print("=" * 80)
    print(f"DICOM投影基路径: {RAW_DCM_BASE}")
    print(f"Mask基路径: {MASK_BASE}")
    print(f"输出基路径: {TRAIN_BASE}")
    print(f"正交配对: 编号相差 {PROJECTION_OFFSET} → 角度相差 {PROJECTION_OFFSET * ANGLE_MULTIPLIER}°")
    print(f"处理内容:")
    print(f"  - 图像归一化 (0-1范围)")
    print(f"  - LAT图像左右翻转")
    print(f"  - LAT翻转图 vs Mask Z轴投影重叠")
    print(f"  - AP图 vs Mask Y轴投影重叠")
    print(f"图像尺寸: {ORIGINAL_SIZE} → {TARGET_SIZE}")
    print(f"跳过已存在文件: {SKIP_EXISTING}")
    print("=" * 80)

    available_cases = find_all_cases()
    if len(available_cases) == 0:
        print("错误：没有找到可用的病例")
        return

    print(f"\n找到 {len(available_cases)} 个可用病例")
    os.makedirs(TRAIN_BASE, exist_ok=True)

    total_pairs = 0
    total_success = 0
    total_fail = 0
    cases_with_pairs = 0

    print("\n开始批量处理...\n")

    for case_num in tqdm(available_cases, desc="处理病例"):
        raw_dcm_case_folder = os.path.join(RAW_DCM_BASE, str(case_num))
        mask_file = os.path.join(MASK_BASE, f"{case_num}_mask.nii.gz")

        if not os.path.exists(mask_file):
            print(f"警告: 病例{case_num}的mask文件不存在，跳过")
            continue

        try:
            mask_nii = nib.load(mask_file)
            mask_volume = mask_nii.get_fdata().astype(np.float32)
            print(f"\n病例{case_num}: Mask形状 {mask_volume.shape}")

            projection_pairs = collect_projection_pairs(raw_dcm_case_folder)
            if len(projection_pairs) == 0:
                print(f"病例{case_num}: 没有找到有效的投影配对")
                continue

            print(f"病例{case_num}: 找到 {len(projection_pairs)} 个投影配对")
            success, fail = process_case(case_num, projection_pairs, mask_volume,
                                         TRAIN_BASE, SKIP_EXISTING)

            total_pairs += len(projection_pairs)
            total_success += success
            total_fail += fail
            if success > 0:
                cases_with_pairs += 1

        except Exception as e:
            print(f"病例{case_num}处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("批量处理完成！")
    print("=" * 80)
    print(f"处理病例数: {len(available_cases)}")
    print(f"有配对的病例数: {cases_with_pairs}")
    print(f"总投影配对数: {total_pairs}")
    print(f"成功处理配对: {total_success}")
    print(f"失败配对: {total_fail}")
    print(f"输出目录: {TRAIN_BASE}")
    print("=" * 80)


def test_single_case():
    """测试单个病例的处理"""
    test_case_num = 0
    test_proj_num = 0

    print("测试模式：处理单个病例")
    print(f"测试病例: {test_case_num}")

    raw_dcm_case_folder = os.path.join(RAW_DCM_BASE, str(test_case_num))
    mask_file = os.path.join(MASK_BASE, f"{test_case_num}_mask.nii.gz")
    output_base = os.path.join(TRAIN_BASE, "test")

    if not os.path.exists(raw_dcm_case_folder):
        print(f"错误: 病例文件夹不存在 {raw_dcm_case_folder}")
        return
    if not os.path.exists(mask_file):
        print(f"错误: Mask文件不存在 {mask_file}")
        return

    mask_nii = nib.load(mask_file)
    mask_volume = mask_nii.get_fdata().astype(np.float32)
    print(f"Mask形状: {mask_volume.shape}")

    projection_pairs = collect_projection_pairs(raw_dcm_case_folder)
    print(f"找到投影配对: {len(projection_pairs)}")

    target_pair = None
    for pair in projection_pairs:
        if pair[0] == test_proj_num:
            target_pair = pair
            break

    if target_pair is None:
        print(f"未找到投影编号 {test_proj_num} 的配对")
        return

    success, fail = process_case(test_case_num, [target_pair], mask_volume, output_base, skip_existing=False)
    print(f"\n测试结果: 成功={success}, 失败={fail}")


def analyze_projection_distribution(case_num=0):
    """分析特定病例的投影分布"""
    raw_dcm_case_folder = os.path.join(RAW_DCM_BASE, str(case_num))
    if not os.path.exists(raw_dcm_case_folder):
        print(f"病例文件夹不存在: {raw_dcm_case_folder}")
        return

    dicom_files = glob.glob(os.path.join(raw_dcm_case_folder, "*.dcm"))
    numbers = []
    for f in dicom_files:
        num = extract_projection_number(f)
        if num is not None:
            numbers.append(num)
    numbers.sort()

    print(f"\n病例{case_num} 投影分析:")
    print(f"总投影数: {len(numbers)}")
    print(f"投影编号范围: {numbers[0]} - {numbers[-1]}")

    pairs = []
    for num in numbers:
        if num + PROJECTION_OFFSET in numbers:
            pairs.append(num)

    print(f"可能的正交配对数: {len(pairs)}")
    if len(pairs) > 0:
        print("\n前10个正交配对:")
        for num in pairs[:10]:
            angle1 = num * ANGLE_MULTIPLIER
            angle2 = (num + PROJECTION_OFFSET) * ANGLE_MULTIPLIER
            print(f"  投影{num} ({angle1}°) <-> 投影{num + PROJECTION_OFFSET} ({angle2}°)")


if __name__ == "__main__":
    RUN_MODE = "batch"  # "batch" 批量处理, "test" 测试单个病例, "analyze" 分析投影分布

    if RUN_MODE == "batch":
        batch_process()
    elif RUN_MODE == "test":
        test_single_case()
    elif RUN_MODE == "analyze":
        analyze_projection_distribution(case_num=0)
    else:
        print("请设置正确的运行模式: 'batch', 'test', 或 'analyze'")