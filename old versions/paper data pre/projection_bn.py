import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import shutil
from scipy.ndimage import rotate


def generate_projection(ct_data, direction='ap'):
    """
    生成投影图像

    参数:
    - ct_data: 3D CT数据，形状为 (X, Y, Z)
      X: 左右方向（左→右）
      Y: 前后方向（前→后）
      Z: 上下方向（上→下）
    - direction: 'ap' 正面投影，'lat' 侧面投影
    """
    if direction == 'ap':
        # 正面投影：沿Y轴（前后）求和
        projection = np.sum(ct_data, axis=1)  # (X, Z)
        projection = projection.T  # 转置为 (Z, X)

    elif direction == 'lat':
        # 侧面投影：沿X轴（左右）求和
        projection = np.sum(ct_data, axis=0)  # (Y, Z)
        projection = projection.T  # (Z, Y)

    else:
        raise ValueError("direction must be 'ap' or 'lat'")

    return projection


def apply_log_transform(projection, epsilon=1e-6):
    """应用对数变换模拟X射线的衰减特性"""
    projection_positive = np.maximum(projection, epsilon)
    log_projection = -np.log(projection_positive / projection_positive.max())
    return log_projection


def normalize_projection(projection, lower_percentile=2, upper_percentile=98):
    """对投影图像进行归一化处理"""
    p_low = np.percentile(projection, lower_percentile)
    p_high = np.percentile(projection, upper_percentile)
    projection_clipped = np.clip(projection, p_low, p_high)

    p_min = projection_clipped.min()
    p_max = projection_clipped.max()

    if p_max - p_min > 0:
        normalized = (projection_clipped - p_min) / (p_max - p_min)
    else:
        normalized = np.zeros_like(projection, dtype=np.float32)

    return normalized.astype(np.float32)


def enhance_contrast(image, gamma=0.7):
    """使用伽马校正增强图像对比度"""
    return np.power(image, gamma)


def process_patient(patient_dir, source_dir, target_dir):
    """处理单个病例：生成AP和LAT投影，复制mask"""

    patient_name = os.path.basename(patient_dir)

    ct_path = os.path.join(patient_dir, "ct.nii.gz")
    mask_path = os.path.join(patient_dir, "mask.nii.gz")

    if not os.path.exists(ct_path):
        print(f"跳过 {patient_name}: 缺少CT文件")
        return False

    # 加载CT数据
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata().astype(np.float32)

    # 将HU值转换为衰减系数
    ct_normalized = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min() + 1e-8)
    ct_attenuation = np.exp(-ct_normalized * 5)

    # 生成AP投影（正面）
    ap_projection = generate_projection(ct_attenuation, direction='ap')
    ap_log = apply_log_transform(ap_projection)
    ap_normalized = normalize_projection(ap_log)
    # 顺时针旋转90度使头部向上
    ap_rotated = rotate(ap_normalized, 90, reshape=True, order=1)
    # 使用flipud修正左右翻转问题（因为图像经过转置后，flipud等同于左右翻转）
    ap_flipped = np.flipud(ap_rotated)
    ap_enhanced = enhance_contrast(ap_flipped, gamma=0.7)

    # 生成LAT投影（侧面）
    lat_projection = generate_projection(ct_attenuation, direction='lat')
    lat_log = apply_log_transform(lat_projection)
    lat_normalized = normalize_projection(lat_log)
    # 顺时针旋转90度使头部向上
    lat_rotated = rotate(lat_normalized, 90, reshape=True, order=1)
    # 使用flipud修正左右翻转问题
    lat_flipped = np.flipud(lat_rotated)
    lat_enhanced = enhance_contrast(lat_flipped, gamma=0.7)

    # 创建目标子文件夹
    target_patient_dir = os.path.join(target_dir, patient_name)
    os.makedirs(target_patient_dir, exist_ok=True)

    # 保存投影图像
    ap_save_path = os.path.join(target_patient_dir, "ap.nii.gz")
    lat_save_path = os.path.join(target_patient_dir, "lat.nii.gz")

    # 创建affine矩阵
    affine_2d = np.eye(4)
    try:
        orig_spacing = ct_img.header.get_zooms()
        if len(orig_spacing) >= 3:
            affine_2d[0, 0] = orig_spacing[2]  # 垂直方向间距
            affine_2d[1, 1] = orig_spacing[0]  # 水平方向间距
    except:
        pass

    # 保存为3D NIfTI
    nib.save(nib.Nifti1Image(ap_enhanced[..., np.newaxis], affine_2d), ap_save_path)
    nib.save(nib.Nifti1Image(lat_enhanced[..., np.newaxis], affine_2d), lat_save_path)

    # 复制mask文件
    if os.path.exists(mask_path):
        target_mask_path = os.path.join(target_patient_dir, "mask.nii.gz")
        shutil.copy2(mask_path, target_mask_path)

    return True


def main():
    source_dir = r"D:\med_data\biron\data\resized"
    target_dir = r"D:\med_data\biron\data\train_all"

    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 {source_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)

    # 获取所有病例文件夹
    patient_folders = sorted([f for f in os.listdir(source_dir)
                              if os.path.isdir(os.path.join(source_dir, f))])

    print(f"找到 {len(patient_folders)} 个病例文件夹")
    print(f"输出目录: {target_dir}")
    print("\n处理参数:")
    print("  投影方向: AP(正面) + LAT(侧面)")
    print("  旋转: 顺时针90度")
    print("  翻转: 上下翻转修正镜像")
    print("  归一化: 0-1范围")

    # 处理所有病例
    print("\n开始处理所有病例...")
    success_count = 0
    fail_count = 0

    for folder in tqdm(patient_folders, desc="Processing"):
        patient_path = os.path.join(source_dir, folder)
        if process_patient(patient_path, source_dir, target_dir):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"  成功: {success_count}/{len(patient_folders)}")
    print(f"  失败: {fail_count}/{len(patient_folders)}")

    # 验证第一个结果
    if success_count > 0:
        first_patient = patient_folders[0]
        first_output = os.path.join(target_dir, first_patient)
        print(f"\n验证第一个病例: {first_patient}")
        print(f"  输出目录: {first_output}")

        ap_path = os.path.join(first_output, "ap.nii.gz")
        lat_path = os.path.join(first_output, "lat.nii.gz")

        if os.path.exists(ap_path):
            ap_img = nib.load(ap_path)
            ap_data = ap_img.get_fdata()
            print(f"  AP: 形状={ap_data.shape}, 值域=[{ap_data.min():.3f}, {ap_data.max():.3f}]")

        if os.path.exists(lat_path):
            lat_img = nib.load(lat_path)
            lat_data = lat_img.get_fdata()
            print(f"  LAT: 形状={lat_data.shape}, 值域=[{lat_data.min():.3f}, {lat_data.max():.3f}]")


if __name__ == "__main__":
    main()