import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import find_objects


def merge_and_save_correctly(patient_dir, output_dir, target_shape=(128, 128, 128)):
    """正确合并并保存双通道mask"""

    ct_path = os.path.join(patient_dir, "CT.nii.gz")
    left_path = os.path.join(patient_dir, "common_carotid_artery_left.nii.gz")
    right_path = os.path.join(patient_dir, "common_carotid_artery_right.nii.gz")

    # 检查文件
    if not os.path.exists(ct_path):
        return False
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"跳过 {os.path.basename(patient_dir)}: 缺少左右动脉文件")
        return False

    # 加载数据
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()

    left_img = nib.load(left_path)
    right_img = nib.load(right_path)

    left_data = (left_img.get_fdata() > 0).astype(np.uint8)
    right_data = (right_img.get_fdata() > 0).astype(np.uint8)

    print(f"\n处理 {os.path.basename(patient_dir)}")
    print(f"  左动脉体素: {np.sum(left_data)}")
    print(f"  右动脉体素: {np.sum(right_data)}")

    # 创建双通道mask - 通道0=左, 通道1=右
    merged_mask = np.stack([left_data, right_data], axis=-1)

    print(f"  合并后形状: {merged_mask.shape}")
    print(f"  合并后左通道(通道0): {np.sum(merged_mask[..., 0])}")
    print(f"  合并后右通道(通道1): {np.sum(merged_mask[..., 1])}")

    # 创建输出目录
    patient_out_dir = os.path.join(output_dir, os.path.basename(patient_dir))
    os.makedirs(patient_out_dir, exist_ok=True)

    # 保存原始尺寸的mask
    org_save_path = os.path.join(patient_out_dir, "mask_org.nii.gz")
    nib.save(nib.Nifti1Image(merged_mask, ct_img.affine, ct_img.header), org_save_path)
    print(f"  已保存: {org_save_path}")

    # 验证保存的文件
    verify_mask = nib.load(org_save_path)
    verify_data = verify_mask.get_fdata()
    print(
        f"  验证保存文件: 形状={verify_data.shape}, 左通道={np.sum(verify_data[..., 0])}, 右通道={np.sum(verify_data[..., 1])}")

    # 裁剪到目标尺寸
    artery_mask = (left_data + right_data) > 0
    if np.any(artery_mask):
        # 获取动脉边界框
        coords = np.where(artery_mask)
        z_min, z_max = np.min(coords[0]), np.max(coords[0])
        y_min, y_max = np.min(coords[1]), np.max(coords[1])
        x_min, x_max = np.min(coords[2]), np.max(coords[2])

        # 计算中心
        z_center = (z_min + z_max) // 2
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2

        # 裁剪
        half_z, half_y, half_x = target_shape[0] // 2, target_shape[1] // 2, target_shape[2] // 2
        new_z_start = max(0, z_center - half_z)
        new_y_start = max(0, y_center - half_y)
        new_x_start = max(0, x_center - half_x)

        new_z_end = min(ct_data.shape[0], new_z_start + target_shape[0])
        new_y_end = min(ct_data.shape[1], new_y_start + target_shape[1])
        new_x_end = min(ct_data.shape[2], new_x_start + target_shape[2])

        # 裁剪
        ct_cropped = ct_data[new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end]
        mask_cropped = merged_mask[new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end, :]

        # 填充到目标尺寸
        ct_resized = np.full(target_shape, np.min(ct_data), dtype=ct_data.dtype)
        mask_resized = np.zeros(target_shape + (2,), dtype=np.uint8)

        ct_resized[:ct_cropped.shape[0], :ct_cropped.shape[1], :ct_cropped.shape[2]] = ct_cropped
        mask_resized[:mask_cropped.shape[0], :mask_cropped.shape[1], :mask_cropped.shape[2], :] = mask_cropped

        # 保存裁剪后的文件
        ct_save_path = os.path.join(patient_out_dir, "ct.nii.gz")
        mask_save_path = os.path.join(patient_out_dir, "mask.nii.gz")

        nib.save(nib.Nifti1Image(ct_resized, ct_img.affine, ct_img.header), ct_save_path)
        nib.save(nib.Nifti1Image(mask_resized, ct_img.affine, ct_img.header), mask_save_path)

        print(f"  裁剪后左通道: {np.sum(mask_resized[..., 0])}")
        print(f"  裁剪后右通道: {np.sum(mask_resized[..., 1])}")

        # 最终验证
        final_mask = nib.load(mask_save_path).get_fdata()
        print(f"  最终验证 mask.nii.gz: 左={np.sum(final_mask[..., 0])}, 右={np.sum(final_mask[..., 1])}")

    return True


def main():
    raw_dir = r"D:\med_data\biron\data\raw"
    output_dir = r"D:\med_data\biron\data\resized"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有病例文件夹
    patient_folders = sorted([f for f in os.listdir(raw_dir)
                              if os.path.isdir(os.path.join(raw_dir, f))])

    print(f"找到 {len(patient_folders)} 个病例文件夹")
    print(f"\n开始处理所有 {len(patient_folders)} 个病例...")

    # 处理所有病例（修改这里：将 test_folders 改为 patient_folders）
    success_count = 0
    for folder in tqdm(patient_folders, desc="Processing all patients"):
        patient_path = os.path.join(raw_dir, folder)
        if merge_and_save_correctly(patient_path, output_dir):
            success_count += 1

    print("\n" + "=" * 50)
    print(f"处理完成！成功: {success_count}/{len(patient_folders)}")
    print("\n验证前5个结果:")

    # 验证前5个结果
    for folder in patient_folders[:5]:
        patient_out_dir = os.path.join(output_dir, folder)
        mask_org_path = os.path.join(patient_out_dir, "mask_org.nii.gz")
        mask_path = os.path.join(patient_out_dir, "mask.nii.gz")
        ct_path = os.path.join(patient_out_dir, "ct.nii.gz")

        if os.path.exists(mask_org_path):
            mask_org = nib.load(mask_org_path).get_fdata()
            print(f"\n{folder}:")
            print(f"  mask_org.nii.gz - 左: {np.sum(mask_org[..., 0] > 0):6d}, 右: {np.sum(mask_org[..., 1] > 0):6d}")

        if os.path.exists(mask_path):
            mask = nib.load(mask_path).get_fdata()
            print(f"  mask.nii.gz     - 左: {np.sum(mask[..., 0] > 0):6d}, 右: {np.sum(mask[..., 1] > 0):6d}")

        if os.path.exists(ct_path):
            ct = nib.load(ct_path).get_fdata()
            print(f"  ct.nii.gz       - 形状: {ct.shape}")


if __name__ == "__main__":
    main()