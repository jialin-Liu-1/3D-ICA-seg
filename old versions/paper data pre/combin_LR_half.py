import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import find_objects


def simple_merge_and_save(patient_dir, output_dir, target_shape=(128, 128, 128)):
    """最简单的合并方法"""

    ct_path = os.path.join(patient_dir, "CT.nii.gz")
    left_path = os.path.join(patient_dir, "common_carotid_artery_left.nii.gz")
    right_path = os.path.join(patient_dir, "common_carotid_artery_right.nii.gz")

    # 检查文件
    if not os.path.exists(ct_path):
        return
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"跳过 {os.path.basename(patient_dir)}: 缺少左右动脉文件")
        return

    # 1. 加载原始数据
    print(f"\n处理 {os.path.basename(patient_dir)}")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()

    left_img = nib.load(left_path)
    right_img = nib.load(right_path)

    left_data = left_img.get_fdata()
    right_data = right_img.get_fdata()

    print(f"  左动脉体素: {np.sum(left_data > 0)}")
    print(f"  右动脉体素: {np.sum(right_data > 0)}")
    print(f"  图像形状: {ct_data.shape}")

    # 2. 创建合并mask (直接使用原始数据，不做任何变换)
    # 注意：这里直接创建4D数组，通道0=左，通道1=右
    merged_mask = np.zeros(ct_data.shape + (2,), dtype=np.uint8)
    merged_mask[..., 0] = (left_data > 0).astype(np.uint8)
    merged_mask[..., 1] = (right_data > 0).astype(np.uint8)

    print(f"  合并后左通道体素: {np.sum(merged_mask[..., 0] > 0)}")
    print(f"  合并后右通道体素: {np.sum(merged_mask[..., 1] > 0)}")

    # 3. 保存原始尺寸的合并mask (不裁剪)
    patient_out_dir = os.path.join(output_dir, os.path.basename(patient_dir))
    os.makedirs(patient_out_dir, exist_ok=True)

    org_save_path = os.path.join(patient_out_dir, "mask_org.nii.gz")
    nib.save(nib.Nifti1Image(merged_mask, ct_img.affine, ct_img.header), org_save_path)
    print(f"  已保存: {org_save_path}")

    # 4. 如果需要裁剪到128x128x128
    # 找到动脉区域的边界
    artery_mask = (merged_mask[..., 0] + merged_mask[..., 1]) > 0
    if np.any(artery_mask):
        slices = find_objects(artery_mask)[0]
        z_start, z_end = slices[0].start, slices[0].stop
        y_start, y_end = slices[1].start, slices[1].stop
        x_start, x_end = slices[2].start, slices[2].stop

        # 计算中心点
        z_center = (z_start + z_end) // 2
        y_center = (y_start + y_end) // 2
        x_center = (x_start + x_end) // 2

        # 以中心裁剪128x128x128
        half_z, half_y, half_x = target_shape[0] // 2, target_shape[1] // 2, target_shape[2] // 2
        new_z_start = max(0, z_center - half_z)
        new_y_start = max(0, y_center - half_y)
        new_x_start = max(0, x_center - half_x)

        new_z_end = min(ct_data.shape[0], new_z_start + target_shape[0])
        new_y_end = min(ct_data.shape[1], new_y_start + target_shape[1])
        new_x_end = min(ct_data.shape[2], new_x_start + target_shape[2])

        # 裁剪CT
        ct_cropped = ct_data[new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end]
        # 裁剪mask
        mask_cropped = merged_mask[new_z_start:new_z_end, new_y_start:new_y_end, new_x_start:new_x_end, :]

        # 如果尺寸不足128，填充
        if ct_cropped.shape != target_shape:
            ct_resized = np.full(target_shape, np.min(ct_data), dtype=ct_data.dtype)
            mask_resized = np.zeros(target_shape + (2,), dtype=np.uint8)

            ct_resized[:ct_cropped.shape[0], :ct_cropped.shape[1], :ct_cropped.shape[2]] = ct_cropped
            mask_resized[:mask_cropped.shape[0], :mask_cropped.shape[1], :mask_cropped.shape[2], :] = mask_cropped
        else:
            ct_resized = ct_cropped
            mask_resized = mask_cropped

        # 保存裁剪后的文件
        ct_save_path = os.path.join(patient_out_dir, "ct.nii.gz")
        mask_save_path = os.path.join(patient_out_dir, "mask.nii.gz")

        nib.save(nib.Nifti1Image(ct_resized, ct_img.affine, ct_img.header), ct_save_path)
        nib.save(nib.Nifti1Image(mask_resized, ct_img.affine, ct_img.header), mask_save_path)

        print(f"  已保存裁剪后CT: {ct_save_path}")
        print(f"  已保存裁剪后mask: {mask_save_path}")
        print(f"  裁剪后左通道体素: {np.sum(mask_resized[..., 0] > 0)}")
        print(f"  裁剪后右通道体素: {np.sum(mask_resized[..., 1] > 0)}")
    else:
        print(f"  警告: 未找到动脉区域")


def main():
    raw_dir = r"D:\med_data\biron\data\raw"
    output_dir = r"D:\med_data\biron\data\resized"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有病例文件夹
    patient_folders = sorted([f for f in os.listdir(raw_dir)
                              if os.path.isdir(os.path.join(raw_dir, f))])

    print(f"找到 {len(patient_folders)} 个病例文件夹")

    # 处理所有病例
    for folder in tqdm(patient_folders, desc="Processing"):
        patient_path = os.path.join(raw_dir, folder)
        simple_merge_and_save(patient_path, output_dir)

    print("\n" + "=" * 50)
    print("处理完成！验证前3个结果:")

    # 验证结果
    for folder in patient_folders[:3]:
        patient_out_dir = os.path.join(output_dir, folder)
        mask_path = os.path.join(patient_out_dir, "mask_org.nii.gz")
        if os.path.exists(mask_path):
            mask = nib.load(mask_path).get_fdata()
            print(f"\n{folder}:")
            print(f"  左动脉(通道0): {np.sum(mask[..., 0] > 0)} 体素")
            print(f"  右动脉(通道1): {np.sum(mask[..., 1] > 0)} 体素")


if __name__ == "__main__":
    main()