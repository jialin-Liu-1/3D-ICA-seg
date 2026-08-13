import nibabel as nib
import numpy as np
import os


def check_mask_channels(mask_path, name):
    """详细检查mask的通道"""
    print(f"\n=== 检查 {name} ===")
    img = nib.load(mask_path)
    data = img.get_fdata()
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    if data.ndim == 4:
        print(f"通道数: {data.shape[-1]}")
        for c in range(data.shape[-1]):
            nonzero = np.sum(data[..., c] > 0)
            print(f"  通道{c}: {nonzero} 非零体素")
            if nonzero > 0:
                # 显示通道中非零值的范围
                unique_vals = np.unique(data[..., c][data[..., c] > 0])
                print(f"    非零值范围: {unique_vals}")
    else:
        print(f"这不是4D数组，是{data.ndim}D数组")
        nonzero = np.sum(data > 0)
        print(f"非零体素: {nonzero}")
        if nonzero > 0:
            unique_vals = np.unique(data[data > 0])
            print(f"非零值: {unique_vals}")


# 检查你生成的文件
output_dir = r"D:\med_data\biron\data\resized"

# 检查一个病例的mask_org.nii.gz
test_patient = "s0112"  # 根据你的打印信息
mask_org_path = os.path.join(output_dir, test_patient, "mask_org.nii.gz")
mask_path = os.path.join(output_dir, test_patient, "mask.nii.gz")

check_mask_channels(mask_org_path, f"{test_patient}/mask_org.nii.gz")
check_mask_channels(mask_path, f"{test_patient}/mask.nii.gz")
