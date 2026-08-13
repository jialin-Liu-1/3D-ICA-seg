import os
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm


def fix_nifti_header(input_path, output_path):
    """
    修复 NIfTI 文件头，使其能被 ITK-SNAP 正确读取
    """
    img = nib.load(input_path)
    img.set_qform(img.affine, code=1)
    img.set_sform(img.affine, code=1)
    nib.save(img, output_path)


def check_nonzero_mask(mask_path):
    """
    检查分割 mask 是否全为零
    返回: (是否非零, 最大值)
    """
    if not os.path.exists(mask_path):
        return False, 0

    img = nib.load(mask_path)
    data = img.get_fdata()
    max_val = data.max()
    return max_val > 0, max_val


def process_dataset():
    # 路径配置
    source_root = r"D:\med_data\biron\Totalsegmentator_dataset_v201"
    target_root = r"D:\med_data\biron\data\raw"

    # 创建目标根目录
    os.makedirs(target_root, exist_ok=True)

    # 获取所有病例子文件夹
    case_folders = [f for f in os.listdir(source_root)
                    if os.path.isdir(os.path.join(source_root, f))
                    and f.startswith('s')]  # 以 s 开头的文件夹（如 s0000, s0001...）

    print(f"找到 {len(case_folders)} 个病例文件夹")

    # 统计变量
    valid_cases = 0
    invalid_cases = 0
    missing_files = 0

    # 遍历每个病例
    for case_name in tqdm(case_folders, desc="处理病例"):
        case_source_path = os.path.join(source_root, case_name)
        seg_source_path = os.path.join(case_source_path, "segmentations")

        # 定义文件路径
        ct_path = os.path.join(case_source_path, "ct.nii.gz")
        left_mask_path = os.path.join(seg_source_path, "common_carotid_artery_left.nii.gz")
        right_mask_path = os.path.join(seg_source_path, "common_carotid_artery_right.nii.gz")

        # 检查必要文件是否存在
        if not os.path.exists(ct_path):
            print(f"  警告: {case_name} 缺少 CT 文件")
            missing_files += 1
            continue

        # 检查左右颈总动脉文件（至少有一个存在即可）
        left_exists = os.path.exists(left_mask_path)
        right_exists = os.path.exists(right_mask_path)

        if not left_exists and not right_exists:
            print(f"  警告: {case_name} 缺少左右颈总动脉分割文件")
            missing_files += 1
            continue

        # 检查分割是否非零
        left_valid, left_max = check_nonzero_mask(left_mask_path) if left_exists else (False, 0)
        right_valid, right_max = check_nonzero_mask(right_mask_path) if right_exists else (False, 0)

        # 如果至少有一个颈总动脉非零，则保留该病例
        if left_valid or right_valid:
            valid_cases += 1

            # 创建目标病例文件夹
            target_case_path = os.path.join(target_root, case_name)
            os.makedirs(target_case_path, exist_ok=True)

            # 修复并复制 CT 图像
            target_ct_path = os.path.join(target_case_path, "CT.nii.gz")
            fix_nifti_header(ct_path, target_ct_path)

            # 修复并复制左颈总动脉（如果存在且非零）
            if left_exists and left_valid:
                target_left_path = os.path.join(target_case_path, "common_carotid_artery_left.nii.gz")
                fix_nifti_header(left_mask_path, target_left_path)
                print(f"  ✓ {case_name}: 左颈总动脉 (max={left_max:.0f})")
            elif left_exists and not left_valid:
                print(f"  ✗ {case_name}: 左颈总动脉全零，已跳过")

            # 修复并复制右颈总动脉（如果存在且非零）
            if right_exists and right_valid:
                target_right_path = os.path.join(target_case_path, "common_carotid_artery_right.nii.gz")
                fix_nifti_header(right_mask_path, target_right_path)
                print(f"  ✓ {case_name}: 右颈总动脉 (max={right_max:.0f})")
            elif right_exists and not right_valid:
                print(f"  ✗ {case_name}: 右颈总动脉全零，已跳过")

        else:
            invalid_cases += 1
            if left_exists or right_exists:
                print(f"  ✗ {case_name}: 颈总动脉分割全零，已跳过")

    # 输出统计信息
    print("\n" + "=" * 50)
    print("处理完成！")
    print(f"总病例数: {len(case_folders)}")
    print(f"有效病例（至少一侧颈总动脉非零）: {valid_cases}")
    print(f"无效病例（颈总动脉全零或缺失）: {invalid_cases}")
    print(f"缺失文件病例数: {missing_files}")
    print(f"输出目录: {target_root}")
    print("=" * 50)


def verify_output():
    """
    验证输出目录的文件结构
    """
    target_root = r"D:\med_data\biron\data\raw"

    print("\n验证输出目录结构...")
    case_folders = [f for f in os.listdir(target_root)
                    if os.path.isdir(os.path.join(target_root, f))]

    print(f"找到 {len(case_folders)} 个病例文件夹")

    for case_name in case_folders[:5]:  # 只显示前5个作为示例
        case_path = os.path.join(target_root, case_name)
        files = os.listdir(case_path)
        print(f"  {case_name}: {files}")


if __name__ == "__main__":
    # 运行主处理程序
    process_dataset()

    # 可选：验证输出
    verify_output()