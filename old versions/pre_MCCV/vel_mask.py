import numpy as np
import nibabel as nib
import os
import glob
from tqdm import tqdm
import re

# ============================================================
# 参数配置
# ============================================================
INPUT_BASE = r"D:\med_data\biron\data1\raw_nii"
OUTPUT_BASE = r"D:\med_data\biron\data1\mask"

# 二值化阈值
THRESHOLD = 0.1  # 大于等于此值的设为1，小于的设为0

# 是否跳过已存在的文件（避免重复处理）
SKIP_EXISTING = True

# 输出数据类型（通常二值图像用uint8即可）
OUTPUT_DTYPE = np.uint8  # 可选: np.uint8, np.uint16, np.float32


def apply_threshold(image_data, threshold=0.1):
    """
    对图像数据应用阈值，生成二值图像

    参数:
        image_data: 输入图像数据（任意维度）
        threshold: 阈值，大于等于此值的设为1，小于的设为0

    返回:
        二值图像，值为0或1
    """
    # 创建二值图像
    binary_data = (image_data >= threshold).astype(OUTPUT_DTYPE)

    return binary_data


def process_nifti_file(input_file, output_dir, threshold=0.1, skip_existing=True):
    """
    处理单个NIfTI文件，转换为二值图像

    参数:
        input_file: 输入的.nii或.nii.gz文件路径
        output_dir: 输出目录
        threshold: 阈值
        skip_existing: 是否跳过已存在的文件

    返回:
        (成功标志, 输出文件路径, 统计信息)
    """
    # 生成输出文件名（保持原文件名，添加_mask后缀）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]  # 移除.nii后缀

    output_file = os.path.join(output_dir, f"{base_name}_mask.nii.gz")

    # 检查是否跳过已存在的文件
    if skip_existing and os.path.exists(output_file):
        print(f"  跳过已存在的文件: {os.path.basename(output_file)}")
        return True, output_file, None

    try:
        # 加载NIfTI文件
        nii_img = nib.load(input_file)
        original_data = nii_img.get_fdata()

        # 获取原始数据类型和范围
        original_dtype = original_data.dtype
        min_val = np.min(original_data)
        max_val = np.max(original_data)
        mean_val = np.mean(original_data)

        print(f"  原始数据: 形状 {original_data.shape}, 类型 {original_dtype}")
        print(f"  数值范围: [{min_val:.6f}, {max_val:.6f}], 均值: {mean_val:.6f}")

        # 应用阈值生成二值图像
        binary_data = apply_threshold(original_data, threshold)

        # 统计二值图像信息
        num_voxels_foreground = np.sum(binary_data == 1)
        num_voxels_background = np.sum(binary_data == 0)
        total_voxels = binary_data.size
        foreground_percent = (num_voxels_foreground / total_voxels) * 100

        print(f"  阈值: {threshold}")
        print(
            f"  二值化结果: 前景体素 {num_voxels_foreground} ({foreground_percent:.2f}%), 背景体素 {num_voxels_background}")

        # 创建新的NIfTI图像，保持原始仿射变换
        binary_nii = nib.Nifti1Image(binary_data, nii_img.affine, nii_img.header)

        # 更新header信息
        binary_nii.header['descrip'] = f'Binary mask (threshold >= {threshold})'
        binary_nii.header['cal_min'] = 0.0
        binary_nii.header['cal_max'] = 1.0

        # 保存文件
        nib.save(binary_nii, output_file)

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ 转换成功: {os.path.basename(output_file)} ({file_size_mb:.2f} MB)")

        stats = {
            'original_shape': original_data.shape,
            'original_range': (min_val, max_val),
            'threshold': threshold,
            'foreground_voxels': int(num_voxels_foreground),
            'background_voxels': int(num_voxels_background),
            'foreground_percent': foreground_percent
        }

        return True, output_file, stats

    except Exception as e:
        print(f"  ✗ 处理失败: {os.path.basename(input_file)} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def find_nifti_files(input_folder):
    """
    查找文件夹中的所有NIfTI文件

    参数:
        input_folder: 输入文件夹路径

    返回:
        NIfTI文件路径列表，按文件名排序
    """
    # 查找所有.nii和.nii.gz文件
    nii_files = []
    nii_files.extend(glob.glob(os.path.join(input_folder, "*.nii")))
    nii_files.extend(glob.glob(os.path.join(input_folder, "*.nii.gz")))

    # 排除已经是mask的文件（可选）
    nii_files = [f for f in nii_files if not os.path.basename(f).endswith('_mask.nii.gz')]

    # 按文件名排序
    def extract_number(filename):
        basename = os.path.basename(filename)
        # 提取数字（例如：0.nii.gz -> 0）
        match = re.search(r'^(\d+)', basename)
        if match:
            return int(match.group(1))
        return 0

    nii_files.sort(key=extract_number)

    return nii_files


def batch_process_nifti_to_mask():
    """
    批量处理所有NIfTI文件，转换为二值图像
    """
    print("=" * 80)
    print("NIfTI 到 二值Mask 批量转换工具")
    print("=" * 80)
    print(f"输入文件夹: {INPUT_BASE}")
    print(f"输出文件夹: {OUTPUT_BASE}")
    print(f"阈值: {THRESHOLD} (像素值 >= {THRESHOLD} 设为 1, 小于设为 0)")
    print(f"输出数据类型: {OUTPUT_DTYPE}")
    print(f"跳过已存在文件: {SKIP_EXISTING}")
    print("=" * 80)

    # 检查输入路径是否存在
    if not os.path.exists(INPUT_BASE):
        print(f"错误：输入路径不存在: {INPUT_BASE}")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 查找所有NIfTI文件
    nii_files = find_nifti_files(INPUT_BASE)

    if len(nii_files) == 0:
        print(f"警告：在 {INPUT_BASE} 中没有找到NIfTI文件 (.nii 或 .nii.gz)")
        return

    print(f"\n找到 {len(nii_files)} 个NIfTI文件")
    print("开始批量转换...\n")

    # 统计信息
    success_count = 0
    fail_count = 0
    all_stats = []
    fail_files = []

    # 使用进度条
    try:
        from tqdm import tqdm
        pbar = tqdm(nii_files, desc="转换进度", unit="文件")
        for nii_file in pbar:
            file_name = os.path.basename(nii_file)
            pbar.set_description(f"处理 {file_name}")

            success, output_file, stats = process_nifti_file(
                nii_file, OUTPUT_BASE,
                threshold=THRESHOLD,
                skip_existing=SKIP_EXISTING
            )

            if success:
                success_count += 1
                if stats:
                    all_stats.append((file_name, stats))
            else:
                fail_count += 1
                fail_files.append(nii_file)
    except ImportError:
        # 如果没有tqdm，使用普通循环
        for i, nii_file in enumerate(nii_files, 1):
            file_name = os.path.basename(nii_file)
            print(f"[{i}/{len(nii_files)}] 处理 {file_name}...")

            success, output_file, stats = process_nifti_file(
                nii_file, OUTPUT_BASE,
                threshold=THRESHOLD,
                skip_existing=SKIP_EXISTING
            )

            if success:
                success_count += 1
                if stats:
                    all_stats.append((file_name, stats))
            else:
                fail_count += 1
                fail_files.append(nii_file)

    # 输出统计结果
    print("\n" + "=" * 80)
    print("批量转换完成！")
    print("=" * 80)
    print(f"总文件数: {len(nii_files)}")
    print(f"成功转换: {success_count}")
    print(f"转换失败: {fail_count}")
    print(f"输出目录: {OUTPUT_BASE}")

    # 显示详细统计信息
    if all_stats:
        print("\n详细统计信息:")
        print("-" * 80)
        for file_name, stats in all_stats[:10]:  # 只显示前10个
            print(f"{file_name}:")
            print(f"  原始形状: {stats['original_shape']}")
            print(f"  原始范围: [{stats['original_range'][0]:.6f}, {stats['original_range'][1]:.6f}]")
            print(f"  前景体素: {stats['foreground_voxels']:,} ({stats['foreground_percent']:.2f}%)")
            print(f"  背景体素: {stats['background_voxels']:,}")

        if len(all_stats) > 10:
            print(f"... 还有 {len(all_stats) - 10} 个文件未显示")

    if fail_files:
        print("\n失败的文件列表:")
        for f in fail_files:
            print(f"  - {os.path.basename(f)}")

    print("=" * 80)


def analyze_threshold_effect(input_file, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25]):
    """
    分析不同阈值对二值化的影响（辅助功能）

    参数:
        input_file: 输入的NIfTI文件
        thresholds: 要测试的阈值列表
    """
    print(f"\n分析文件: {os.path.basename(input_file)}")
    print("=" * 60)

    try:
        # 加载数据
        nii_img = nib.load(input_file)
        data = nii_img.get_fdata()

        print(f"数据形状: {data.shape}")
        print(f"数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
        print(f"数据均值: {np.mean(data):.6f}")
        print(f"数据标准差: {np.std(data):.6f}")

        print("\n不同阈值的效果:")
        print("-" * 60)
        print(f"{'阈值':<10} {'前景体素数':<15} {'前景百分比':<12} {'背景体素数':<15}")
        print("-" * 60)

        for threshold in thresholds:
            binary = (data >= threshold).astype(np.uint8)
            foreground = np.sum(binary == 1)
            background = np.sum(binary == 0)
            percent = (foreground / binary.size) * 100
            print(f"{threshold:<10.2f} {foreground:<15,} {percent:<12.2f}% {background:<15,}")

        print("=" * 60)

    except Exception as e:
        print(f"分析失败: {str(e)}")


def test_single_file():
    """
    测试单个文件的转换（用于调试）
    """
    test_file = r"D:\med_data\biron\data1\raw_nii\0.nii.gz"
    output_dir = r"D:\med_data\biron\data1\mask_test"

    print("测试模式：转换单个NIfTI文件为二值Mask")
    print(f"测试文件: {test_file}")

    # 先分析不同阈值的效果
    analyze_threshold_effect(test_file, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25])

    os.makedirs(output_dir, exist_ok=True)

    # 使用设定的阈值进行转换
    success, output_file, stats = process_nifti_file(
        test_file, output_dir,
        threshold=THRESHOLD,
        skip_existing=False
    )

    if success and stats:
        print(f"\n转换成功!")
        print(f"输出文件: {output_file}")
        print(f"统计信息: {stats}")

    print(f"\n转换{'成功' if success else '失败'}")


def verify_binary_mask(mask_file):
    """
    验证二值mask文件
    """
    print(f"\n验证Mask文件: {os.path.basename(mask_file)}")
    print("-" * 60)

    try:
        img = nib.load(mask_file)
        data = img.get_fdata()

        unique_values = np.unique(data)

        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"唯一值: {unique_values}")
        print(f"最小值: {np.min(data)}")
        print(f"最大值: {np.max(data)}")

        # 检查是否为有效的二值图像
        is_binary = np.all((data == 0) | (data == 1))
        print(f"是否为有效二值图像: {'是' if is_binary else '否'}")

        if is_binary:
            foreground = np.sum(data == 1)
            background = np.sum(data == 0)
            total = data.size
            print(f"前景体素: {foreground:,} ({foreground / total * 100:.2f}%)")
            print(f"背景体素: {background:,} ({background / total * 100:.2f}%)")

        print("-" * 60)

    except Exception as e:
        print(f"验证失败: {str(e)}")


if __name__ == "__main__":
    # 选择运行模式
    RUN_MODE = "batch"  # "batch" 批量处理所有文件, "test" 测试单个文件

    # 阈值设置（可根据需要调整）
    THRESHOLD = 0.1  # 像素值大于等于0.1设为1，小于0.1设为0

    if RUN_MODE == "batch":
        batch_process_nifti_to_mask()
    elif RUN_MODE == "test":
        test_single_file()
    else:
        print("请设置正确的运行模式: 'batch' 或 'test'")

    # 可选：验证生成的mask文件
    # verify_binary_mask(r"D:\med_data\biron\data1\mask\0_mask.nii.gz")