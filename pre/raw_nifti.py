import numpy as np
import nibabel as nib
import os
import glob
from tqdm import tqdm
import re

# ============================================================
# 参数配置
# ============================================================
IMAGE_SIZE = 256  # 三维图像尺寸：256×256×256
INPUT_BASE = r"E:\3D-DSA"
OUTPUT_BASE = r"D:\med_data\biron\data1\raw_nii"

# 是否跳过已存在的文件（避免重复转换）
SKIP_EXISTING = True

# 体素尺寸（毫米/体素）- 根据实际情况调整
# 从原始代码中，重建体素大小为0.5mm
VOXEL_SIZE_MM = [0.5, 0.5, 0.5]  # x, y, z方向的体素尺寸


def read_3d_raw(file_path, size=256, dtype=np.float32):
    """
    读取三维RAW文件

    参数:
        file_path: .raw文件路径
        size: 三维尺寸（size×size×size）
        dtype: 数据类型

    返回:
        numpy数组，形状为[size, size, size]，指定dtype
    """
    # 计算期望的数据点数
    expected_points = size * size * size

    # 读取原始数据
    data = np.fromfile(file_path, dtype=dtype)

    if data.size != expected_points:
        raise ValueError(
            f"文件大小异常: {data.size} 个值\n"
            f"期望: {expected_points} 个值 ({size}×{size}×{size})\n"
            f"文件: {file_path}"
        )

    # 重塑为3D数组
    # 注意：RAW文件存储顺序为 [z, y, x] 还是 [x, y, z]？
    # 根据原始代码中的投影图像是 [nz, nx]，三维重建输出是 [recon_size, recon_size, recon_size]
    # ASTRA的输出通常是 [x, y, z] 顺序，但保存时是连续的
    # 这里假设存储顺序为 [x, y, z]
    volume_3d = data.reshape(size, size, size)

    return volume_3d


def save_as_nifti(volume_data, output_path, affine=None, voxel_size=None):
    """
    将3D numpy数组保存为NIfTI格式，保持原始数据类型

    参数:
        volume_data: 3D numpy数组
        output_path: 输出文件路径（.nii或.nii.gz）
        affine: 仿射变换矩阵（如果为None则根据voxel_size创建）
        voxel_size: 体素尺寸 [x, y, z]（毫米）
    """
    if affine is None:
        if voxel_size is None:
            # 默认体素尺寸为1mm
            voxel_size = [1.0, 1.0, 1.0]

        # 创建简单的仿射变换矩阵
        # 标准的RAS+坐标系
        affine = np.diag(voxel_size + [1.0])

    # 创建NIfTI图像，保持原始数据类型
    nii_img = nib.Nifti1Image(volume_data, affine)

    # 设置元数据
    nii_img.header['descrip'] = 'Converted from MC-GPU 3D DSA raw data (float32)'
    nii_img.header['cal_min'] = float(np.min(volume_data))
    nii_img.header['cal_max'] = float(np.max(volume_data))

    # 保存文件（支持.nii和.nii.gz）
    nib.save(nii_img, output_path)


def find_3d_raw_files(input_folder):
    """
    查找文件夹中的所有3D RAW文件（数字命名的.raw文件）

    参数:
        input_folder: 输入文件夹路径

    返回:
        raw文件路径列表，按数字编号排序
    """
    # 查找所有.raw文件
    raw_files = glob.glob(os.path.join(input_folder, "*.raw"))

    # 提取数字编号并排序
    raw_files_with_numbers = []
    for f in raw_files:
        base_name = os.path.basename(f)
        # 提取文件名中的数字（例如：2.raw -> 2）
        match = re.search(r'^(\d+)\.raw$', base_name)
        if match:
            number = int(match.group(1))
            raw_files_with_numbers.append((number, f))

    # 按数字编号排序
    raw_files_with_numbers.sort(key=lambda x: x[0])
    sorted_files = [f for num, f in raw_files_with_numbers]

    return sorted_files


def convert_single_3d_raw(raw_file, output_dir, voxel_size=None, skip_existing=True):
    """
    转换单个3D RAW文件为NIfTI格式，保持原始精度（float32）

    参数:
        raw_file: 输入的.raw文件路径
        output_dir: 输出目录
        voxel_size: 体素尺寸（毫米）
        skip_existing: 是否跳过已存在的文件

    返回:
        (成功标志, 输出文件路径)
    """
    # 生成输出文件名（保持原始数据类型，明确标注float32）
    base_name = os.path.splitext(os.path.basename(raw_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.nii.gz")

    # 检查是否跳过已存在的文件
    if skip_existing and os.path.exists(output_file):
        print(f"  跳过已存在的文件: {os.path.basename(output_file)}")
        return True, output_file

    try:
        # 读取RAW文件（保持float32）
        volume_data = read_3d_raw(raw_file, size=IMAGE_SIZE, dtype=np.float32)

        # 输出数据信息
        min_val = np.min(volume_data)
        max_val = np.max(volume_data)
        mean_val = np.mean(volume_data)
        std_val = np.std(volume_data)

        print(f"  读取成功: 形状 {volume_data.shape}, 类型 {volume_data.dtype}")
        print(f"  数值范围: [{min_val:.6f}, {max_val:.6f}], 均值: {mean_val:.6f}, 标准差: {std_val:.6f}")

        # 保存为NIfTI（保持float32）
        save_as_nifti(volume_data, output_file, voxel_size=voxel_size)

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ 转换成功: {os.path.basename(output_file)} ({file_size_mb:.2f} MB, 保持float32精度)")

        return True, output_file

    except Exception as e:
        print(f"  ✗ 转换失败: {os.path.basename(raw_file)} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def batch_convert_3d_raw():
    """
    批量转换所有3D RAW文件
    """
    print("=" * 80)
    print("3D RAW 到 NIfTI 批量转换工具 (保持原始精度 - float32)")
    print("=" * 80)
    print(f"输入文件夹: {INPUT_BASE}")
    print(f"输出文件夹: {OUTPUT_BASE}")
    print(f"图像尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"体素尺寸: {VOXEL_SIZE_MM[0]}×{VOXEL_SIZE_MM[1]}×{VOXEL_SIZE_MM[2]} mm")
    print(f"数据类型: 保持原始 float32 (不进行归一化)")
    print(f"跳过已存在文件: {SKIP_EXISTING}")
    print("=" * 80)

    # 检查输入路径是否存在
    if not os.path.exists(INPUT_BASE):
        print(f"错误：输入路径不存在: {INPUT_BASE}")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 查找所有3D RAW文件
    raw_files = find_3d_raw_files(INPUT_BASE)

    if len(raw_files) == 0:
        print(f"警告：在 {INPUT_BASE} 中没有找到数字命名的.raw文件")
        print("请确保文件命名格式为：数字.raw (例如：0.raw, 1.raw, 2.raw)")
        return

    print(f"\n找到 {len(raw_files)} 个3D RAW文件")
    print("开始批量转换...\n")

    # 统计信息
    success_count = 0
    fail_count = 0
    success_files = []
    fail_files = []

    # 使用进度条
    try:
        from tqdm import tqdm
        pbar = tqdm(raw_files, desc="转换进度", unit="文件")
        for raw_file in pbar:
            file_name = os.path.basename(raw_file)
            pbar.set_description(f"转换 {file_name}")

            success, output_file = convert_single_3d_raw(
                raw_file, OUTPUT_BASE,
                voxel_size=VOXEL_SIZE_MM,
                skip_existing=SKIP_EXISTING
            )

            if success:
                success_count += 1
                success_files.append((raw_file, output_file))
            else:
                fail_count += 1
                fail_files.append(raw_file)
    except ImportError:
        # 如果没有tqdm，使用普通循环
        for i, raw_file in enumerate(raw_files, 1):
            file_name = os.path.basename(raw_file)
            print(f"[{i}/{len(raw_files)}] 转换 {file_name}...")

            success, output_file = convert_single_3d_raw(
                raw_file, OUTPUT_BASE,
                voxel_size=VOXEL_SIZE_MM,
                skip_existing=SKIP_EXISTING
            )

            if success:
                success_count += 1
                success_files.append((raw_file, output_file))
            else:
                fail_count += 1
                fail_files.append(raw_file)

    # 输出统计结果
    print("\n" + "=" * 80)
    print("批量转换完成！")
    print("=" * 80)
    print(f"总文件数: {len(raw_files)}")
    print(f"成功转换: {success_count}")
    print(f"转换失败: {fail_count}")
    print(f"输出目录: {OUTPUT_BASE}")
    print(f"数据类型: 保持原始 float32 精度")

    if success_count > 0:
        # 显示第一个成功文件的详细信息作为示例
        first_file = success_files[0][1]
        print(f"\n示例输出文件: {os.path.basename(first_file)}")
        verify_nifti_file(first_file)

    if fail_files:
        print("\n失败的文件列表:")
        for f in fail_files:
            print(f"  - {os.path.basename(f)}")

    print("=" * 80)


def verify_nifti_file(nii_file):
    """
    验证NIfTI文件是否正确保存
    """
    try:
        import nibabel as nib
        img = nib.load(nii_file)
        data = img.get_fdata()
        print(f"验证结果:")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  数值范围: [{np.min(data):.6f}, {np.max(data):.6f}]")
        print(f"  均值: {np.mean(data):.6f}")
        print(f"  标准差: {np.std(data):.6f}")
        print(f"  仿射变换:\n{img.affine}")
        return True
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return False


def test_single_file():
    """
    测试单个文件的转换（用于调试）
    """
    test_file = r"E:\3D-DSA\0.raw"
    output_dir = r"D:\med_data\biron\data1\raw_nii_test"

    print("测试模式：转换单个3D RAW文件 (保持原始精度)")
    print(f"测试文件: {test_file}")

    os.makedirs(output_dir, exist_ok=True)

    success, output_file = convert_single_3d_raw(
        test_file, output_dir,
        voxel_size=VOXEL_SIZE_MM,
        skip_existing=False
    )

    if success:
        print(f"\n详细验证转换结果:")
        verify_nifti_file(output_file)

    print(f"\n转换{'成功' if success else '失败'}")


if __name__ == "__main__":
    # 选择运行模式
    RUN_MODE = "batch"  # "batch" 批量处理所有文件, "test" 测试单个文件

    # 体素尺寸设置（从原始代码中 recon_voxel_size_mm = 0.5）
    # 256×256×256 的重建结果，每个体素0.5mm
    VOXEL_SIZE_MM = [0.5, 0.5, 0.5]

    if RUN_MODE == "batch":
        batch_convert_3d_raw()
    elif RUN_MODE == "test":
        test_single_file()
    else:
        print("请设置正确的运行模式: 'batch' 或 'test'")