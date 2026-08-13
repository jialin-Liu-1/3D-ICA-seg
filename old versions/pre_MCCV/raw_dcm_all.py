import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage
import os
from datetime import datetime
import glob
import re
from tqdm import tqdm  # 用于显示进度条，如果没有这个库可以注释掉

# ============================================================
# 参数配置（从原始代码中提取）
# ============================================================
NX = 512  # 图像宽度（列数）
NZ = 512  # 图像高度（行数）
IMAGE_INDEX = 1  # 如果是5通道文件，选择哪个通道（0-4）
DETECTOR_SIZE_MM = 280.0  # 探测器尺寸（毫米）
PIXEL_SPACING_MM = DETECTOR_SIZE_MM / NX  # 像素间距（毫米/像素）

# 输入输出路径
INPUT_BASE = r"E:\1e08"
OUTPUT_BASE = r"D:\med_data\biron\data1\raw_dcm"

# 是否跳过已存在的文件（避免重复转换）
SKIP_EXISTING = True


def read_mcgpu_raw(file_path, image_index=1, nx=512, nz=512):
    """
    读取MC-GPU生成的.raw投影文件

    参数:
        file_path: .raw文件路径
        image_index: 如果是5通道文件，选择哪个通道（0-4）
        nx: 图像宽度（列数）
        nz: 图像高度（行数）

    返回:
        numpy数组，形状为[nz, nx]，dtype=float32
    """
    # 读取原始数据
    data = np.fromfile(file_path, dtype=np.float32)

    expected_one = nx * nz
    expected_five = 5 * nx * nz

    if data.size == expected_five:
        # 5通道文件：重塑并选择指定通道
        img5 = data.reshape(5, nz, nx)
        image = img5[image_index].astype(np.float32)

    elif data.size == expected_one:
        # 单通道文件：直接重塑
        image = data.reshape(nz, nx).astype(np.float32)

    else:
        raise ValueError(
            f"文件大小异常: {data.size} 个值\n"
            f"期望: {expected_one} 或 {expected_five} 个值\n"
            f"文件: {file_path}"
        )

    return image


def normalize_to_uint16(image_data):
    """
    将float32图像数据归一化到uint16范围（0-65535）

    参数:
        image_data: float32类型的图像数据

    返回:
        uint16类型的图像数据，以及原始的最小值和最大值
    """
    # 获取图像的最小值和最大值
    min_val = np.min(image_data)
    max_val = np.max(image_data)

    # 避免除零错误
    if max_val == min_val:
        print(f"  警告：图像所有像素值相同 ({min_val:.6f})")
        normalized = np.zeros_like(image_data, dtype=np.uint16)
    else:
        # 线性归一化到 [0, 65535]
        normalized = (image_data - min_val) / (max_val - min_val) * 65535.0
        normalized = np.clip(normalized, 0, 65535)
        normalized = normalized.astype(np.uint16)

    return normalized, min_val, max_val


def extract_projection_number(file_path):
    """
    从文件名提取投影编号
    例如：contrast_vessel_image.dat_0003.raw -> 3
         contrast_vessel_image.dat.raw -> 0
    """
    name = os.path.basename(file_path)

    # 匹配 dat_数字.raw 格式
    match = re.search(r"dat_(\d+)\.raw$", name)
    if match:
        return int(match.group(1))

    # 匹配 .dat.raw 格式
    if name.endswith(".dat.raw"):
        return 0

    return None


def create_dicom_from_raw(image_data, output_path, patient_id="Unknown",
                          study_uid=None, series_uid=None, instance_number=1,
                          projection_number=None):
    """
    从numpy数组创建DICOM文件

    参数:
        image_data: 2D numpy数组（uint16类型）
        output_path: 输出DICOM文件路径
        patient_id: 患者ID
        study_uid: Study UID（如果为None则自动生成）
        series_uid: Series UID（如果为None则自动生成）
        instance_number: 实例号
        projection_number: 投影编号（用于标记）
    """

    # 获取图像尺寸
    rows, cols = image_data.shape

    # 生成UID
    if study_uid is None:
        study_uid = generate_uid()
    if series_uid is None:
        series_uid = generate_uid()

    # 创建文件元数据
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # 创建数据集
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # 患者信息
    ds.PatientName = patient_id
    ds.PatientID = patient_id

    # 研究信息
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "MC-GPU Simulation"

    # 序列信息
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = f"Projection Images - Case {patient_id}"
    ds.Modality = "CT"

    # 图像信息
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = instance_number

    # 如果有投影编号，添加到图像注释中
    if projection_number is not None:
        ds.ImageComments = f"Projection {projection_number}"

    # 图像像素信息
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # 无符号整数

    # 像素数据
    ds.PixelData = image_data.tobytes()

    # 图像类型（投影图像）
    ds.ImageType = ["ORIGINAL", "PRIMARY", "PROJECTION"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # 像素间距（毫米/像素）
    ds.PixelSpacing = [PIXEL_SPACING_MM, PIXEL_SPACING_MM]

    # CT相关标签
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.WindowCenter = "32768"
    ds.WindowWidth = "65536"

    # 保存DICOM文件
    ds.save_as(output_path, write_like_original=False)


def find_raw_files(case_folder):
    """
    查找病例文件夹中的所有.raw文件

    参数:
        case_folder: 病例文件夹路径

    返回:
        raw文件路径列表，按投影编号排序
    """
    # 查找所有.raw文件
    raw_files = glob.glob(os.path.join(case_folder, "*.raw"))

    # 过滤掉可能不需要的文件（只保留contrast_vessel_image相关）
    # 如果你想包含所有.raw文件，可以注释掉下面的过滤
    raw_files = [f for f in raw_files if "contrast_vessel_image" in os.path.basename(f)]

    if len(raw_files) == 0:
        # 如果没有contrast_vessel_image，则返回所有.raw文件
        raw_files = glob.glob(os.path.join(case_folder, "*.raw"))

    # 按投影编号排序
    raw_files_with_numbers = []
    for f in raw_files:
        num = extract_projection_number(f)
        if num is not None:
            raw_files_with_numbers.append((num, f))

    # 按投影编号排序
    raw_files_with_numbers.sort(key=lambda x: x[0])
    sorted_files = [f for num, f in raw_files_with_numbers]

    return sorted_files


def process_case(case_folder, output_base, skip_existing=True):
    """
    处理单个病例文件夹

    参数:
        case_folder: 病例文件夹路径
        output_base: 输出基路径
        skip_existing: 是否跳过已存在的文件

    返回:
        (成功转换的文件数, 失败的文件数)
    """
    case_name = os.path.basename(case_folder)
    output_case_dir = os.path.join(output_base, case_name)

    # 创建输出目录
    os.makedirs(output_case_dir, exist_ok=True)

    # 查找所有raw文件
    raw_files = find_raw_files(case_folder)

    if len(raw_files) == 0:
        print(f"  警告：在 {case_folder} 中没有找到.raw文件")
        return 0, 0

    print(f"\n处理病例 {case_name}: 找到 {len(raw_files)} 个.raw文件")

    success_count = 0
    fail_count = 0

    # 为每个病例生成固定的UID（这样同一病例的所有图像共享Study和Series）
    study_uid = generate_uid()
    series_uid = generate_uid()

    for idx, raw_file in enumerate(raw_files, start=1):
        # 生成输出文件名（将.raw扩展名改为.dcm）
        base_name = os.path.splitext(os.path.basename(raw_file))[0]
        output_file = os.path.join(output_case_dir, f"{base_name}.dcm")

        # 检查是否跳过已存在的文件
        if skip_existing and os.path.exists(output_file):
            print(f"  跳过已存在的文件: {base_name}.dcm")
            success_count += 1  # 算作成功（已存在）
            continue

        try:
            # 读取raw文件
            image_float = read_mcgpu_raw(raw_file, image_index=IMAGE_INDEX, nx=NX, nz=NZ)

            # 归一化到uint16
            image_uint16, min_val, max_val = normalize_to_uint16(image_float)

            # 获取投影编号
            proj_number = extract_projection_number(raw_file)

            # 创建DICOM文件
            create_dicom_from_raw(
                image_uint16,
                output_file,
                patient_id=case_name,  # 使用病例文件夹名作为患者ID
                study_uid=study_uid,
                series_uid=series_uid,
                instance_number=idx,
                projection_number=proj_number
            )

            print(f"  ✓ 转换成功: {base_name}.dcm (投影编号: {proj_number}, 范围: [{min_val:.4f}, {max_val:.4f}])")
            success_count += 1

        except Exception as e:
            print(f"  ✗ 转换失败: {base_name}.raw - {str(e)}")
            fail_count += 1
            continue

    return success_count, fail_count


def batch_process():
    """
    批量处理所有病例文件夹
    """
    print("=" * 80)
    print("MC-GPU RAW 到 DICOM 批量转换工具")
    print("=" * 80)
    print(f"输入基路径: {INPUT_BASE}")
    print(f"输出基路径: {OUTPUT_BASE}")
    print(f"图像尺寸: {NX} x {NZ}")
    print(f"像素间距: {PIXEL_SPACING_MM:.4f} mm")
    print(f"跳过已存在文件: {SKIP_EXISTING}")
    print("=" * 80)

    # 检查输入路径是否存在
    if not os.path.exists(INPUT_BASE):
        print(f"错误：输入路径不存在: {INPUT_BASE}")
        return

    # 查找所有病例文件夹
    case_folders = []
    for item in os.listdir(INPUT_BASE):
        item_path = os.path.join(INPUT_BASE, item)
        if os.path.isdir(item_path):
            # 尝试判断是否为数字文件夹（可根据需要调整）
            case_folders.append(item_path)

    # 排序病例文件夹（按数字排序）
    case_folders.sort(key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else x)

    print(f"\n找到 {len(case_folders)} 个病例文件夹")
    print("开始批量转换...\n")

    # 统计信息
    total_success = 0
    total_fail = 0
    total_cases = len(case_folders)

    # 使用tqdm显示进度条（如果没有tqdm库，可以注释掉并使用普通循环）
    try:
        from tqdm import tqdm
        pbar = tqdm(case_folders, desc="处理进度", unit="病例")
        for case_folder in pbar:
            case_name = os.path.basename(case_folder)
            pbar.set_description(f"处理病例 {case_name}")
            success, fail = process_case(case_folder, OUTPUT_BASE, SKIP_EXISTING)
            total_success += success
            total_fail += fail
    except ImportError:
        # 如果没有tqdm，使用普通循环
        for i, case_folder in enumerate(case_folders, 1):
            case_name = os.path.basename(case_folder)
            print(f"[{i}/{total_cases}] 处理病例 {case_name}...")
            success, fail = process_case(case_folder, OUTPUT_BASE, SKIP_EXISTING)
            total_success += success
            total_fail += fail

    # 输出统计结果
    print("\n" + "=" * 80)
    print("批量转换完成！")
    print("=" * 80)
    print(f"处理病例数: {total_cases}")
    print(f"成功转换文件数: {total_success}")
    print(f"失败文件数: {total_fail}")
    print(f"输出目录: {OUTPUT_BASE}")
    print("=" * 80)


def test_single_case():
    """
    测试单个病例的转换（用于调试）
    """
    test_case = r"E:\1e08\0"  # 测试用的病例文件夹
    output_base = r"D:\med_data\biron\data1\raw_dcm"

    print("测试模式：处理单个病例")
    print(f"测试病例: {test_case}")

    success, fail = process_case(test_case, output_base, SKIP_EXISTING)
    print(f"\n转换完成: 成功 {success}, 失败 {fail}")


if __name__ == "__main__":
    # 选择运行模式
    RUN_MODE = "batch"  # "batch" 批量处理所有病例, "test" 测试单个病例

    if RUN_MODE == "batch":
        batch_process()
    elif RUN_MODE == "test":
        test_single_case()
    else:
        print("请设置正确的运行模式: 'batch' 或 'test'")