import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage
import os
from datetime import datetime
import argparse

# ============================================================
# 参数配置（从原始代码中提取）
# ============================================================
NX = 512  # 图像宽度（列数）
NZ = 512  # 图像高度（行数）
IMAGE_INDEX = 1  # 如果是5通道文件，选择哪个通道（0-4）
DETECTOR_SIZE_MM = 280.0  # 探测器尺寸（毫米）
PIXEL_SPACING_MM = DETECTOR_SIZE_MM / NX  # 像素间距（毫米/像素）


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
        print(f"检测到5通道文件，选择通道 {image_index}")
        image = img5[image_index].astype(np.float32)

    elif data.size == expected_one:
        # 单通道文件：直接重塑
        print("检测到单通道文件")
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
        uint16类型的图像数据
    """
    # 获取图像的最小值和最大值
    min_val = np.min(image_data)
    max_val = np.max(image_data)

    print(f"原始数据范围: [{min_val:.6f}, {max_val:.6f}]")

    # 避免除零错误
    if max_val == min_val:
        print("警告：图像所有像素值相同")
        normalized = np.zeros_like(image_data, dtype=np.uint16)
    else:
        # 线性归一化到 [0, 65535]
        normalized = (image_data - min_val) / (max_val - min_val) * 65535.0
        normalized = np.clip(normalized, 0, 65535)
        normalized = normalized.astype(np.uint16)

    print(f"归一化后范围: [{np.min(normalized)}, {np.max(normalized)}]")

    return normalized, min_val, max_val


def create_dicom_from_raw(image_data, output_path, series_number=1, instance_number=1,
                          patient_id="Unknown", study_uid=None, series_uid=None):
    """
    从numpy数组创建DICOM文件

    参数:
        image_data: 2D numpy数组（uint16类型）
        output_path: 输出DICOM文件路径
        series_number: 序列号
        instance_number: 实例号
        patient_id: 患者ID
        study_uid: Study UID（如果为None则自动生成）
        series_uid: Series UID（如果为None则自动生成）
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
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    # 研究信息
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "MC-GPU Simulation"
    ds.AccessionNumber = ""

    # 序列信息
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDescription = "Projection Image"
    ds.Modality = "CT"

    # 图像信息
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = instance_number

    # 图像像素信息
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # 无符号整数

    # 像素数据
    ds.PixelData = image_data.tobytes()

    # 图像位置和方向（投影图像）
    ds.ImageType = ["ORIGINAL", "PRIMARY", "PROJECTION"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # 像素间距（毫米/像素）
    ds.PixelSpacing = [PIXEL_SPACING_MM, PIXEL_SPACING_MM]

    # 其他CT相关标签
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"

    # 保存DICOM文件
    ds.save_as(output_path, write_like_original=False)
    print(f"DICOM文件已保存: {output_path}")


def raw_to_dicom(input_file, output_dir, patient_id="MCGPU_Patient",
                 series_number=1, image_index=1, normalize=True):
    """
    将MC-GPU的.raw文件转换为DICOM格式

    参数:
        input_file: 输入的.raw文件路径
        output_dir: 输出目录
        patient_id: 患者ID
        series_number: 序列号
        image_index: 如果是5通道文件，选择哪个通道
        normalize: 是否归一化到0-65535范围
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.dcm")

    print("=" * 60)
    print(f"处理文件: {input_file}")

    try:
        # 1. 读取raw文件
        image_float = read_mcgpu_raw(input_file, image_index=image_index, nx=NX, nz=NZ)
        print(f"图像形状: {image_float.shape}")
        print(f"数据类型: {image_float.dtype}")

        # 2. 归一化到uint16
        if normalize:
            image_uint16, min_val, max_val = normalize_to_uint16(image_float)
        else:
            # 如果不归一化，直接转换（可能丢失信息）
            image_uint16 = np.clip(image_float, 0, 65535).astype(np.uint16)
            print("未归一化，直接转换为uint16")

        # 3. 创建DICOM文件
        create_dicom_from_raw(
            image_uint16,
            output_file,
            series_number=series_number,
            instance_number=1,
            patient_id=patient_id
        )

        print(f"✓ 转换成功!")
        print(f"  输出文件: {output_file}")
        print(f"  图像尺寸: {NX} x {NZ}")
        print(f"  像素间距: {PIXEL_SPACING_MM:.4f} mm")
        print("=" * 60)

        return output_file

    except Exception as e:
        print(f"✗ 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 配置参数
    input_file = r"E:\1e08\0\contrast_vessel_image.dat_0000.raw"
    output_dir = r"D:\med_data\biron\data1"

    # 可选：批量处理多个文件
    # 如果需要处理整个目录的所有.raw文件，可以使用下面的代码
    process_single = True  # 改为True处理单个文件，False批量处理

    if process_single:
        # 处理单个文件
        raw_to_dicom(
            input_file=input_file,
            output_dir=output_dir,
            patient_id="MCGPU_Patient_001",
            series_number=1,
            image_index=IMAGE_INDEX,
            normalize=False
        )
    else:
        # 批量处理：处理所有.raw文件
        input_dir = r"E:\1e08\0"
        raw_files = [f for f in os.listdir(input_dir) if f.endswith('.raw')]

        for i, raw_file in enumerate(sorted(raw_files)):
            input_path = os.path.join(input_dir, raw_file)
            raw_to_dicom(
                input_file=input_path,
                output_dir=output_dir,
                patient_id="MCGPU_Patient_001",
                series_number=1,
                image_index=IMAGE_INDEX,
                normalize=False
            )


if __name__ == "__main__":
    main()