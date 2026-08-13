import os
import pydicom
import numpy as np
from tqdm import tqdm
import logging
import re
from collections import defaultdict
from skimage.transform import resize
import nibabel as nib

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_pixel_data(pixel_array):
    """
    将DICOM像素数据归一化到0-1范围
    """
    # 转换为float类型
    pixel_array = pixel_array.astype(np.float32)

    # 获取全局最小值和最大值
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)

    logger.debug(f"归一化前 - 最小值: {pixel_min}, 最大值: {pixel_max}")

    if pixel_max > pixel_min:
        normalized = (pixel_array - pixel_min) / (pixel_max - pixel_min)
        logger.debug(f"归一化后 - 最小值: {np.min(normalized)}, 最大值: {np.max(normalized)}")
    else:
        # 如果所有像素值相同，返回全0数组
        logger.warning(f"所有像素值相同: {pixel_min}，返回全0数组")
        normalized = np.zeros_like(pixel_array, dtype=np.float32)

    return normalized


def resize_image(pixel_array, target_size=(256, 256)):
    """
    将图像调整为指定尺寸
    """
    # 记录原始形状
    original_shape = pixel_array.shape
    logger.debug(f"调整尺寸前形状: {original_shape}")

    if len(pixel_array.shape) == 3:
        # 形状为 [时间帧, 高度, 宽度]
        num_frames = pixel_array.shape[0]
        resized_frames = []

        for i in range(num_frames):
            frame = pixel_array[i, :, :]
            # 使用skimage的resize函数
            resized_frame = resize(frame, target_size, preserve_range=True, anti_aliasing=True)
            resized_frames.append(resized_frame)

        resized_array = np.stack(resized_frames, axis=0)
        logger.debug(f"调整尺寸后形状: {resized_array.shape}")
        return resized_array

    elif len(pixel_array.shape) == 2:
        # 2D图像直接调整大小
        resized_array = resize(pixel_array, target_size, preserve_range=True, anti_aliasing=True)
        logger.debug(f"调整尺寸后形状: {resized_array.shape}")
        return resized_array

    else:
        logger.warning(f"不支持的图像维度: {pixel_array.shape}")
        return pixel_array


def invert_pixel_values(pixel_array):
    """
    反转像素值 (简单的线性反转)
    对于归一化到0-1的数据：反转后为 1 - pixel_value
    """
    inverted = 1.0 - pixel_array
    return inverted


def process_image_for_saving(pixel_array, target_size=(256, 256), invert=False):
    """
    完整的图像处理流程：归一化 + 调整尺寸 + 可选像素值反转
    """
    logger.info(f"处理图像 - 原始形状: {pixel_array.shape}, 数据类型: {pixel_array.dtype}")
    logger.info(f"原始像素值范围: [{np.min(pixel_array)}, {np.max(pixel_array)}]")

    # 1. 调整图像尺寸
    resized_array = resize_image(pixel_array, target_size)

    # 2. 归一化到0-1
    normalized_array = normalize_pixel_data(resized_array)
    logger.info(f"归一化后像素值范围: [{np.min(normalized_array):.6f}, {np.max(normalized_array):.6f}]")

    # 3. 像素值反转（如果需要）
    if invert:
        inverted_array = invert_pixel_values(normalized_array)
        # 确保反转后的值仍在0-1范围内
        inverted_array = np.clip(inverted_array, 0.0, 1.0)
        logger.info(f"反转后像素值范围: [{np.min(inverted_array):.6f}, {np.max(inverted_array):.6f}]")
        return inverted_array.astype(np.float32)

    # 确保值在0-1范围内
    normalized_array = np.clip(normalized_array, 0.0, 1.0)
    return normalized_array.astype(np.float32)


def save_as_nifti(pixel_array, output_path, affine=None, invert_pixels=False):
    """
    将像素数组保存为NIFTI格式（.nii.gz）
    """
    try:
        # 处理图像：归一化 + 尺寸调整 + 可选反转
        processed_array = process_image_for_saving(pixel_array, (256, 256), invert_pixels)

        # 最终验证
        min_val = np.min(processed_array)
        max_val = np.max(processed_array)

        logger.info(f"最终像素值范围: [{min_val:.6f}, {max_val:.6f}]")

        if min_val < 0 or max_val > 1:
            logger.warning(f"像素值超出0-1范围: [{min_val}, {max_val}]，进行裁剪")
            processed_array = np.clip(processed_array, 0.0, 1.0)

        # 如果未提供仿射变换矩阵，使用单位矩阵
        if affine is None:
            affine = np.eye(4)

        # 创建NIFTI图像对象
        nifti_img = nib.Nifti1Image(processed_array, affine)

        # 添加元数据到头信息
        invert_str = "inverted" if invert_pixels else "original"
        nifti_img.header['descrip'] = f'DSA_min_projection_{invert_str}'.encode('utf-8')
        nifti_img.header['cal_min'] = 0.0
        nifti_img.header['cal_max'] = 1.0

        # 保存为.nii.gz格式
        nib.save(nifti_img, output_path)

        logger.info(f"✓ NIFTI文件保存成功: {output_path}")
        logger.info(f"  图像形状: {processed_array.shape}")
        logger.info(f"  数据类型: {processed_array.dtype}")
        logger.info(f"  像素值范围: [{np.min(processed_array):.6f}, {np.max(processed_array):.6f}]")

        return True

    except Exception as e:
        logger.error(f"保存NIFTI文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def parse_filename(filename):
    """
    解析文件名，支持两种格式：
    1. ANY_病例号_视角方向号_第几次投影
    2. ANY_病例号_检查序号_视角方向号_第几次投影

    返回包含病例号、视角方向号、投影次数的字典
    """
    # 尝试匹配第一种格式：ANY_病例号_视角方向号_第几次投影
    pattern1 = r'ANY_(\d+)_(view\d+)_(\d+)'
    match1 = re.match(pattern1, filename)

    if match1:
        case_id = match1.group(1)
        view_direction = match1.group(2)
        projection_num = match1.group(3)
        return {
            'case_id': case_id,
            'view_direction': view_direction,
            'projection_num': projection_num,
            'has_series_num': False,
            'base_name': f"ANY_{case_id}_{view_direction}_{projection_num}"
        }

    # 尝试匹配第二种格式：ANY_病例号_检查序号_视角方向号_第几次投影
    pattern2 = r'ANY_(\d+)_(\d+)_(view\d+)_(\d+)'
    match2 = re.match(pattern2, filename)

    if match2:
        case_id = match2.group(1)
        series_num = match2.group(2)
        view_direction = match2.group(3)
        projection_num = match2.group(4)
        return {
            'case_id': case_id,
            'view_direction': view_direction,
            'projection_num': projection_num,
            'has_series_num': True,
            'series_num': series_num,
            'base_name': f"ANY_{case_id}_{series_num}_{view_direction}_{projection_num}"
        }

    # 如果都不匹配，尝试用下划线分割的通用方法
    parts = filename.split('_')
    if len(parts) >= 4 and parts[0] == 'ANY':
        # 尝试判断是哪种格式
        if len(parts) == 4:
            # ANY_病例号_视角方向号_第几次投影
            case_id = parts[1]
            view_direction = parts[2]
            projection_num = parts[3]
            return {
                'case_id': case_id,
                'view_direction': view_direction,
                'projection_num': projection_num,
                'has_series_num': False,
                'base_name': f"ANY_{case_id}_{view_direction}_{projection_num}"
            }
        elif len(parts) == 5:
            # ANY_病例号_检查序号_视角方向号_第几次投影
            case_id = parts[1]
            series_num = parts[2]
            view_direction = parts[3]
            projection_num = parts[4]
            return {
                'case_id': case_id,
                'view_direction': view_direction,
                'projection_num': projection_num,
                'has_series_num': True,
                'series_num': series_num,
                'base_name': f"ANY_{case_id}_{series_num}_{view_direction}_{projection_num}"
            }

    return None


def extract_angle(ds):
    """
    从DICOM文件头提取角度信息 (0018,1510)
    """
    try:
        if hasattr(ds, 'PositionerPrimaryAngle'):
            angle = float(ds.PositionerPrimaryAngle)
            return angle
        elif (0x0018, 0x1510) in ds:
            angle = float(ds[0x0018, 0x1510].value)
            return angle
        else:
            logger.warning("未找到角度信息 (0018,1510)")
            return None
    except Exception as e:
        logger.error(f"提取角度时出错: {str(e)}")
        return None


def format_angle(angle):
    """
    格式化角度：如果是负数，则转换为360+角度
    """
    if angle is None:
        return "unknown"

    if angle < 0:
        formatted_angle = int(360 + angle)
    else:
        formatted_angle = int(angle)

    return str(formatted_angle)


def pair_images_by_projection(dicom_files_info):
    """
    根据病例号和投影次数配对view1和view2图像
    只要有配对的view1和view2就进行配对，支持多对多配对
    """
    # 第一步：按(病例号, 投影次数)分组
    grouped = defaultdict(list)

    for file_info in dicom_files_info:
        pair_key = (file_info['case_id'], file_info['projection_num'])
        grouped[pair_key].append(file_info)

    paired_images = []

    for (case_id, projection_num), files in grouped.items():
        # 分离view1和view2
        view1_files = [f for f in files if f['view_direction'] == 'view1']
        view2_files = [f for f in files if f['view_direction'] == 'view2']

        if not view1_files or not view2_files:
            logger.warning(f"病例 {case_id} 投影 {projection_num} 缺少配对的图像 "
                           f"(view1: {len(view1_files)}, view2: {len(view2_files)})")
            continue

        # 检查哪些文件有检查序号
        view1_with_series = [f for f in view1_files if f.get('has_series_num', False)]
        view2_with_series = [f for f in view2_files if f.get('has_series_num', False)]

        # 情况1：都有检查序号，按检查序号配对
        if view1_with_series and view2_with_series:
            # 按检查序号分组
            view1_by_series = {f['series_num']: f for f in view1_with_series}
            view2_by_series = {f['series_num']: f for f in view2_with_series}

            # 找相同检查序号的配对
            common_series = set(view1_by_series.keys()) & set(view2_by_series.keys())

            # 配对相同检查序号的
            for series_num in common_series:
                logger.info(f"✓ 配对: 病例 {case_id} 投影 {projection_num} 检查序号 {series_num}")
                paired_images.append({
                    'case_id': case_id,
                    'projection_num': projection_num,
                    'view1': view1_by_series[series_num],
                    'view2': view2_by_series[series_num],
                    'series_num': series_num
                })

            # 处理没有配对的view1
            unmatched_view1 = []
            for f in view1_files:
                if f.get('has_series_num', False) and f['series_num'] not in common_series:
                    unmatched_view1.append(f)
                elif not f.get('has_series_num', False):
                    unmatched_view1.append(f)

            # 处理没有配对的view2
            unmatched_view2 = []
            for f in view2_files:
                if f.get('has_series_num', False) and f['series_num'] not in common_series:
                    unmatched_view2.append(f)
                elif not f.get('has_series_num', False):
                    unmatched_view2.append(f)

            # 尝试配对剩余的view1和view2（按顺序）
            for v1 in unmatched_view1:
                if unmatched_view2:
                    v2 = unmatched_view2.pop(0)
                    series_label = "none"
                    if v1.get('has_series_num', False) and v2.get('has_series_num', False):
                        series_label = f"{v1['series_num']}_{v2['series_num']}"
                    elif v1.get('has_series_num', False):
                        series_label = v1['series_num']
                    elif v2.get('has_series_num', False):
                        series_label = v2['series_num']

                    logger.info(f"✓ 配对: 病例 {case_id} 投影 {projection_num} (混合配对)")
                    paired_images.append({
                        'case_id': case_id,
                        'projection_num': projection_num,
                        'view1': v1,
                        'view2': v2,
                        'series_num': series_label
                    })
                else:
                    if v1.get('has_series_num', False):
                        logger.warning(f"病例 {case_id} 投影 {projection_num} "
                                       f"view1 检查序号 {v1['series_num']} 没有配对的view2")
                    else:
                        logger.warning(f"病例 {case_id} 投影 {projection_num} "
                                       f"view1 (无检查序号) 没有配对的view2")

            # 如果还有剩余的view2，记录警告
            if unmatched_view2:
                for v2 in unmatched_view2:
                    if v2.get('has_series_num', False):
                        logger.warning(f"病例 {case_id} 投影 {projection_num} "
                                       f"view2 检查序号 {v2['series_num']} 没有配对的view1")
                    else:
                        logger.warning(f"病例 {case_id} 投影 {projection_num} "
                                       f"view2 (无检查序号) 没有配对的view1")

        else:
            # 情况2：没有检查序号或部分有检查序号
            # 进行全配对
            for v1 in view1_files:
                for v2 in view2_files:
                    series_label = "none"
                    if v1.get('has_series_num', False) and v2.get('has_series_num', False):
                        if v1['series_num'] == v2['series_num']:
                            series_label = v1['series_num']
                        else:
                            series_label = f"{v1['series_num']}_{v2['series_num']}"
                    elif v1.get('has_series_num', False):
                        series_label = v1['series_num']
                    elif v2.get('has_series_num', False):
                        series_label = v2['series_num']

                    logger.info(f"✓ 配对: 病例 {case_id} 投影 {projection_num}")
                    paired_images.append({
                        'case_id': case_id,
                        'projection_num': projection_num,
                        'view1': v1,
                        'view2': v2,
                        'series_num': series_label
                    })

    logger.info(f"成功配对 {len(paired_images)} 对图像")
    return paired_images


def process_paired_dsa_images(input_folder, output_folder, time_fraction=0.5,
                              target_size=(256, 256), invert_pixels=True):
    """
    处理配对的DSA图像
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"创建输出文件夹: {output_folder}")

    # 获取所有DICOM文件
    dicom_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.dcm') or ('.' not in file and not os.path.isdir(os.path.join(input_folder, file))):
            dicom_files.append(file)

    logger.info(f"找到 {len(dicom_files)} 个DICOM文件")

    if len(dicom_files) == 0:
        logger.error("在输入文件夹中未找到DICOM文件")
        return

    # 解析所有文件名
    files_info = []
    unsupported_files = []
    for filename in dicom_files:
        info = parse_filename(filename)
        if info:
            info['filename'] = filename
            info['full_path'] = os.path.join(input_folder, filename)
            files_info.append(info)
        else:
            unsupported_files.append(filename)

    logger.info(f"成功解析 {len(files_info)} 个文件名")

    if unsupported_files:
        logger.warning(f"无法解析 {len(unsupported_files)} 个文件名:")
        for f in unsupported_files[:5]:
            logger.warning(f"  - {f}")
        if len(unsupported_files) > 5:
            logger.warning(f"  ... 还有 {len(unsupported_files) - 5} 个文件")

    # 配对图像
    paired_images = pair_images_by_projection(files_info)
    logger.info(f"找到 {len(paired_images)} 对配对的图像")

    if len(paired_images) == 0:
        logger.error("未找到任何配对的图像对")
        return

    # 处理统计
    processed_count = 0
    skipped_count = 0

    logger.info(f"开始处理配对的DSA图像...")
    logger.info(f"目标图像尺寸: {target_size}")
    logger.info(f"像素值反转: {'启用' if invert_pixels else '禁用'}")
    logger.info(f"输出格式: NIFTI (.nii.gz)")

    for pair_info in tqdm(paired_images, desc="处理配对图像"):
        case_id = pair_info['case_id']
        projection_num = pair_info['projection_num']
        view1_info = pair_info['view1']
        view2_info = pair_info['view2']
        series_num = pair_info.get('series_num', '')

        try:
            logger.info(f"\n处理病例 {case_id}，投影 {projection_num}，配对序号 {series_num}")

            # 读取view1
            ds1 = pydicom.dcmread(view1_info['full_path'])
            pixel_array1 = ds1.pixel_array
            logger.info(f"View1 - 原始形状: {pixel_array1.shape}, 数据类型: {pixel_array1.dtype}")
            logger.info(f"View1 - 像素值范围: [{np.min(pixel_array1)}, {np.max(pixel_array1)}]")

            # 读取view2
            ds2 = pydicom.dcmread(view2_info['full_path'])
            pixel_array2 = ds2.pixel_array
            logger.info(f"View2 - 原始形状: {pixel_array2.shape}, 数据类型: {pixel_array2.dtype}")
            logger.info(f"View2 - 像素值范围: [{np.min(pixel_array2)}, {np.max(pixel_array2)}]")

            # 提取角度
            angle1 = extract_angle(ds1)
            angle2 = extract_angle(ds2)

            angle1_formatted = format_angle(angle1)
            angle2_formatted = format_angle(angle2)

            logger.info(f"View1角度: {angle1} -> {angle1_formatted}")
            logger.info(f"View2角度: {angle2} -> {angle2_formatted}")

            # 处理最小值投影
            if len(pixel_array1.shape) == 3:
                total_frames1 = pixel_array1.shape[0]
                frames_to_use1 = max(1, int(total_frames1 * time_fraction))
                selected_frames1 = pixel_array1[:frames_to_use1, :, :]
                min_projection1 = np.min(selected_frames1, axis=0)
                logger.info(f"View1 - 最小值投影形状: {min_projection1.shape}")
                logger.info(f"View1 - 投影后像素范围: [{np.min(min_projection1)}, {np.max(min_projection1)}]")
            else:
                logger.error(f"View1 不是3D序列")
                min_projection1 = pixel_array1

            if len(pixel_array2.shape) == 3:
                total_frames2 = pixel_array2.shape[0]
                frames_to_use2 = max(1, int(total_frames2 * time_fraction))
                selected_frames2 = pixel_array2[:frames_to_use2, :, :]
                min_projection2 = np.min(selected_frames2, axis=0)
                logger.info(f"View2 - 最小值投影形状: {min_projection2.shape}")
                logger.info(f"View2 - 投影后像素范围: [{np.min(min_projection2)}, {np.max(min_projection2)}]")
            else:
                logger.error(f"View2 不是3D序列")
                min_projection2 = pixel_array2

            # 生成输出文件名
            if series_num and series_num != 'none':
                # 使用配对时确定的系列号
                output_filename1 = f"ANY_{case_id}_{series_num}_view1_{projection_num}_{angle1_formatted}.nii.gz"
                output_filename2 = f"ANY_{case_id}_{series_num}_view2_{projection_num}_{angle2_formatted}.nii.gz"
            else:
                # 分别使用各自的检查序号（如果有）
                if view1_info.get('has_series_num'):
                    series_num1 = view1_info.get('series_num')
                    output_filename1 = f"ANY_{case_id}_{series_num1}_view1_{projection_num}_{angle1_formatted}.nii.gz"
                else:
                    output_filename1 = f"ANY_{case_id}_view1_{projection_num}_{angle1_formatted}.nii.gz"

                if view2_info.get('has_series_num'):
                    series_num2 = view2_info.get('series_num')
                    output_filename2 = f"ANY_{case_id}_{series_num2}_view2_{projection_num}_{angle2_formatted}.nii.gz"
                else:
                    output_filename2 = f"ANY_{case_id}_view2_{projection_num}_{angle2_formatted}.nii.gz"

            output_path1 = os.path.join(output_folder, output_filename1)
            output_path2 = os.path.join(output_folder, output_filename2)

            # 保存为NIFTI格式
            affine = np.eye(4)

            logger.info(f"保存 View1...")
            if save_as_nifti(min_projection1, output_path1, affine, invert_pixels):
                processed_count += 1
            else:
                skipped_count += 1
                continue

            logger.info(f"保存 View2...")
            if save_as_nifti(min_projection2, output_path2, affine, invert_pixels):
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            logger.error(f"处理病例 {case_id} 时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            skipped_count += 1
            continue

    # 输出处理总结
    logger.info("=" * 60)
    logger.info("配对图像处理完成!")
    logger.info(f"成功处理: {processed_count} 个图像")
    logger.info(f"跳过/失败: {skipped_count} 个图像")
    logger.info(f"总计配对: {len(paired_images)} 对")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"F:\view"
    output_folder = r"F:\view\output_nifti1"

    # 设置参数
    time_fraction = 0.8
    target_size = (256, 256)
    invert_pixels = True

    logger.info("开始处理配对的DSA图像序列...")
    logger.info(f"输入文件夹: {input_folder}")
    logger.info(f"输出文件夹: {output_folder}")
    logger.info(f"支持的文件命名格式:")
    logger.info(f"  1. ANY_病例号_视角方向号_第几次投影")
    logger.info(f"  2. ANY_病例号_检查序号_视角方向号_第几次投影")
    logger.info(f"参数设置:")
    logger.info(f"  - 时间截取比例: {time_fraction * 100:.0f}%")
    logger.info(f"  - 目标图像尺寸: {target_size}")
    logger.info(f"  - 像素值反转: {'启用' if invert_pixels else '禁用'}")

    try:
        process_paired_dsa_images(input_folder, output_folder, time_fraction, target_size, invert_pixels)
        logger.info("处理完成！")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())