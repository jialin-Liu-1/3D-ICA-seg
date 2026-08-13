import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def remove_edge_padding_background(image_data, extend_pixels=10):
    """
    移除图像边缘处像素值恰好等于1的补零背景区域
    专用于反转DSA图像：无效补零区域像素值=1，血管区域像素值>1

    Parameters:
    image_data: 2D numpy array (像素值范围0-1)
    extend_pixels: 背景扩展像素数（向内部扩展的像素数）

    Returns:
    processed_image: 处理后的图像
    mask: 背景掩码 (1=保留, 0=背景)
    """
    height, width = image_data.shape

    # 图像的最小值（补零区域反转后为1，但可能有其他背景）
    # 实际上，像素值为1的区域就是要移除的补零背景
    min_val = np.min(image_data)

    logger.debug(f"图像最小值: {min_val:.6f}")

    # 初始化掩码：1=保留，0=背景（要移除）
    mask = np.ones((height, width), dtype=np.uint8)

    # 精确检测像素值 == 1 的区域（允许浮点误差）
    bg_threshold_low = 0.999
    bg_threshold_high = 1.001

    # ===== 1. 从顶部向下搜索 =====
    for j in range(width):
        for i in range(height):
            if bg_threshold_low <= image_data[i, j] <= bg_threshold_high:
                # 找到像素值≈1的像素，向上扩展
                extend_position = max(i - extend_pixels, 0)
                mask[:extend_position, j] = 0
                break

    # ===== 2. 从底部向上搜索 =====
    for j in range(width):
        for i in range(height - 1, -1, -1):
            if bg_threshold_low <= image_data[i, j] <= bg_threshold_high:
                # 找到像素值≈1的像素，向下扩展
                extend_position = min(i + extend_pixels, height)
                mask[extend_position:, j] = 0
                break

    # ===== 3. 从左向右搜索 =====
    for i in range(height):
        for j in range(width):
            if bg_threshold_low <= image_data[i, j] <= bg_threshold_high:
                extend_position = max(j - extend_pixels, 0)
                mask[i, :extend_position] = 0
                break

    # ===== 4. 从右向左搜索 =====
    for i in range(height):
        for j in range(width - 1, -1, -1):
            if bg_threshold_low <= image_data[i, j] <= bg_threshold_high:
                extend_position = min(j + extend_pixels, width)
                mask[i, extend_position:] = 0
                break

    # ===== 处理图像：将背景区域替换为最小值 =====
    processed_image = image_data.copy()
    processed_image[mask == 0] = min_val

    # 统计
    bg_ratio = np.sum(mask == 0) / (height * width) * 100
    pixel1_count = np.sum((bg_threshold_low <= image_data) & (image_data <= bg_threshold_high))
    pixel1_removed = np.sum((bg_threshold_low <= image_data) & (image_data <= bg_threshold_high) & (mask == 0))

    logger.debug(f"图像中像素值≈1的像素总数: {pixel1_count}")
    logger.debug(f"被移除的像素值≈1的像素数: {pixel1_removed}")
    logger.debug(f"背景移除比例: {bg_ratio:.1f}%")

    return processed_image, mask


def remove_edge_padding_binary(image_data, extend_pixels=15):
    """
    二值化方法：直接创建像素值==1的掩码
    更精确地检测补零区域
    """
    height, width = image_data.shape
    min_val = np.min(image_data)

    # 精确检测像素值==1的区域
    bg_threshold_low = 0.999
    bg_threshold_high = 1.001

    # 直接创建背景掩码：像素值≈1的区域
    bg_mask = ((image_data >= bg_threshold_low) & (image_data <= bg_threshold_high)).astype(np.uint8)

    # 初始化最终掩码：1=保留，0=背景
    mask = np.ones((height, width), dtype=np.uint8)

    # ===== 从顶部向下：找到第一个非背景行 =====
    first_non_bg_row = 0
    for i in range(height):
        if np.any(~((image_data[i, :] >= bg_threshold_low) & (image_data[i, :] <= bg_threshold_high))):
            first_non_bg_row = i
            break

    # 标记顶部背景区域（扩展到first_non_bg_row + extend_pixels）
    if first_non_bg_row > 0:
        end_row = min(first_non_bg_row + extend_pixels, height)
        mask[:end_row, :] = 0

    # ===== 从底部向上：找到第一个非背景行 =====
    last_non_bg_row = height - 1
    for i in range(height - 1, -1, -1):
        if np.any(~((image_data[i, :] >= bg_threshold_low) & (image_data[i, :] <= bg_threshold_high))):
            last_non_bg_row = i
            break

    # 标记底部背景区域
    if last_non_bg_row < height - 1:
        start_row = max(last_non_bg_row - extend_pixels, 0)
        mask[start_row:, :] = 0

    # ===== 从左向右：找到第一个非背景列 =====
    first_non_bg_col = 0
    for j in range(width):
        if np.any(~((image_data[:, j] >= bg_threshold_low) & (image_data[:, j] <= bg_threshold_high))):
            first_non_bg_col = j
            break

    # 标记左侧背景区域
    if first_non_bg_col > 0:
        end_col = min(first_non_bg_col + extend_pixels, width)
        mask[:, :end_col] = 0

    # ===== 从右向左：找到第一个非背景列 =====
    last_non_bg_col = width - 1
    for j in range(width - 1, -1, -1):
        if np.any(~((image_data[:, j] >= bg_threshold_low) & (image_data[:, j] <= bg_threshold_high))):
            last_non_bg_col = j
            break

    # 标记右侧背景区域
    if last_non_bg_col < width - 1:
        start_col = max(last_non_bg_col - extend_pixels, 0)
        mask[:, start_col:] = 0

    # ===== 处理图像：将背景区域替换为最小值 =====
    processed_image = image_data.copy()
    processed_image[mask == 0] = min_val

    # 统计
    bg_ratio = np.sum(mask == 0) / (height * width) * 100
    logger.debug(f"背景移除比例: {bg_ratio:.1f}%")

    return processed_image, mask


def process_nifti_files(input_folder, output_folder, method='binary',
                        extend_pixels=15, show_comparison=False):
    """
    批量处理NIFTI文件，移除边缘补零背景
    """
    os.makedirs(output_folder, exist_ok=True)

    case_folders = [d for d in os.listdir(input_folder)
                    if os.path.isdir(os.path.join(input_folder, d))]

    logger.info(f"找到 {len(case_folders)} 个病例文件夹")
    logger.info(f"处理方法: {method}")
    logger.info(f"扩展像素: {extend_pixels}")
    logger.info(f"背景检测: 像素值 ≈ 1 (补零区域)")

    processed_count = 0
    skipped_count = 0

    for folder_name in tqdm(case_folders, desc="处理病例"):
        input_case_path = os.path.join(input_folder, folder_name)
        output_case_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_case_path, exist_ok=True)

        nii_files = [f for f in os.listdir(input_case_path) if f.endswith('.nii.gz')]

        for nii_file in nii_files:
            input_path = os.path.join(input_case_path, nii_file)
            output_path = os.path.join(output_case_path, nii_file)

            try:
                img = nib.load(input_path)
                data = img.get_fdata().astype(np.float32)

                # 检查数据范围
                logger.debug(f"文件: {nii_file}, 像素范围: [{data.min():.6f}, {data.max():.6f}]")

                if len(data.shape) == 2:
                    # 2D图像
                    if method == 'binary':
                        processed_data, mask = remove_edge_padding_binary(
                            data, extend_pixels
                        )
                    else:
                        processed_data, mask = remove_edge_padding_background(
                            data, extend_pixels
                        )

                    processed_img = nib.Nifti1Image(processed_data, img.affine, img.header)
                    nib.save(processed_img, output_path)
                    processed_count += 1

                    # 显示对比图
                    if show_comparison and processed_count <= 3:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

                        # 原始图像
                        axes[0, 0].imshow(data, cmap='gray', vmin=0, vmax=1)
                        axes[0, 0].set_title(f'原始图像\n范围: [{data.min():.4f}, {data.max():.4f}]')
                        axes[0, 0].axis('off')

                        # 标记像素值=1的区域（红色显示）
                        pixel1_mask = (data >= 0.999) & (data <= 1.001)
                        axes[0, 1].imshow(data, cmap='gray', vmin=0, vmax=1)
                        axes[0, 1].imshow(pixel1_mask, cmap='Reds', alpha=0.5)
                        axes[0, 1].set_title(f'像素值≈1区域 (红色)\n数量: {np.sum(pixel1_mask)}')
                        axes[0, 1].axis('off')

                        # 背景掩码
                        axes[1, 0].imshow(mask, cmap='gray')
                        axes[1, 0].set_title(
                            f'背景掩码\n(白色=保留, 黑色=背景)\n背景: {np.sum(mask == 0) / mask.size * 100:.1f}%')
                        axes[1, 0].axis('off')

                        # 处理后图像
                        axes[1, 1].imshow(processed_data, cmap='gray', vmin=0, vmax=1)
                        axes[1, 1].set_title(f'处理后\n范围: [{processed_data.min():.4f}, {processed_data.max():.4f}]')
                        axes[1, 1].axis('off')

                        plt.tight_layout()
                        plt.show()

                elif len(data.shape) == 3:
                    # 3D图像，逐帧处理
                    processed_frames = []
                    for t in range(data.shape[0]):
                        frame = data[t, :, :]
                        if method == 'binary':
                            processed_frame, _ = remove_edge_padding_binary(
                                frame, extend_pixels
                            )
                        else:
                            processed_frame, _ = remove_edge_padding_background(
                                frame, extend_pixels
                            )
                        processed_frames.append(processed_frame)

                    processed_data = np.stack(processed_frames, axis=0)
                    processed_img = nib.Nifti1Image(processed_data, img.affine, img.header)
                    nib.save(processed_img, output_path)
                    processed_count += 1
                else:
                    logger.warning(f"不支持的维度: {data.shape}")
                    skipped_count += 1

            except Exception as e:
                logger.error(f"处理失败 {input_path}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                skipped_count += 1
                continue

    logger.info(f"处理完成: {processed_count} 成功, {skipped_count} 跳过")


def main():
    INPUT_FOLDER = r"D:\med_data\biron\data2\output_nifti(1)"
    OUTPUT_FOLDER = r"D:\med_data\biron\data2\output_nifti(b)"

    # 方法选择:
    # 'binary' - 二值化检测（推荐），直接检测像素值≈1的区域
    # 'edge' - 边缘搜索方法
    METHOD = 'binary'

    # 扩展像素：向内部扩展多少个像素
    EXTEND_PIXELS = 10

    SHOW_COMPARISON = True

    logger.info("=" * 70)
    logger.info("移除反转DSA边缘补零背景")
    logger.info("=" * 70)
    logger.info(f"输入文件夹: {INPUT_FOLDER}")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info(f"处理方法: {METHOD}")
    logger.info(f"扩展像素: {EXTEND_PIXELS}")
    logger.info(f"背景检测: 像素值 ≈ 1 (补零区域，反转后由0变为1)")

    if not os.path.exists(INPUT_FOLDER):
        logger.error(f"输入文件夹不存在: {INPUT_FOLDER}")
        return

    process_nifti_files(
        INPUT_FOLDER, OUTPUT_FOLDER, METHOD,
        EXTEND_PIXELS, SHOW_COMPARISON
    )

    logger.info("=" * 70)
    logger.info("处理完成!")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()