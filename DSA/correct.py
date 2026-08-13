import os
import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm
import pickle
import shutil
from pathlib import Path
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingStats:
    """训练集统计信息计算和存储"""

    def __init__(self, standard_folder, cache_file=None):
        self.standard_folder = standard_folder
        self.cache_file = cache_file
        self.stats = None

    def compute_or_load(self, force_recompute=False):
        """计算或加载统计信息"""
        if self.cache_file and os.path.exists(self.cache_file) and not force_recompute:
            logger.info(f"加载缓存的统计信息: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.stats = pickle.load(f)
            return self.stats

        logger.info("计算标准数据统计信息...")
        self.stats = self._compute_stats()

        if self.cache_file:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.stats, f)
            logger.info(f"统计信息已缓存: {self.cache_file}")

        return self.stats

    def _compute_stats(self):
        """计算统计信息"""
        all_ap_pixels = []
        all_lat_pixels = []

        # 获取所有子文件夹
        case_folders = [d for d in os.listdir(self.standard_folder)
                        if os.path.isdir(os.path.join(self.standard_folder, d))]

        logger.info(f"找到 {len(case_folders)} 个标准病例")

        for folder_name in tqdm(case_folders, desc="处理标准数据"):
            folder_path = os.path.join(self.standard_folder, folder_name)

            # 读取AP图像
            ap_path = os.path.join(folder_path, "ap.nii.gz")
            if os.path.exists(ap_path):
                try:
                    ap_img = nib.load(ap_path).get_fdata().astype(np.float32)
                    all_ap_pixels.extend(ap_img.flatten())
                except Exception as e:
                    logger.warning(f"读取AP失败 {ap_path}: {e}")

            # 读取LAT图像
            lat_path = os.path.join(folder_path, "lat.nii.gz")
            if os.path.exists(lat_path):
                try:
                    lat_img = nib.load(lat_path).get_fdata().astype(np.float32)
                    all_lat_pixels.extend(lat_img.flatten())
                except Exception as e:
                    logger.warning(f"读取LAT失败 {lat_path}: {e}")

        # 转换为numpy数组
        all_ap_pixels = np.array(all_ap_pixels, dtype=np.float32)
        all_lat_pixels = np.array(all_lat_pixels, dtype=np.float32)

        logger.info(f"AP像素总数: {len(all_ap_pixels):,}")
        logger.info(f"LAT像素总数: {len(all_lat_pixels):,}")

        # 计算统计量
        stats = {}
        for name, pixels in [('ap', all_ap_pixels), ('lat', all_lat_pixels)]:
            if len(pixels) == 0:
                logger.warning(f"{name} 没有有效像素")
                continue

            # 移除极端值（前后1%）避免异常值影响
            p1, p99 = np.percentile(pixels, [1, 99])
            filtered = pixels[(pixels >= p1) & (pixels <= p99)]

            if len(filtered) == 0:
                filtered = pixels

            hist, bin_edges = np.histogram(filtered, bins=256, density=True)

            stats[name] = {
                'mean': float(np.mean(filtered)),
                'std': float(np.std(filtered)),
                'median': float(np.median(filtered)),
                'percentiles': [float(x) for x in np.percentile(filtered, [1, 5, 10, 25, 50, 75, 90, 95, 99])],
                'histogram': (hist.tolist(), bin_edges.tolist()),
                'min': float(np.min(filtered)),
                'max': float(np.max(filtered)),
                'n_samples': int(len(filtered))
            }

            logger.info(f"{name.upper()} - 均值: {stats[name]['mean']:.4f}, 标准差: {stats[name]['std']:.4f}")
            logger.info(f"{name.upper()} - 百分位数: {stats[name]['percentiles']}")

        return stats


class RealDSACorrector:
    """真实DSA矫正器 - 使用百分位数归一化"""

    def __init__(self, stats):
        self.stats = stats

    def correct_image(self, image, view='ap'):
        """
        矫正图像

        Args:
            image: 输入图像 (H, W)
            view: 'ap' 或 'lat'

        Returns:
            矫正后的图像
        """
        if view not in self.stats:
            logger.warning(f"未找到 {view} 的统计信息，直接返回原图")
            return image

        stats = self.stats[view]

        # 获取百分位数
        p1 = stats['percentiles'][0]  # 1%
        p99 = stats['percentiles'][-1]  # 99%

        # 裁剪到标准数据的百分位范围
        image_clipped = np.clip(image, p1, p99)

        # 归一化到0-1
        if p99 - p1 > 1e-8:
            normalized = (image_clipped - p1) / (p99 - p1)
        else:
            normalized = np.zeros_like(image_clipped)

        return np.clip(normalized, 0, 1).astype(np.float32)

    def correct_ap(self, image):
        """矫正AP图像"""
        return self.correct_image(image, 'ap')

    def correct_lat(self, image):
        """矫正LAT图像"""
        return self.correct_image(image, 'lat')


def process_directory(input_folder, output_folder, corrector, stats):
    """
    处理整个目录

    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        corrector: RealDSACorrector实例
        stats: 统计信息
    """
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有子文件夹
    case_folders = [d for d in os.listdir(input_folder)
                    if os.path.isdir(os.path.join(input_folder, d))]

    logger.info(f"找到 {len(case_folders)} 个病例文件夹")

    processed_count = 0
    skipped_count = 0

    for folder_name in tqdm(case_folders, desc="处理病例"):
        input_case_path = os.path.join(input_folder, folder_name)
        output_case_path = os.path.join(output_folder, folder_name)

        # 创建输出子文件夹
        os.makedirs(output_case_path, exist_ok=True)

        # 处理该文件夹中的所有.nii.gz文件
        nii_files = [f for f in os.listdir(input_case_path) if f.endswith('.nii.gz')]

        for nii_file in nii_files:
            input_path = os.path.join(input_case_path, nii_file)
            output_path = os.path.join(output_case_path, nii_file)

            try:
                # 读取图像
                img = nib.load(input_path)
                data = img.get_fdata().astype(np.float32)

                # 确定是AP还是LAT
                if nii_file.startswith('ap_'):
                    corrected_data = corrector.correct_ap(data)
                    logger.debug(f"  矫正AP: {folder_name}/{nii_file}")
                elif nii_file.startswith('lat_'):
                    corrected_data = corrector.correct_lat(data)
                    logger.debug(f"  矫正LAT: {folder_name}/{nii_file}")
                else:
                    # 如果不是以ap_或lat_开头，尝试从文件名判断
                    if 'ap' in nii_file.lower():
                        corrected_data = corrector.correct_ap(data)
                    elif 'lat' in nii_file.lower():
                        corrected_data = corrector.correct_lat(data)
                    else:
                        # 无法判断，直接复制
                        logger.warning(f"无法判断视角类型，直接复制: {nii_file}")
                        corrected_data = data

                # 保存矫正后的图像（保留原始仿射变换）
                corrected_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
                nib.save(corrected_img, output_path)

                processed_count += 1

            except Exception as e:
                logger.error(f"处理失败 {input_path}: {str(e)}")
                skipped_count += 1
                # 出错时复制原文件
                try:
                    shutil.copy2(input_path, output_path)
                except:
                    pass

    logger.info(f"处理完成: {processed_count} 个文件成功, {skipped_count} 个文件跳过")

    # 保存处理信息
    info_path = os.path.join(output_folder, '_processing_info.json')
    import json
    info = {
        'input_folder': input_folder,
        'output_folder': output_folder,
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'total_files': processed_count + skipped_count,
        'stats_summary': {
            'ap': {
                'mean': stats['ap']['mean'],
                'std': stats['ap']['std'],
                'percentiles': stats['ap']['percentiles']
            },
            'lat': {
                'mean': stats['lat']['mean'],
                'std': stats['lat']['std'],
                'percentiles': stats['lat']['percentiles']
            }
        }
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    # ============================================================
    # 配置
    # ============================================================
    STANDARD_FOLDER = r"D:\med_data\biron\data2\standard"
    INPUT_FOLDER = r"D:\med_data\biron\data2\output_nifti(b)"
    OUTPUT_FOLDER = r"D:\med_data\biron\data2\corrected"
    CACHE_FILE = r"D:\med_data\biron\data2\standard_stats.pkl"

    logger.info("=" * 70)
    logger.info("DSA图像百分位数归一化矫正")
    logger.info("=" * 70)
    logger.info(f"标准数据文件夹: {STANDARD_FOLDER}")
    logger.info(f"待矫正文件夹: {INPUT_FOLDER}")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")

    # 检查路径
    if not os.path.exists(STANDARD_FOLDER):
        logger.error(f"标准数据文件夹不存在: {STANDARD_FOLDER}")
        return

    if not os.path.exists(INPUT_FOLDER):
        logger.error(f"输入文件夹不存在: {INPUT_FOLDER}")
        return

    # ============================================================
    # 1. 计算标准数据的统计信息
    # ============================================================
    logger.info("\n" + "-" * 50)
    logger.info("步骤1: 计算标准数据统计信息")
    logger.info("-" * 50)

    stats_computer = TrainingStats(STANDARD_FOLDER, CACHE_FILE)
    stats = stats_computer.compute_or_load(force_recompute=False)

    if not stats:
        logger.error("统计信息计算失败")
        return

    if 'ap' not in stats or 'lat' not in stats:
        logger.error("统计信息不完整，缺少AP或LAT数据")
        return

    logger.info("\n统计信息摘要:")
    logger.info(f"  AP - 均值: {stats['ap']['mean']:.4f}, 标准差: {stats['ap']['std']:.4f}")
    logger.info(f"  AP - 百分位数: {stats['ap']['percentiles']}")
    logger.info(f"  LAT - 均值: {stats['lat']['mean']:.4f}, 标准差: {stats['lat']['std']:.4f}")
    logger.info(f"  LAT - 百分位数: {stats['lat']['percentiles']}")

    # ============================================================
    # 2. 创建矫正器
    # ============================================================
    logger.info("\n" + "-" * 50)
    logger.info("步骤2: 创建矫正器")
    logger.info("-" * 50)

    corrector = RealDSACorrector(stats)

    # ============================================================
    # 3. 处理所有图像
    # ============================================================
    logger.info("\n" + "-" * 50)
    logger.info("步骤3: 批量矫正图像")
    logger.info("-" * 50)

    process_directory(INPUT_FOLDER, OUTPUT_FOLDER, corrector, stats)

    # ============================================================
    # 4. 验证
    # ============================================================
    logger.info("\n" + "-" * 50)
    logger.info("步骤4: 验证矫正结果")
    logger.info("-" * 50)

    # 检查输出目录
    if os.path.exists(OUTPUT_FOLDER):
        case_folders = [d for d in os.listdir(OUTPUT_FOLDER)
                        if os.path.isdir(os.path.join(OUTPUT_FOLDER, d))]
        logger.info(f"输出目录包含 {len(case_folders)} 个病例文件夹")

        # 检查第一个文件
        if case_folders:
            sample_folder = os.path.join(OUTPUT_FOLDER, case_folders[0])
            sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.nii.gz')]
            if sample_files:
                sample_path = os.path.join(sample_folder, sample_files[0])
                sample_img = nib.load(sample_path)
                sample_data = sample_img.get_fdata()
                logger.info(f"样本文件: {case_folders[0]}/{sample_files[0]}")
                logger.info(f"  形状: {sample_data.shape}")
                logger.info(f"  范围: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
                logger.info(f"  均值: {sample_data.mean():.4f}, 标准差: {sample_data.std():.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("处理完成!")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info("=" * 70)


def visualize_comparison():
    """
    可视化对比矫正前后的图像（可选）
    """
    import matplotlib.pyplot as plt

    # 选择一个样本进行可视化
    sample_input = r"D:\med_data\biron\data2\output_nifti(b)\ANY_104_0\ap_0.nii.gz"
    sample_output = r"D:\med_data\biron\data2\corrected\ANY_104_0\ap_0.nii.gz"

    if not os.path.exists(sample_input) or not os.path.exists(sample_output):
        logger.warning("样本文件不存在，跳过可视化")
        return

    input_img = nib.load(sample_input).get_fdata()
    output_img = nib.load(sample_output).get_fdata()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title(f'矫正前\n范围: [{input_img.min():.3f}, {input_img.max():.3f}]')
    axes[0].axis('off')

    axes[1].imshow(output_img, cmap='gray')
    axes[1].set_title(f'矫正后\n范围: [{output_img.min():.3f}, {output_img.max():.3f}]')
    axes[1].axis('off')

    # 直方图对比
    axes[2].hist(input_img.flatten(), bins=50, alpha=0.5, label='矫正前', density=True)
    axes[2].hist(output_img.flatten(), bins=50, alpha=0.5, label='矫正后', density=True)
    axes[2].set_title('像素值分布对比')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(r"D:\med_data\biron\data2\correction_comparison.png", dpi=150)
    plt.close()
    logger.info("可视化对比已保存: correction_comparison.png")


if __name__ == "__main__":
    main()

    # 可选：生成可视化对比
    try:
        visualize_comparison()
    except Exception as e:
        logger.warning(f"可视化失败: {e}")
