"""
独立测试脚本 - 加载训练好的模型并在测试集上推理
适配新数据格式：从output_nifti读取ap和lat文件，计算夹角
只进行前向传播和保存结果，不计算评价指标
"""

import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm
import time
import warnings
import re

warnings.filterwarnings('ignore')

# 导入模型 - 修改为您的模型路径
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model.x2shap_256_any_new import X2Shape

# ============================================================
# 配置
# ============================================================

# 数据路径
TEST_DATA_DIR = Path('/mnt/d/med_data/biron/data2/corrected')  # 包含AP和LAT图像的目录
MODEL_PATH = Path('/mnt/d/med_data/biron/model_new/20260805_303/best_model.pth')
OUTPUT_DIR = Path('/mnt/d/med_data/biron/model_new/20260805_303/test_DSA_32_C')

# 模型参数 (必须与训练时一致)
IMG_SIZE = 256
VBP_VOLUME_SIZE = 64
ENCODER_CHANNELS = 32
VBP_OUTPUT_CHANNELS = 64
DIMS_3D = [32, 64, 128, 256]

# 图像尺寸
TARGET_SIZE = (256, 256)


def normalize_angle(angle):
    """将角度归一化到 -180° 到 180° 之间"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


# ============================================================
# 1. 数据集类 - 直接从文件读取
# ============================================================

class TestDataset(Dataset):
    """测试数据集 - 直接从output_nifti读取AP和LAT图像"""

    def __init__(self, data_dir, target_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.case_list = []
        self.case_info = {}  # 存储每个病例的详细信息

        # 扫描数据目录
        self._scan_data()

    def _scan_data(self):
        """扫描数据目录，查找所有病例"""
        if not self.data_dir.exists():
            print(f"错误: 数据目录不存在: {self.data_dir}")
            return

        for folder_name in self.data_dir.iterdir():
            if not folder_name.is_dir():
                continue

            # 解析文件夹名称: ANY_病例号_投影次数 或 ANY_病例号_检查序号_投影次数
            parts = folder_name.name.split('_')
            if len(parts) < 2 or parts[0] != 'ANY':
                continue

            # 查找ap和lat文件
            ap_files = []
            lat_files = []

            for file in folder_name.iterdir():
                if file.name.endswith('.nii.gz'):
                    if file.name.startswith('ap_'):
                        ap_files.append(file)
                    elif file.name.startswith('lat_'):
                        lat_files.append(file)

            # 检查是否有ap和lat文件
            if not ap_files or not lat_files:
                print(f"  跳过 {folder_name.name}: 缺少AP或LAT文件 (AP: {len(ap_files)}, LAT: {len(lat_files)})")
                continue

            # 提取角度信息
            for ap_file in ap_files:
                ap_match = re.match(r'ap_(\d+)\.nii\.gz', ap_file.name)
                if not ap_match:
                    continue
                ap_angle = int(ap_match.group(1))

                for lat_file in lat_files:
                    lat_match = re.match(r'lat_(\d+)\.nii\.gz', lat_file.name)
                    if not lat_match:
                        continue
                    lat_angle = int(lat_match.group(1))

                    # 计算夹角 (LAT相对于AP)
                    angle_diff = lat_angle - ap_angle
                    angle_diff_norm = normalize_angle(angle_diff)

                    # 解析病例信息
                    if len(parts) == 3:
                        # ANY_病例号_投影次数
                        case_id = parts[1]
                        projection = parts[2]
                        series_num = None
                        case_key = f"{case_id}_{projection}"
                    elif len(parts) == 4:
                        # ANY_病例号_检查序号_投影次数
                        case_id = parts[1]
                        series_num = parts[2]
                        projection = parts[3]
                        case_key = f"{case_id}_{series_num}_{projection}"
                    else:
                        print(f"  跳过 {folder_name.name}: 无法解析文件夹名称")
                        continue

                    # 存储病例信息 - 将Path对象转换为字符串
                    self.case_info[case_key] = {
                        'case_id': case_id,
                        'series_num': series_num,
                        'projection': projection,
                        'folder_name': folder_name.name,
                        'folder_path': str(folder_name),  # 转换为字符串
                        'ap_angle': ap_angle,
                        'lat_angle': lat_angle,
                        'angle_diff': angle_diff_norm,  # LAT相对于AP的夹角
                        'ap_path': str(ap_file),  # 转换为字符串
                        'lat_path': str(lat_file),  # 转换为字符串
                        'ap_file': ap_file.name,
                        'lat_file': lat_file.name,
                    }

                    self.case_list.append(case_key)
                    print(f"  发现: {folder_name.name} - AP角度: {ap_angle}°, LAT角度: {lat_angle}°, 夹角: {angle_diff_norm}°")

        print(f"\n总共找到 {len(self.case_list)} 个有效病例")

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case_key = self.case_list[idx]
        info = self.case_info[case_key]

        # 加载AP图像 - 使用字符串路径
        ap_path = Path(info['ap_path'])
        ap_nii = nib.load(ap_path)
        ap_data = ap_nii.get_fdata().astype(np.float32)

        # 加载LAT图像
        lat_path = Path(info['lat_path'])
        lat_nii = nib.load(lat_path)
        lat_data = lat_nii.get_fdata().astype(np.float32)

        # 调整尺寸到统一大小


        # 归一化到0-1
        ap_min, ap_max = ap_data.min(), ap_data.max()
        if ap_max > ap_min:
            ap_data = (ap_data - ap_min) / (ap_max - ap_min + 1e-8)
        else:
            ap_data = np.zeros_like(ap_data)

        lat_min, lat_max = lat_data.min(), lat_data.max()
        if lat_max > lat_min:
            lat_data = (lat_data - lat_min) / (lat_max - lat_min + 1e-8)
        else:
            lat_data = np.zeros_like(lat_data)

        # 转换为tensor (B, C, H, W)
        ap_tensor = torch.from_numpy(ap_data).unsqueeze(0).float()
        lat_tensor = torch.from_numpy(lat_data).unsqueeze(0).float()

        return {
            'x_ap': ap_tensor,
            'x_lat': lat_tensor,
            'angle': torch.tensor(info['angle_diff'], dtype=torch.long),  # 使用夹角
            'ap_angle': info['ap_angle'],
            'lat_angle': info['lat_angle'],
            'angle_diff': info['angle_diff'],
            'case_name': case_key,
            'case_info': info  # 现在所有的Path都转换为了字符串
        }


# ============================================================
# 2. 自定义collate函数 - 处理case_info字典
# ============================================================

def custom_collate(batch):
    """自定义collate函数，处理包含字典的batch"""
    if len(batch) == 0:
        return {}

    # 获取batch中的所有键
    keys = batch[0].keys()

    collated = {}
    for key in keys:
        if key == 'case_info':
            # case_info是字典，直接保存为列表
            collated[key] = [item[key] for item in batch]
        elif key == 'case_name':
            # case_name是字符串，保存为列表
            collated[key] = [item[key] for item in batch]
        elif key in ['ap_angle', 'lat_angle', 'angle_diff']:
            # 这些是数值，转换为tensor
            collated[key] = torch.tensor([item[key] for item in batch])
        else:
            # 其他是tensor，使用默认collate
            try:
                collated[key] = torch.utils.data.dataloader.default_collate([item[key] for item in batch])
            except:
                # 如果默认collate失败，保存为列表
                collated[key] = [item[key] for item in batch]

    return collated


# ============================================================
# 3. 主测试函数
# ============================================================

def main():
    print("=" * 70)
    print("独立测试脚本 - 加载训练好的模型进行推理")
    print("适配新数据格式: output_nifti -> AP/LAT + 夹角计算")
    print("=" * 70)
    print(f"测试集目录: {TEST_DATA_DIR}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 检查路径
    if not TEST_DATA_DIR.exists():
        print(f"\n错误: 测试集目录不存在: {TEST_DATA_DIR}")
        return

    if not MODEL_PATH.exists():
        print(f"\n错误: 模型文件不存在: {MODEL_PATH}")
        return

    # 创建数据集
    print(f"\n扫描数据目录...")
    print("-" * 50)
    test_dataset = TestDataset(TEST_DATA_DIR, TARGET_SIZE)

    if len(test_dataset) == 0:
        print("\n错误: 未找到任何有效病例!")
        print("请确保数据目录结构为:")
        print("  output_nifti/")
        print("    ├── ANY_病例号_投影次数/")
        print("    │   ├── ap_角度.nii.gz")
        print("    │   └── lat_角度.nii.gz")
        print("    └── ...")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 每个病例单独处理，便于保存
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=True,
        collate_fn=custom_collate  # 使用自定义collate函数
    )

    # 创建模型
    print(f"\n创建模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = X2Shape(
        img_size=IMG_SIZE,
        in_chans=1,
        num_classes=1,
        encoder_channels=ENCODER_CHANNELS,
        dims_3d=DIMS_3D,
        vbp_output_channels=VBP_OUTPUT_CHANNELS,
        vbp_volume_size=VBP_VOLUME_SIZE,
        dropout=0.0  # 测试时关闭dropout
    ).to(device)

    # 加载模型权重
    print(f"\n加载模型权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 兼容不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_dice = checkpoint.get('best_dice', 'unknown')
        print(f"加载成功! 最佳Dice: {best_dice}")
    else:
        model.load_state_dict(checkpoint)
        print("加载成功!")

    model.eval()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = OUTPUT_DIR / 'predictions'
    info_dir = OUTPUT_DIR / 'case_info'
    pred_dir.mkdir(exist_ok=True)
    info_dir.mkdir(exist_ok=True)

    # 统计
    num_samples = 0
    inference_times = []

    print(f"\n开始推理...")
    print("=" * 60)

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="推理中")):
            x_ap = batch['x_ap'].to(device)
            x_lat = batch['x_lat'].to(device)
            angle = batch['angle'].to(device)  # 夹角

            # 获取case_name和case_info
            # 由于batch_size=1，直接取第一个元素
            case_name = batch['case_name'][0]
            case_info = batch['case_info'][0]

            # 推理计时
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            # 模型前向传播 - 传入夹角
            output = model(x_ap, x_lat, angle=angle)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            num_samples += 1

            # 转换到CPU numpy
            pred_mask = torch.sigmoid(output).cpu().numpy()
            pred_single = pred_mask[0, 0]
            pred_binary = (pred_single > 0.5).astype(np.float32)

            # 保存预测结果
            pred_path = pred_dir / f"{case_name}_pred.nii.gz"
            nib.save(nib.Nifti1Image(pred_binary, np.eye(4)), pred_path)

            # 保存概率图（可选）
            prob_path = pred_dir / f"{case_name}_prob.nii.gz"
            nib.save(nib.Nifti1Image(pred_single.astype(np.float32), np.eye(4)), prob_path)

            # 保存病例信息
            info_path = info_dir / f"{case_name}_info.json"
            with open(info_path, 'w') as f:
                json.dump({
                    'case_name': case_name,
                    'ap_angle': case_info['ap_angle'],
                    'lat_angle': case_info['lat_angle'],
                    'angle_diff': case_info['angle_diff'],
                    'folder_name': case_info['folder_name'],
                    'ap_file': case_info['ap_file'],
                    'lat_file': case_info['lat_file'],
                    'inference_time_ms': inference_time * 1000
                }, f, indent=2)

            # 存储结果
            all_results.append({
                'case': case_name,
                'ap_angle': case_info['ap_angle'],
                'lat_angle': case_info['lat_angle'],
                'angle_diff': case_info['angle_diff'],
                'inference_time_ms': inference_time * 1000
            })

            # 打印单个病例结果
            print(f"\n  {case_name}:")
            print(f"    AP角度: {case_info['ap_angle']}°, LAT角度: {case_info['lat_angle']}°, 夹角: {case_info['angle_diff']}°")
            print(f"    推理时间: {inference_time * 1000:.1f}ms")
            print(f"    保存到: {pred_path}")

    # 计算统计信息
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

    # 保存结果
    results = {
        'model_path': str(MODEL_PATH),
        'test_dir': str(TEST_DATA_DIR),
        'num_samples': num_samples,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'total_inference_time_s': sum(inference_times),
        'per_case_results': all_results
    }

    # 保存结果JSON
    results_path = OUTPUT_DIR / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)
    print(f"总样本数: {num_samples}")
    print(f"平均推理时间: {avg_inference_time * 1000:.1f}ms")
    print(f"总推理时间: {sum(inference_times):.2f}s")

    # 按夹角分组统计
    print(f"\n按夹角分组统计:")
    angle_groups = defaultdict(list)
    for case_result in all_results:
        angle_groups[case_result['angle_diff']].append(case_result['case'])

    for angle, cases in sorted(angle_groups.items()):
        print(f"  夹角 {angle}°: {len(cases)} 个病例")

    print(f"\n结果保存到: {OUTPUT_DIR}")
    print(f"  - 预测二值图: {pred_dir}/*_pred.nii.gz")
    print(f"  - 预测概率图: {pred_dir}/*_prob.nii.gz")
    print(f"  - 病例信息: {info_dir}/*_info.json")
    print(f"  - 结果JSON: {results_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()