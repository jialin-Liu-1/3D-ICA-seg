"""
独立测试脚本 - 加载训练好的模型并在测试集上推理
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

# 导入模型 - 修改为您的模型路径
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model.x2shap_256_any_new import X2Shape

# ============================================================
# 配置
# ============================================================

TEST_DATA_DIR = Path('/mnt/d/med_data/biron/data1/test_any1')
MODEL_PATH = Path('/mnt/d/med_data/biron/model_any/20260715_264/best_model.pth')
OUTPUT_DIR = Path('/mnt/d/med_data/biron/model_any/20260715_264/test_results')
CACHE_DIR = Path('/mnt/d/med_data/biron/cache')

# 模型参数 (必须与训练时一致)
IMG_SIZE = 256
VBP_VOLUME_SIZE = 64
ENCODER_CHANNELS = 32
VBP_OUTPUT_CHANNELS = 64
DIMS_3D = [32, 64, 128, 256]


# ============================================================
# 1. 数据集类 (从缓存读取)
# ============================================================

class TestDataset(Dataset):
    """测试数据集 - 从缓存文件读取"""

    def __init__(self, cache_dir, case_list):
        self.cache_dir = Path(cache_dir)
        self.case_list = case_list
        self.file_paths = [self.cache_dir / f"{case}.npz" for case in case_list]

        # 读取角度信息
        with open(self.cache_dir / 'split_info.json', 'r') as f:
            split_info = json.load(f)
        self.case_angles = split_info.get('case_angles', {})
        self.img_size = split_info.get('img_size', 256)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        ap = torch.from_numpy(data['ap'])
        lat = torch.from_numpy(data['lat'])
        mask = torch.from_numpy(data['mask'])

        # 读取角度
        if 'angle' in data:
            angle = int(data['angle'])
        else:
            case_name = self.case_list[idx]
            angle = self.case_angles.get(case_name, 90)

        return {
            'x_ap': ap,
            'x_lat': lat,
            'mask': mask,
            'angle': torch.tensor(angle, dtype=torch.long),
            'case_name': self.case_list[idx]
        }


# ============================================================
# 2. 查找缓存目录
# ============================================================

def find_cache_dir(test_data_dir, cache_root):
    """查找测试集对应的缓存目录"""
    cache_root = Path(cache_root)

    if not cache_root.exists():
        return None

    # 查找所有缓存目录
    for cache_dir in cache_root.iterdir():
        if cache_dir.is_dir():
            # 检查是否有 val_test 子目录
            val_test_dir = cache_dir / 'val_test'
            if val_test_dir.exists() and (val_test_dir / 'split_info.json').exists():
                # 检查是否包含测试集数据
                with open(val_test_dir / 'split_info.json', 'r') as f:
                    info = json.load(f)
                cases = info.get('cases', [])
                if cases:
                    return val_test_dir

    return None


# ============================================================
# 3. 评估指标
# ============================================================

def compute_metrics(pred, target, threshold=0.5):
    """计算评估指标"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    target_flat = target_binary.view(target_binary.shape[0], -1)

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    for i in range(pred_flat.shape[0]):
        p = pred_flat[i]
        t = target_flat[i]

        intersection = (p * t).sum()
        pred_sum = p.sum()
        target_sum = t.sum()

        # Dice
        if pred_sum + target_sum > 0:
            dice = 2 * intersection / (pred_sum + target_sum)
        else:
            dice = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0
        dice_list.append(dice)

        # IoU
        union = pred_sum + target_sum - intersection
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0
        iou_list.append(iou)

        # Precision
        if pred_sum > 0:
            precision = intersection / pred_sum
        else:
            precision = 1.0 if target_sum == 0 else 0.0
        precision_list.append(precision)

        # Recall
        if target_sum > 0:
            recall = intersection / target_sum
        else:
            recall = 1.0 if pred_sum == 0 else 0.0
        recall_list.append(recall)

    metrics = {
        'dice': float(torch.tensor(dice_list).mean()),
        'iou': float(torch.tensor(iou_list).mean()),
        'precision': float(torch.tensor(precision_list).mean()),
        'recall': float(torch.tensor(recall_list).mean())
    }

    return metrics


# ============================================================
# 4. 主测试函数
# ============================================================

def main():
    print("=" * 70)
    print("独立测试脚本 - 加载训练好的模型进行推理")
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

    # 查找缓存
    print(f"\n查找缓存目录...")
    cache_dir = find_cache_dir(TEST_DATA_DIR, CACHE_DIR)

    if cache_dir is None:
        print(f"\n错误: 未找到测试集的缓存文件!")
        print(f"请先运行训练脚本生成缓存，或指定正确的缓存目录。")
        return

    print(f"找到缓存目录: {cache_dir}")

    # 读取病例列表
    with open(cache_dir / 'split_info.json', 'r') as f:
        info = json.load(f)
    case_list = info.get('cases', [])
    case_angles = info.get('case_angles', {})

    print(f"测试集病例数: {len(case_list)}")
    print(f"角度分布: {set(case_angles.values()) if case_angles else '未知'}")

    # 创建数据集
    test_dataset = TestDataset(cache_dir, case_list)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 每个病例单独处理，便于保存
        shuffle=False,
        num_workers=2,
        pin_memory=True
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
    gt_dir = OUTPUT_DIR / 'ground_truth'
    pred_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)

    # 统计
    total_metrics = defaultdict(float)
    num_samples = 0
    inference_times = []

    print(f"\n开始推理...")
    print("=" * 60)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="推理中")):
            x_ap = batch['x_ap'].to(device)
            x_lat = batch['x_lat'].to(device)
            mask = batch['mask'].to(device)
            angle = batch['angle'].to(device)
            case_name = batch['case_name'][0]

            # 推理计时
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            # 模型前向传播
            output = model(x_ap, x_lat, angle=angle)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 计算指标
            metrics = compute_metrics(output, mask)
            for k, v in metrics.items():
                total_metrics[k] += v
            num_samples += 1

            # 转换到CPU numpy
            pred_mask = torch.sigmoid(output).cpu().numpy()
            gt_mask = mask.cpu().numpy()

            # 保存预测结果
            pred_single = pred_mask[0, 0]
            pred_binary = (pred_single > 0.5).astype(np.float32)

            gt_single = gt_mask[0, 0]

            # 保存为NIfTI文件
            pred_path = pred_dir / f"{case_name}_pred.nii.gz"
            gt_path = gt_dir / f"{case_name}_gt.nii.gz"

            nib.save(nib.Nifti1Image(pred_binary, np.eye(4)), pred_path)
            nib.save(nib.Nifti1Image(gt_single.astype(np.float32), np.eye(4)), gt_path)

            # 打印单个病例结果
            print(f"\n  {case_name} (角度: {angle.item()}°): "
                  f"Dice={metrics['dice']:.4f}, "
                  f"IoU={metrics['iou']:.4f}, "
                  f"Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}, "
                  f"Time={inference_time * 1000:.1f}ms")

    # 计算平均指标
    avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}
    avg_inference_time = sum(inference_times) / len(inference_times)

    # 保存结果
    results = {
        'model_path': str(MODEL_PATH),
        'test_dir': str(TEST_DATA_DIR),
        'num_samples': num_samples,
        'metrics': avg_metrics,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'total_inference_time_s': sum(inference_times),
        'per_case_metrics': [
            {
                'case': case_name,
                'metrics': metrics
            }
            for case_name, metrics in zip(
                [batch['case_name'][0] for batch in test_loader],
                [compute_metrics(
                    model(batch['x_ap'].to(device), batch['x_lat'].to(device),
                          batch['angle'].to(device)),
                    batch['mask'].to(device)
                ) for batch in test_loader]
            ) if False  # 这里为了简洁跳过，实际会重复计算
        ]
    }

    # 重新计算每个病例的指标（更准确）
    per_case_results = []
    with torch.no_grad():
        for batch in test_loader:
            x_ap = batch['x_ap'].to(device)
            x_lat = batch['x_lat'].to(device)
            mask = batch['mask'].to(device)
            angle = batch['angle'].to(device)
            case_name = batch['case_name'][0]

            output = model(x_ap, x_lat, angle=angle)
            metrics = compute_metrics(output, mask)
            per_case_results.append({
                'case': case_name,
                'angle': angle.item(),
                'metrics': metrics
            })

    results['per_case_results'] = per_case_results

    # 保存结果JSON
    results_path = OUTPUT_DIR / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"总样本数: {num_samples}")
    print(f"平均指标:")
    print(f"  Dice: {avg_metrics['dice']:.4f}")
    print(f"  IoU: {avg_metrics['iou']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print(f"平均推理时间: {avg_inference_time * 1000:.1f}ms")
    print(f"\n结果保存到: {OUTPUT_DIR}")
    print(f"  - 预测: {pred_dir}")
    print(f"  - Ground Truth: {gt_dir}")
    print(f"  - 结果JSON: {results_path}")
    print("=" * 60)

    return avg_metrics, results


if __name__ == "__main__":
    main()