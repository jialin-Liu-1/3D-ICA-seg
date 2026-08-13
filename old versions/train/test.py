"""
X2Shape 模型测试脚本 - 修复数据加载，保持三维结构
"""

import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import time
import json
from datetime import datetime

# 导入模型
from model.x2shap_single160 import X2Shape

# ============================================================
# 配置路径
# ============================================================

MODEL_PATH = "/mnt/d/med_data/biron/model_new/20260608_875/best_model.pth"
AP_PATH = "/mnt/d/med_data/biron/data1/train/1_5/ap.nii.gz"
LAT_PATH = "/mnt/d/med_data/biron/data1/train/1_5/lat.nii.gz"
MASK_PATH = "/mnt/d/med_data/biron/model_new/20260608_875/test2/mask.nii.gz"
OUTPUT_DIR = "/mnt/d/med_data/biron/model_new/20260608_875/test2"

IMG_SIZE = 256

# ============================================================
# 评估指标计算函数
# ============================================================

def compute_metrics(pred, target, threshold=0.5):
    """
    计算DICE, IoU, Precision, Recall
    """
    # 确保尺寸一致
    if pred.shape != target.shape:
        print(f"  警告: 预测尺寸 {pred.shape} 与目标尺寸 {target.shape} 不匹配")
        # 裁剪到相同尺寸
        min_h = min(pred.shape[0], target.shape[0])
        min_w = min(pred.shape[1], target.shape[1])
        min_d = min(pred.shape[2], target.shape[2])
        pred = pred[:min_h, :min_w, :min_d]
        target = target[:min_h, :min_w, :min_d]

    # 二值化
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    # 展平
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    # 计算
    intersection = np.sum(pred_flat * target_flat)
    pred_sum = np.sum(pred_flat)
    target_sum = np.sum(target_flat)

    # DICE
    if pred_sum + target_sum > 0:
        dice = 2.0 * intersection / (pred_sum + target_sum)
    else:
        dice = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0

    # IoU
    union = pred_sum + target_sum - intersection
    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0

    # Precision
    if pred_sum > 0:
        precision = intersection / pred_sum
    else:
        precision = 1.0 if target_sum == 0 else 0.0

    # Recall
    if target_sum > 0:
        recall = intersection / target_sum
    else:
        recall = 1.0 if pred_sum == 0 else 0.0

    metrics = {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }

    return metrics

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("X2Shape 模型测试")
    print("=" * 60)

    # 1. 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 2. 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = X2Shape(
        img_size=IMG_SIZE,
        in_chans=1,
        num_classes=1,
        dims_2d=[32, 64, 128, 256],
        depths_2d=[1, 1, 2, 1],
        dims_3d=[32, 64, 128, 256],
        vbp_output_channels=64
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  加载模型 (Epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("  加载模型权重")

    model.eval()

    # 3. 加载数据
    print(f"\n加载测试数据:")

    # 加载AP (2D输入)
    ap_img = nib.load(AP_PATH).get_fdata().astype(np.float32)
    ap_img = ap_img.squeeze()
    if ap_img.ndim == 3:
        ap_img = ap_img[:, :, ap_img.shape[2] // 2]
    # 添加通道维度 (1, H, W)
    ap_img = ap_img[None, :, :]
    ap_tensor = torch.from_numpy(ap_img).unsqueeze(0).to(device)

    # 加载LAT (2D输入)
    lat_img = nib.load(LAT_PATH).get_fdata().astype(np.float32)
    lat_img = lat_img.squeeze()
    if lat_img.ndim == 3:
        lat_img = lat_img[:, :, lat_img.shape[2] // 2]
    lat_img = lat_img[None, :, :]
    lat_tensor = torch.from_numpy(lat_img).unsqueeze(0).to(device)

    # 加载mask - 保持三维，不使用squeeze()
    mask_img = nib.load(MASK_PATH).get_fdata().astype(np.float32)
    # 确保是三维 (H, W, D)
    if mask_img.ndim == 2:
        mask_img = mask_img[:, :, np.newaxis]
    elif mask_img.ndim == 4:
        # 如果有通道维度，去除
        mask_img = mask_img.squeeze()
        if mask_img.ndim == 2:
            mask_img = mask_img[:, :, np.newaxis]

    print(f"  AP尺寸: {ap_img.shape}")
    print(f"  LAT尺寸: {lat_img.shape}")
    print(f"  Mask尺寸: {mask_img.shape}")

    # 4. 推理
    print(f"\n开始推理...")

    # Warm-up
    with torch.no_grad():
        _ = model(ap_tensor, lat_tensor)

    # 正式推理
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        output = model(ap_tensor, lat_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    print(f"  前向传播时间: {inference_time:.4f} 秒")
    print(f"  输出尺寸: {output.shape}")

    # 5. 后处理 - 与训练代码完全一致
    pred_prob = torch.sigmoid(output).cpu().numpy()

    # 提取预测 (与训练代码一致: pred_mask[i, 0])
    pred_single = pred_prob[0, 0]  # (D, H, W)
    print(f"  预测尺寸: {pred_single.shape}")
    print(f"  GT尺寸: {mask_img.shape}")

    # 检查维度顺序，如果预测是(D, H, W)而GT是(H, W, D)，需要转置
    if pred_single.shape != mask_img.shape:
        # 尝试转置预测
        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            pred_perm = np.transpose(pred_single, perm)
            if pred_perm.shape == mask_img.shape:
                pred_single = pred_perm
                print(f"  调整维度顺序: {perm}")
                break

    # 二值化
    pred_binary = (pred_single > 0.5).astype(np.float32)

    # 6. 计算评价指标
    print(f"\n计算评价指标...")
    metrics = compute_metrics(pred_single, mask_img)

    print(f"\n{'=' * 50}")
    print(f"测试结果")
    print(f"{'=' * 50}")
    print(f"Dice Score:    {metrics['dice']:.4f}")
    print(f"IoU:           {metrics['iou']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"前向传播时间:  {inference_time:.4f} 秒")
    print(f"{'=' * 50}")

    # 7. 保存结果
    print(f"\n保存结果到: {output_dir}")

    # 保存预测mask
    nib.save(
        nib.Nifti1Image(pred_binary, np.eye(4)),
        output_dir / "pred_mask.nii.gz"
    )
    print(f"  预测mask已保存")

    # 保存概率图
    nib.save(
        nib.Nifti1Image(pred_single.astype(np.float32), np.eye(4)),
        output_dir / "pred_probability.nii.gz"
    )
    print(f"  概率图已保存")

    # 保存GT mask
    nib.save(
        nib.Nifti1Image(mask_img.astype(np.float32), np.eye(4)),
        output_dir / "gt_mask.nii.gz"
    )
    print(f"  GT mask已保存")

    # 保存评价指标
    metrics_with_time = {
        **metrics,
        'inference_time_seconds': inference_time,
        'model_path': MODEL_PATH,
        'test_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'output_shape': list(pred_single.shape),
        'gt_shape': list(mask_img.shape)
    }

    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump(metrics_with_time, f, indent=2)
    print(f"  评价指标已保存")

    # 保存文本报告
    with open(output_dir / "test_report.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("X2Shape 模型测试报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {MODEL_PATH}\n\n")
        f.write("评价指标:\n")
        f.write(f"  Dice Score:    {metrics['dice']:.4f}\n")
        f.write(f"  IoU:           {metrics['iou']:.4f}\n")
        f.write(f"  Precision:     {metrics['precision']:.4f}\n")
        f.write(f"  Recall:        {metrics['recall']:.4f}\n\n")
        f.write(f"前向传播时间:  {inference_time:.4f} 秒\n")
        f.write(f"输出尺寸: {pred_single.shape}\n")
        f.write(f"GT尺寸: {mask_img.shape}\n")
        f.write("=" * 60 + "\n")

    print(f"  测试报告已保存")

    print(f"\n测试完成！")


if __name__ == "__main__":
    main()
