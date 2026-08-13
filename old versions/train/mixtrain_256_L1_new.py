"""
X2Shape 完整训练脚本 - 单通道输出版（混合精度训练版）
模型导入: from model.X2shap import X2Shape
数据格式: mask 为单通道二值图像 (背景:0, 目标:1)
数据集划分: train1 -> 训练集, test1 -> 验证集+测试集
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import defaultdict
import torch.nn.functional as F
# ========== 修改1: 导入混合精度训练所需模块 ==========
from torch.amp import autocast, GradScaler

# 导入模型
from model_new_any.x2shap_org import X2Shape

# ============================================================
# 路径配置 (WSL2 兼容)
# ============================================================

TRAIN_DATA_DIR = Path('/mnt/d/med_data/biron/data1/train_90')  # 训练集
TEST_DATA_DIR = Path('/mnt/d/med_data/biron/data1/test2')  # 验证集 + 测试集
CACHE_ROOT = Path('/mnt/d/med_data/biron/cache')
MODEL_ROOT = Path('/mnt/d/med_data/biron/model_new')


# ============================================================
# 1. 数据预处理（缓存到 .npz 文件）- 自动适应尺寸
# ============================================================
def generate_cache_id():
    """生成缓存文件夹ID：日期_三位随机数"""
    date_str = datetime.now().strftime("%Y%m%d")
    random_num = random.randint(100, 999)
    return f"{date_str}_{random_num}"


def preprocess_and_cache(data_dir, cache_root, split_name, img_size=256):
    """
    预处理数据并保存为 .npz 缓存文件 - 直接使用原始尺寸

    参数:
        data_dir: 数据目录路径
        cache_root: 缓存根目录
        split_name: 数据集划分名称 ('train' 或 'val_test')
        img_size: 目标图像尺寸
    """
    data_dir = Path(data_dir)
    cache_root = Path(cache_root)

    # 创建缓存目录
    cache_id = generate_cache_id()
    cache_dir = cache_root / cache_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录区分训练集和验证集
    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"数据预处理 - {split_name}")
    print(f"{'=' * 60}")
    print(f"原始数据目录: {data_dir}")
    print(f"缓存目录: {split_dir}")

    # 获取所有病例
    case_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    case_dirs.sort()
    print(f"找到 {len(case_dirs)} 个病例")

    # 验证每个病例的必要文件
    valid_cases = []
    for case_dir in case_dirs:
        ap_path = case_dir / "ap.nii.gz"
        lat_path = case_dir / "lat.nii.gz"
        mask_path = case_dir / "mask.nii.gz"

        if ap_path.exists() and lat_path.exists() and mask_path.exists():
            valid_cases.append(case_dir.name)
        else:
            print(f"警告: {case_dir.name} 缺少必要文件，已跳过")

    print(f"有效病例: {len(valid_cases)}")

    # 保存划分信息
    split_info = {
        'cache_id': cache_id,
        'img_size': img_size,
        'split_name': split_name,
        'cases': valid_cases,
        'num_cases': len(valid_cases)
    }

    with open(split_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # 处理并保存每个病例
    print(f"\n处理 {split_name} 数据...")
    for case_name in tqdm(valid_cases, desc=f"{split_name} cases"):
        case_dir = data_dir / case_name

        ap = nib.load(case_dir / "ap.nii.gz").get_fdata()
        lat = nib.load(case_dir / "lat.nii.gz").get_fdata()
        mask = nib.load(case_dir / "mask.nii.gz").get_fdata()

        ap = ap.squeeze()
        if ap.ndim == 2:
            ap = ap[None, :, :].astype(np.float32)
        elif ap.ndim == 3 and ap.shape[-1] == 1:
            ap = ap[:, :, 0][None, :, :].astype(np.float32)
        else:
            ap = ap[None, :, :].astype(np.float32) if ap.ndim == 2 else ap.astype(np.float32)

        lat = lat.squeeze()
        if lat.ndim == 2:
            lat = lat[None, :, :].astype(np.float32)
        elif lat.ndim == 3 and lat.shape[-1] == 1:
            lat = lat[:, :, 0][None, :, :].astype(np.float32)
        else:
            lat = lat[None, :, :].astype(np.float32) if lat.ndim == 2 else lat.astype(np.float32)

        mask = mask.squeeze()
        if mask.ndim == 3:
            mask = mask[None, :, :, :].astype(np.float32)
        elif mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.astype(np.float32)
        else:
            mask = mask[None, :, :, :].astype(np.float32)

        np.savez(
            split_dir / f"{case_name}.npz",
            ap=ap,
            lat=lat,
            mask=mask
        )

    sample_data = np.load(split_dir / f"{valid_cases[0]}.npz")
    actual_img_size = sample_data['ap'].shape[-1]

    print(f"\n预处理完成！")
    print(f"{split_name} 集: {len(valid_cases)} 个病例")
    print(
        f"实际图像尺寸: {actual_img_size}×{actual_img_size} (2D), {actual_img_size}×{actual_img_size}×{actual_img_size} (3D)")
    print(f"缓存目录: {split_dir}")

    return split_dir, valid_cases


def preprocess_all_data(train_dir, test_dir, cache_root, img_size=256):
    """
    预处理所有数据（训练集和验证集）

    返回:
        train_cache_dir: 训练集缓存目录
        train_cases: 训练集病例列表
        val_cache_dir: 验证集缓存目录
        val_cases: 验证集病例列表
    """
    print(f"\n{'=' * 60}")
    print(f"开始预处理所有数据")
    print(f"{'=' * 60}")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {test_dir}")
    print(f"缓存根目录: {cache_root}")
    print(f"目标图像尺寸: {img_size}×{img_size}")

    # 预处理训练集
    train_cache_dir, train_cases = preprocess_and_cache(
        data_dir=train_dir,
        cache_root=cache_root,
        split_name='train',
        img_size=img_size
    )

    # 预处理验证集（test1）
    val_cache_dir, val_cases = preprocess_and_cache(
        data_dir=test_dir,
        cache_root=cache_root,
        split_name='val_test',
        img_size=img_size
    )

    # 创建一个总的缓存目录信息
    total_cache_dir = train_cache_dir.parent  # 使用共同的父目录
    total_info = {
        'cache_id': total_cache_dir.name,
        'img_size': img_size,
        'train_cache_dir': str(train_cache_dir),
        'val_cache_dir': str(val_cache_dir),
        'num_train': len(train_cases),
        'num_val': len(val_cases),
        'train_cases': train_cases,
        'val_cases': val_cases
    }

    with open(total_cache_dir / 'total_split_info.json', 'w') as f:
        json.dump(total_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"所有数据预处理完成！")
    print(f"{'=' * 60}")
    print(f"训练集: {len(train_cases)} 个病例")
    print(f"验证集: {len(val_cases)} 个病例")
    print(f"总缓存目录: {total_cache_dir}")
    print(f"{'=' * 60}")

    return train_cache_dir, train_cases, val_cache_dir, val_cases


# ============================================================
# 2. 数据集类（流式读取）- 自动适应尺寸
# ============================================================
class CarotidDataset(Dataset):
    """从缓存文件读取数据 - 自动适应尺寸"""

    def __init__(self, cache_dir, case_list):
        self.cache_dir = Path(cache_dir)
        self.case_list = case_list
        self.file_paths = [self.cache_dir / f"{case}.npz" for case in case_list]

        with open(self.cache_dir / 'split_info.json', 'r') as f:
            split_info = json.load(f)
        self.img_size = split_info.get('img_size', 256)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        ap = torch.from_numpy(data['ap'])
        lat = torch.from_numpy(data['lat'])
        mask = torch.from_numpy(data['mask'])

        return {
            'x_ap': ap,
            'x_lat': lat,
            'mask': mask,
            'case_name': self.case_list[idx]
        }


# ============================================================
# 3. 损失函数（Dice + L1 组合）
# ============================================================

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_l1=1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_l1 = weight_l1

    def dice_loss(self, pred, target):
        """Dice Loss"""
        pred_sigmoid = torch.sigmoid(pred)

        pred_flat = pred_sigmoid.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return dice_loss

    def l1_loss(self, pred, target):
        """L1 Loss (MAE)"""
        pred_sigmoid = torch.sigmoid(pred)
        l1 = torch.abs(pred_sigmoid - target).mean()
        return l1

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        #l1_loss = self.l1_loss(pred, target)

        total_loss = self.weight_dice * dice_loss# + self.weight_l1 * l1_loss

        return total_loss, dice_loss

class DiceFocalLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_focal=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.gamma = gamma
        self.alpha = alpha

    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)

        pred_flat = pred_sigmoid.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return dice_loss

    def focal_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (self.alpha * focal_weight * bce).mean()

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return self.weight_dice * dice_loss + self.weight_focal * focal_loss, {'dice_loss': dice_loss, 'focal_loss': focal_loss}

# ============================================================
# 4. 评估指标（单通道）
# ============================================================
def compute_metrics(pred, target, threshold=0.5):
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    target_flat = target_binary.view(target_binary.shape[0], -1)

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    l1_list = []

    for i in range(pred_flat.shape[0]):
        p = pred_flat[i]
        t = target_flat[i]

        # 计算 L1 loss (MAE) 针对这个样本
        p_sigmoid = pred_sigmoid[i].view(-1)
        t_flat = target[i].view(-1)
        l1_sample = torch.abs(p_sigmoid - t_flat).mean().item()
        l1_list.append(l1_sample)

        intersection = (p * t).sum()
        pred_sum = p.sum()
        target_sum = t.sum()

        if pred_sum + target_sum > 0:
            dice = 2 * intersection / (pred_sum + target_sum)
        else:
            dice = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0
        dice_list.append(dice)

        union = pred_sum + target_sum - intersection
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0
        iou_list.append(iou)

        if pred_sum > 0:
            precision = intersection / pred_sum
        else:
            precision = 1.0 if target_sum == 0 else 0.0
        precision_list.append(precision)

        if target_sum > 0:
            recall = intersection / target_sum
        else:
            recall = 1.0 if pred_sum == 0 else 0.0
        recall_list.append(recall)

    metrics = {
        'dice': float(torch.tensor(dice_list).mean()),
        'iou': float(torch.tensor(iou_list).mean()),
        'precision': float(torch.tensor(precision_list).mean()),
        'recall': float(torch.tensor(recall_list).mean()),
        'l1': float(np.mean(l1_list))
    }

    return metrics


# ============================================================
# 5. 可视化工具
# ============================================================

def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss (Dice + L1)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_dice'], label='Train Dice', color='blue')
    axes[1].plot(history['val_dice'], label='Val Dice', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_title('Dice Score')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history['lr'], color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def save_test_results(test_metrics, save_dir):
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"测试结果")
    print(f"{'=' * 50}")
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"L1 Loss: {test_metrics['l1']:.4f}")
    print(f"{'=' * 50}")


# ============================================================
# 6. 训练器（混合精度版本 + 修复早停和学习率调度）
# ============================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-4, epochs=200, device='cuda',
                 save_dir=None, patience=6, min_delta=0.005,
                 lr_cosine_cycles=1, min_lr=1e-7,
                 lr_patience=3, warmup_epochs=0,
                 start_reduce_epoch=10):
        """
        参数:
            patience: 早停耐心值 (连续patience轮Dice不改进后停止训练)
            min_delta: Dice最小改进阈值 (0.5%的Dice提升才认为是改进)
            lr_patience: 学习率衰减耐心值 (连续lr_patience轮Dice不改进后触发余弦退火)
            lr_cosine_cycles: 每次余弦退火的周期数
            min_lr: 最小学习率 (余弦退火的目标值)
            warmup_epochs: 预热轮数
            start_reduce_epoch: 从第几轮开始启用早停和学习率衰减
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')

        # 早停参数
        self.patience = patience
        self.min_delta = min_delta

        # 学习率衰减参数（余弦退火）
        self.lr_patience = lr_patience
        self.lr_cosine_cycles = lr_cosine_cycles
        self.min_lr = min_lr

        self.warmup_epochs = warmup_epochs
        self.start_reduce_epoch = start_reduce_epoch

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # 直接使用初始学习率，不预热
        self.initial_lr = learning_rate
        self.current_lr = learning_rate

        # 使用 Dice + L1 损失
        #self.criterion = CombinedLoss(weight_dice=1.0, weight_l1=0.5)
        self.criterion = DiceFocalLoss(weight_dice=1.0, weight_focal=0.4)

        # GradScaler用于混合精度训练
        self.scaler = GradScaler('cuda')

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'lr': []
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_dice = 0.0

        # 早停计数器
        self.patience_counter = 0
        # 学习率衰减计数器
        self.lr_patience_counter = 0
        self.lr_reduce_counter = 0

        # 余弦退火状态
        self.cosine_epoch_counter = 0
        self.cosine_total_epochs = 10

    def cosine_annealing_lr(self):
        """
        余弦退火学习率衰减
        """
        current_lr = self.optimizer.param_groups[0]['lr']

        if current_lr <= self.min_lr * 1.1:
            print(f"\n  → 当前学习率 {current_lr:.2e} 已达到或接近最小学习率 {self.min_lr:.2e}，停止衰减")
            return False

        self.cosine_epoch_counter += 1
        t = self.cosine_epoch_counter % self.cosine_total_epochs
        T = self.cosine_total_epochs

        cos_val = np.cos(np.pi * t / T)
        new_lr = self.min_lr + 0.5 * (current_lr - self.min_lr) * (1 + cos_val)
        new_lr = max(new_lr, self.min_lr)

        if t == T - 1:
            self.cosine_epoch_counter = 0

        if new_lr < current_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            self.lr_reduce_counter += 1
            print(
                f"\n  → 余弦退火衰减: {current_lr:.2e} -> {new_lr:.2e} (第{self.lr_reduce_counter}次衰减, 步进 {self.cosine_epoch_counter}/{self.cosine_total_epochs})")
            return True
        else:
            print(f"\n  → 学习率未变化，停止衰减")
            return False

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Train]')

        for batch in pbar:
            x_ap = batch['x_ap'].to(self.device)
            x_lat = batch['x_lat'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda'):
                output = self.model(x_ap, x_lat)
                loss, loss_dict = self.criterion(output, mask)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  ⚠️ 检测到 NaN/Inf Loss，跳过此批次")
                continue

            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            metrics = compute_metrics(output, mask)
            total_dice += metrics['dice']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice"]:.4f}'
            })

        if num_batches == 0:
            print(f"\n  ⚠️ 警告: 所有批次都产生了 NaN Loss，使用上次的有效值")
            return float('inf'), 0.0

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return avg_loss, avg_dice

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Val]')

        with torch.no_grad():
            for batch in pbar:
                x_ap = batch['x_ap'].to(self.device)
                x_lat = batch['x_lat'].to(self.device)
                mask = batch['mask'].to(self.device)

                with autocast('cuda'):
                    output = self.model(x_ap, x_lat)
                    loss, loss_dict = self.criterion(output, mask)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  ⚠️ 验证时检测到 NaN/Inf Loss，跳过此批次")
                    continue

                total_loss += loss.item()

                metrics = compute_metrics(output, mask)
                for k, v in metrics.items():
                    total_metrics[k] += v

                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if num_batches == 0:
            print(f"\n  ⚠️ 警告: 验证集所有批次都产生了 NaN Loss")
            return float('inf'), {k: 0.0 for k in total_metrics}

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        print(f"\n  Val L1 Loss: {avg_metrics.get('l1', 0.0):.4f}")

        return avg_loss, avg_metrics

    def save_checkpoint_predictions(self, epoch):
        """保存验证集中前两个病例的预测结果到checkpoint文件夹"""
        self.model.eval()

        checkpoint_dir = self.save_dir / 'checkpoint'
        checkpoint_dir.mkdir(exist_ok=True)

        saved_count = 0
        print(f"\n  → 保存最佳模型检查点预测结果...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                x_ap = batch['x_ap'].to(self.device)
                x_lat = batch['x_lat'].to(self.device)
                case_names = batch['case_name']

                with autocast('cuda'):
                    output = self.model(x_ap, x_lat)

                pred_mask = torch.sigmoid(output).cpu().numpy()

                for i, case_name in enumerate(case_names):
                    if saved_count >= 2:
                        break

                    pred_single = pred_mask[i, 0]
                    pred_binary = (pred_single > 0.5).astype(np.float32)

                    mask_filename = f"{case_name}_{epoch + 1}.nii.gz"
                    mask_save_path = checkpoint_dir / mask_filename

                    nib.save(
                        nib.Nifti1Image(pred_binary, np.eye(4)),
                        mask_save_path
                    )

                    print(f"    → 已保存: {mask_filename}")
                    saved_count += 1

                if saved_count >= 2:
                    break

        print(f"  → 已保存 {saved_count} 个预测mask到: {checkpoint_dir}")

    def train(self):
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 病例")
        print(f"验证集: {len(self.val_loader.dataset)} 病例")
        print(f"学习率: {self.initial_lr:.2e}")
        print(f"损失函数: Dice + L1 (weight_dice=1.0, weight_l1=0.5)")
        print(f"学习率预热: {'启用' if self.warmup_epochs > 0 else '禁用'}")
        print(f"开始学习率衰减轮次: 第 {self.start_reduce_epoch + 1} 轮")
        print(f"早停耐心值 (Dice连续不改进): {self.patience} 轮")
        print(f"学习率衰减耐心值 (Dice连续不改进): {self.lr_patience} 轮")
        print(f"Dice最小改进阈值: {self.min_delta:.4f}")
        print(f"余弦退火周期长度: {self.cosine_total_epochs} epochs")
        print(f"最小学习率 (余弦退火目标): {self.min_lr:.2e}")
        print(f"混合精度训练: 已启用 (AMP + GradScaler)")
        print(f"梯度裁剪: 已启用 (max_norm=1.0)")
        print("=" * 60)

        for epoch in range(self.epochs):
            current_lr = self.optimizer.param_groups[0]['lr']

            # 训练和验证
            train_loss, train_dice = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            # 检查是否有 NaN
            if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
                print(f"\n  ⚠️ 训练 Loss 为 NaN，跳过此轮更新")
                continue

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['lr'].append(current_lr)

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
            print(f"  Val: Loss={val_loss:.4f}, Dice={val_metrics['dice']:.4f}, L1={val_metrics.get('l1', 0.0):.4f}")
            print(f"  Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")
            print(f"  LR={current_lr:.2e}")

            # ========== 1. 保存最佳模型（用Dice） ==========
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_epoch = epoch
                self.best_val_loss = val_loss  # 同步更新最佳Loss

                # 重置所有计数器
                self.patience_counter = 0
                self.lr_patience_counter = 0
                self.cosine_epoch_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'metrics': val_metrics
                }, self.save_dir / 'best_model.pth')
                print(f"  → 保存最佳模型 (Dice: {self.best_dice:.4f})")
                self.save_checkpoint_predictions(epoch)

            # ========== 2. 从指定轮次开始启用早停和学习率衰减（统一使用Dice） ==========
            if epoch >= self.start_reduce_epoch:
                # 使用Dice判断是否改进（与模型保存逻辑一致）
                if val_metrics['dice'] > self.best_dice - self.min_delta:
                    # Dice有改进（或接近最佳），重置计数器
                    self.best_val_loss = val_loss  # 更新最佳Loss
                    self.patience_counter = 0
                    self.lr_patience_counter = 0
                    print(f"  ✓ Dice接近最佳，重置计数器")
                else:
                    # Dice没有改进，两个计数器都增加
                    self.patience_counter += 1
                    self.lr_patience_counter += 1

                    print(
                        f"  未改进: 早停计数器={self.patience_counter}/{self.patience}, 学习率衰减计数器={self.lr_patience_counter}/{self.lr_patience}")

                    # ========== 学习率衰减策略 ==========
                    if self.lr_patience_counter >= self.lr_patience:
                        print(f"\n  ⚠️ 连续 {self.lr_patience_counter} 次Dice未改进！触发余弦退火学习率衰减...")
                        reduced = self.cosine_annealing_lr()
                        if reduced:
                            self.lr_patience_counter = 0
                            print(f"  → 重置学习率衰减计数器")
                        else:
                            print(f"\n学习率已达最小值，停止衰减")

                    # ========== 早停策略 ==========
                    if self.patience_counter >= self.patience:
                        print(f"\n  ⚠️ 连续 {self.patience_counter} 次Dice未改进！触发早停...")
                        print(f"最佳轮次: {self.best_epoch + 1} (Dice: {self.best_dice:.4f})")
                        break

        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, self.save_dir / 'final_model.pth')

        plot_training_history(self.history, self.save_dir / 'training_curves.png')

        history_json = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'train_dice': [float(x) for x in self.history['train_dice']],
            'val_dice': [float(x) for x in self.history['val_dice']],
            'lr': [float(x) for x in self.history['lr']]
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history_json, f, indent=2)

        return self.history

    def test(self):
        """测试最佳模型 - 保存所有病例的结果"""
        print(f"\n{'=' * 50}")
        print(f"开始测试...")
        print(f"{'=' * 50}")

        checkpoint = torch.load(self.save_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (Dice: {checkpoint['best_dice']:.4f})")

        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = 0

        test_dir = self.save_dir / 'test'
        test_dir.mkdir(exist_ok=True)

        pbar = tqdm(self.test_loader, desc="Testing")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                x_ap = batch['x_ap'].to(self.device)
                x_lat = batch['x_lat'].to(self.device)
                mask = batch['mask'].to(self.device)
                case_names = batch['case_name']

                with autocast('cuda'):
                    output = self.model(x_ap, x_lat)

                metrics = compute_metrics(output, mask)
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1

                pred_mask = torch.sigmoid(output).cpu().numpy()
                gt_mask = mask.cpu().numpy()

                for i, case_name in enumerate(case_names):
                    pred_single = pred_mask[i, 0]
                    gt_single = gt_mask[i, 0]
                    pred_binary = (pred_single > 0.5).astype(np.float32)

                    nib.save(
                        nib.Nifti1Image(pred_binary, np.eye(4)),
                        test_dir / f"{case_name}_pred.nii.gz"
                    )

                    nib.save(
                        nib.Nifti1Image(gt_single.astype(np.float32), np.eye(4)),
                        test_dir / f"{case_name}_gt.nii.gz"
                    )

                pbar.set_postfix({'dice': metrics['dice']})

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        save_test_results(avg_metrics, test_dir)

        print(f"\n{'=' * 50}")
        print(f"总体测试 L1 Loss: {avg_metrics.get('l1', 0.0):.4f}")
        print(f"{'=' * 50}")

        return avg_metrics


# ============================================================
# 7. 主函数
# ============================================================
def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    IMG_SIZE = 256

    print("=" * 60)
    print("X2Shape 颈总动脉分割训练 (单通道输出)")
    print("=" * 60)
    print(f"训练集目录: {TRAIN_DATA_DIR}")
    print(f"验证/测试集目录: {TEST_DATA_DIR}")
    print(f"缓存目录: {CACHE_ROOT}")
    print(f"模型目录: {MODEL_ROOT}")
    print(f"目标图像尺寸: {IMG_SIZE}×{IMG_SIZE} (2D), {IMG_SIZE}×{IMG_SIZE}×{IMG_SIZE} (3D)")

    if not TRAIN_DATA_DIR.exists():
        print(f"\n错误: 训练集目录不存在: {TRAIN_DATA_DIR}")
        return

    if not TEST_DATA_DIR.exists():
        print(f"\n错误: 验证集目录不存在: {TEST_DATA_DIR}")
        return

    existing_caches = [d for d in CACHE_ROOT.iterdir() if d.is_dir()] if CACHE_ROOT.exists() else []

    train_cache_dir = None
    train_cases = None
    val_cache_dir = None
    val_cases = None

    if existing_caches:
        print(f"\n发现已有缓存目录:")
        for cache in existing_caches:
            try:
                if (cache / 'total_split_info.json').exists():
                    with open(cache / 'total_split_info.json', 'r') as f:
                        cache_info = json.load(f)
                        cache_size = cache_info.get('img_size', 'unknown')
                        num_train = cache_info.get('num_train', 0)
                        num_val = cache_info.get('num_val', 0)
                    print(f"  - {cache.name} (尺寸: {cache_size}, 训练: {num_train}, 验证: {num_val})")
                elif (cache / 'train' / 'split_info.json').exists():
                    print(f"  - {cache.name} (旧版本缓存)")
            except:
                print(f"  - {cache.name}")

        use_existing = input("\n是否使用最新的缓存? (y/n): ").strip().lower()

        if use_existing == 'y':
            latest_cache = None
            for cache in sorted(existing_caches, reverse=True):
                if (cache / 'total_split_info.json').exists():
                    latest_cache = cache
                    break

            if latest_cache is not None:
                with open(latest_cache / 'total_split_info.json', 'r') as f:
                    total_info = json.load(f)

                cached_size = total_info.get('img_size', 256)
                if cached_size != IMG_SIZE:
                    print(f"警告: 缓存图像尺寸({cached_size})与目标尺寸({IMG_SIZE})不一致!")
                    print("将重新预处理数据...")
                    train_cache_dir, train_cases, val_cache_dir, val_cases = preprocess_all_data(
                        train_dir=TRAIN_DATA_DIR,
                        test_dir=TEST_DATA_DIR,
                        cache_root=CACHE_ROOT,
                        img_size=IMG_SIZE
                    )
                else:
                    train_cache_dir = Path(total_info['train_cache_dir'])
                    val_cache_dir = Path(total_info['val_cache_dir'])
                    train_cases = total_info['train_cases']
                    val_cases = total_info['val_cases']
                    print(f"\n使用缓存: {latest_cache}")
                    print(f"训练集缓存: {train_cache_dir}")
                    print(f"验证集缓存: {val_cache_dir}")
                    print(f"训练集: {len(train_cases)} 个病例")
                    print(f"验证集: {len(val_cases)} 个病例")
            else:
                print("未找到完整缓存，将重新预处理数据...")
                train_cache_dir, train_cases, val_cache_dir, val_cases = preprocess_all_data(
                    train_dir=TRAIN_DATA_DIR,
                    test_dir=TEST_DATA_DIR,
                    cache_root=CACHE_ROOT,
                    img_size=IMG_SIZE
                )
        else:
            train_cache_dir, train_cases, val_cache_dir, val_cases = preprocess_all_data(
                train_dir=TRAIN_DATA_DIR,
                test_dir=TEST_DATA_DIR,
                cache_root=CACHE_ROOT,
                img_size=IMG_SIZE
            )
    else:
        train_cache_dir, train_cases, val_cache_dir, val_cases = preprocess_all_data(
            train_dir=TRAIN_DATA_DIR,
            test_dir=TEST_DATA_DIR,
            cache_root=CACHE_ROOT,
            img_size=IMG_SIZE
        )

    train_dataset = CarotidDataset(train_cache_dir, train_cases)
    val_dataset = CarotidDataset(val_cache_dir, val_cases)
    test_dataset = CarotidDataset(val_cache_dir, val_cases)

    # ========== 优化2: 增加batch size和workers ==========
    batch_size = 2
    num_workers = 4  # 增加到8
    prefetch_factor = 2  # 预加载

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=False  # 保持worker进程存活
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    model_save_dir = MODEL_ROOT / train_cache_dir.parent.name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n模型保存目录: {model_save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    """model = X2Shape(
        img_size=256,
        in_chans=1,
        num_classes=1,
        dims_2d=[32, 64, 128],  # 减少通道
        depths_2d=[1, 1, 1],  # 3层
        dims_3d=[32, 64, 128],  # 减少通道
        vbp_output_channels=64  # 减半
    ).to(device)"""
    model = X2Shape(
        img_size=256,
        in_chans=1,
        num_classes=1,
        dims_2d=[16, 32, 64, 128],
        depths_2d=[1, 1, 2, 1],
        dims_3d=[16, 32, 64, 128],
        vbp_output_channels=32,
        vbp_embed_dim=32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # ========== 创建 Trainer 使用修复后的参数 ==========
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=5e-5,
        epochs=80,
        device=device,
        save_dir=model_save_dir,
        patience=12,  # 早停: Dice连续12次不改进后停止
        min_delta=0.005,  # Dice最小改进阈值 (0.5%)
        lr_cosine_cycles=1,
        min_lr=1e-6,
        lr_patience=4,  # 学习率衰减: Dice连续4次不改进后触发
        warmup_epochs=0,
        start_reduce_epoch=10  # 从第11轮开始启用早停和学习率衰减
    )

    history = trainer.train()
    test_metrics = trainer.test()

    print(f"\n训练完成！")
    print(f"最佳验证 Dice: {trainer.best_dice:.4f} (Epoch {trainer.best_epoch + 1})")
    print(f"模型保存在: {model_save_dir}")
    print(f"测试结果保存在: {model_save_dir}/test/")


if __name__ == "__main__":
    main()