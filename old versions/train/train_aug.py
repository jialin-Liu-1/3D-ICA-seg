"""
X2Shape 完整训练脚本 - 双通道输出版（带数据增强 + 分阶段训练）
模型导入: from model.X2shap import X2Shape
数据格式: mask 为双通道 (左颈总动脉, 右颈总动脉)

数据增强策略：
- 对比度变化: ±15% 随机变化
- 亮度变化: ±15% 随机变化
- 独立参数更新方式（每个增强版本独立更新参数，真正扩大数据集）

分阶段训练：
- phase1_epochs: 不使用数据增强的轮数（默认0，即从一开始就使用增强）
- 可以设置 phase1_epochs=8，表示前8轮不使用增强，第9轮开始使用增强
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

# 导入模型
from model.X2shap import X2Shape

# ============================================================
# 路径配置 (WSL2 兼容)
# ============================================================

DATA_DIR = Path('/mnt/d/med_data/biron/data/train_all')
CACHE_ROOT = Path('/mnt/d/med_data/biron/cache')
MODEL_ROOT = Path('/mnt/d/med_data/biron/model')


# ============================================================
# 0. 数据增强函数
# ============================================================

class MedicalImageAugmenter:
    """医学图像数据增强器（对比度、亮度变化）"""

    @staticmethod
    def adjust_contrast(image, change_percent):
        """
        调整对比度
        Args:
            image: numpy array or torch tensor, shape (C, H, W) or (H, W)
            change_percent: 变化百分比，范围 -0.15 到 0.15
                          负值降低对比度，正值增加对比度
        Returns:
            调整后的图像
        """
        # 转换为 numpy 进行处理
        was_tensor = isinstance(image, torch.Tensor)
        if was_tensor:
            image_np = image.cpu().numpy()
        else:
            image_np = image.copy()

        # 计算对比度调整因子
        # 公式: new = mean + (old - mean) * (1 + factor)
        factor = change_percent

        # 对每个通道独立处理（如果是多通道）
        if image_np.ndim == 3:  # (C, H, W)
            for c in range(image_np.shape[0]):
                mean = image_np[c].mean()
                image_np[c] = mean + (image_np[c] - mean) * (1 + factor)
        else:  # (H, W)
            mean = image_np.mean()
            image_np = mean + (image_np - mean) * (1 + factor)

        # 裁剪到有效范围 [0, 1]
        image_np = np.clip(image_np, 0, 1)

        if was_tensor:
            return torch.from_numpy(image_np).to(image.device)
        return image_np

    @staticmethod
    def adjust_brightness(image, change_percent):
        """
        调整亮度
        Args:
            image: numpy array or torch tensor, shape (C, H, W) or (H, W)
            change_percent: 变化百分比，范围 -0.15 到 0.15
                          负值降低亮度，正值增加亮度
        Returns:
            调整后的图像
        """
        was_tensor = isinstance(image, torch.Tensor)
        if was_tensor:
            image_np = image.cpu().numpy()
        else:
            image_np = image.copy()

        # 亮度调整：直接添加偏移量
        image_np = image_np + change_percent

        # 裁剪到有效范围 [0, 1]
        image_np = np.clip(image_np, 0, 1)

        if was_tensor:
            return torch.from_numpy(image_np).to(image.device)
        return image_np

    @classmethod
    def random_contrast_change(cls, image):
        """随机对比度变化（-15% 到 +15%）"""
        change_percent = random.uniform(-0.15, 0.15)
        return cls.adjust_contrast(image, change_percent), change_percent

    @classmethod
    def random_brightness_change(cls, image):
        """随机亮度变化（-15% 到 +15%）"""
        change_percent = random.uniform(-0.15, 0.15)
        return cls.adjust_brightness(image, change_percent), change_percent


# ============================================================
# 1. 数据预处理（缓存到 .npz 文件）
# ============================================================

def generate_cache_id():
    """生成缓存文件夹ID：日期_三位随机数"""
    date_str = datetime.now().strftime("%Y%m%d")
    random_num = random.randint(100, 999)
    return f"{date_str}_{random_num}"


def preprocess_and_cache(data_dir, cache_root, img_size=128, train_ratio=0.8, seed=42):
    """预处理数据并保存为 .npz 缓存文件"""
    data_dir = Path(data_dir)
    cache_root = Path(cache_root)

    # 创建缓存目录
    cache_id = generate_cache_id()
    cache_dir = cache_root / cache_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"数据预处理")
    print(f"{'=' * 60}")
    print(f"原始数据目录: {data_dir}")
    print(f"缓存目录: {cache_dir}")

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

    # 划分训练集和验证集
    random.seed(seed)
    indices = list(range(len(valid_cases)))
    random.shuffle(indices)

    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_cases = [valid_cases[i] for i in train_indices]
    val_cases = [valid_cases[i] for i in val_indices]

    splits = {'train': train_cases, 'val': val_cases}

    # 保存划分信息
    split_info = {
        'cache_id': cache_id,
        'train_ratio': train_ratio,
        'seed': seed,
        'train_cases': train_cases,
        'val_cases': val_cases,
        'num_train': len(train_cases),
        'num_val': len(val_cases)
    }

    with open(cache_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    # 处理并保存每个病例
    print("\n处理训练集...")
    for case_name in tqdm(train_cases, desc="Training cases"):
        case_dir = data_dir / case_name

        # 加载数据
        ap = nib.load(case_dir / "ap.nii.gz").get_fdata()
        lat = nib.load(case_dir / "lat.nii.gz").get_fdata()
        mask = nib.load(case_dir / "mask.nii.gz").get_fdata()

        # 确保形状正确
        # ap: (128, 128, 1) -> (1, 128, 128)
        ap = ap.squeeze()[None, :, :].astype(np.float32)
        # lat: (128, 128, 1) -> (1, 128, 128)
        lat = lat.squeeze()[None, :, :].astype(np.float32)
        # mask: (128, 128, 128, 2) -> (2, 128, 128, 128)
        mask = np.transpose(mask, (3, 0, 1, 2)).astype(np.float32)

        # 保存为 npz 文件
        np.savez(
            cache_dir / f"{case_name}.npz",
            ap=ap,
            lat=lat,
            mask=mask
        )

    print("\n处理验证集...")
    for case_name in tqdm(val_cases, desc="Validation cases"):
        case_dir = data_dir / case_name

        ap = nib.load(case_dir / "ap.nii.gz").get_fdata()
        lat = nib.load(case_dir / "lat.nii.gz").get_fdata()
        mask = nib.load(case_dir / "mask.nii.gz").get_fdata()

        ap = ap.squeeze()[None, :, :].astype(np.float32)
        lat = lat.squeeze()[None, :, :].astype(np.float32)
        mask = np.transpose(mask, (3, 0, 1, 2)).astype(np.float32)

        np.savez(
            cache_dir / f"{case_name}.npz",
            ap=ap,
            lat=lat,
            mask=mask
        )

    print(f"\n预处理完成！")
    print(f"训练集: {len(train_cases)} 个病例")
    print(f"验证集: {len(val_cases)} 个病例")
    print(f"缓存目录: {cache_dir}")

    return cache_dir, splits


# ============================================================
# 2. 数据集类（流式读取，带数据增强）
# ============================================================

class CarotidDataset(Dataset):
    """从缓存文件读取数据，支持数据增强"""

    def __init__(self, cache_dir, case_list, augment=False):
        """
        Args:
            cache_dir: 缓存目录
            case_list: 病例列表
            augment: 是否启用数据增强（训练模式启用，验证模式禁用）
        """
        self.cache_dir = Path(cache_dir)
        self.case_list = case_list
        self.file_paths = [self.cache_dir / f"{case}.npz" for case in case_list]
        self.augment = augment
        self.augmenter = MedicalImageAugmenter() if augment else None

    def __len__(self):
        return len(self.file_paths)

    def _augment_sample(self, ap, lat):
        """
        对样本应用数据增强
        先使用原始数据，然后生成两个增强版本：对比度变化、亮度变化

        Returns:
            ap_original, lat_original: 原始数据
            ap_contrast, lat_contrast: 对比度增强数据
            ap_brightness, lat_brightness: 亮度增强数据
            contrast_factor: 对比度变化因子
            brightness_factor: 亮度变化因子
        """
        # 随机生成变化因子（-15% 到 +15%）
        contrast_factor = random.uniform(-0.15, 0.15)
        brightness_factor = random.uniform(-0.15, 0.15)

        # 应用对比度变化
        ap_contrast = self.augmenter.adjust_contrast(ap, contrast_factor)
        lat_contrast = self.augmenter.adjust_contrast(lat, contrast_factor)

        # 应用亮度变化
        ap_brightness = self.augmenter.adjust_brightness(ap, brightness_factor)
        lat_brightness = self.augmenter.adjust_brightness(lat, brightness_factor)

        return ap, lat, ap_contrast, lat_contrast, ap_brightness, lat_brightness, contrast_factor, brightness_factor

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        ap = torch.from_numpy(data['ap'])  # (1, 128, 128)
        lat = torch.from_numpy(data['lat'])  # (1, 128, 128)
        mask = torch.from_numpy(data['mask'])  # (2, 128, 128, 128)

        case_name = self.case_list[idx]

        if self.augment:
            # 训练模式：生成原始数据和增强数据
            (ap_original, lat_original,
             ap_contrast, lat_contrast,
             ap_brightness, lat_brightness,
             contrast_factor, brightness_factor) = self._augment_sample(ap, lat)

            return {
                'ap_original': ap_original,
                'lat_original': lat_original,
                'ap_contrast': ap_contrast,
                'lat_contrast': lat_contrast,
                'ap_brightness': ap_brightness,
                'lat_brightness': lat_brightness,
                'mask': mask,
                'case_name': case_name,
                'contrast_factor': contrast_factor,
                'brightness_factor': brightness_factor,
                'use_augmentation': True
            }
        else:
            # 验证模式或非增强训练模式：只返回原始数据
            return {
                'ap': ap,
                'lat': lat,
                'mask': mask,
                'case_name': case_name,
                'use_augmentation': False
            }


# ============================================================
# 3. 损失函数（双通道 BCE + Dice）
# ============================================================

class CombinedLoss(nn.Module):
    """组合损失: Dice Loss + BCE Loss (适用于多通道二分类)"""

    def __init__(self, weight_dice=1.0, weight_bce=1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        """多通道 Dice Loss"""
        pred_sigmoid = torch.sigmoid(pred)

        # 展平
        pred_flat = pred_sigmoid.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return dice_loss

    def forward(self, pred, target):
        """
        pred: (B, 2, D, H, W) logits
        target: (B, 2, D, H, W) binary mask
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)

        total_loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss

        return total_loss, {'dice_loss': dice_loss, 'bce_loss': bce_loss}


# ============================================================
# 4. 评估指标
# ============================================================

def compute_metrics(pred, target, threshold=0.5):
    """
    计算分割指标

    Args:
        pred: (B, 2, D, H, W) logits
        target: (B, 2, D, H, W) binary mask
        threshold: 二值化阈值

    Returns:
        metrics: dict with dice_left, dice_right, iou_left, iou_right
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    metrics = {}

    # 通道0: 左颈总动脉, 通道1: 右颈总动脉
    for ch, name in [(0, 'left'), (1, 'right')]:
        pred_c = pred_binary[:, ch, ...]
        target_c = target[:, ch, ...]

        intersection = (pred_c * target_c).sum().float()
        pred_sum = pred_c.sum().float()
        target_sum = target_c.sum().float()

        # Dice
        if pred_sum + target_sum > 0:
            dice = 2 * intersection / (pred_sum + target_sum)
        else:
            dice = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0

        # IoU
        union = pred_sum + target_sum - intersection
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0

        metrics[f'dice_{name}'] = dice.item()
        metrics[f'iou_{name}'] = iou.item()

    metrics['dice_mean'] = (metrics['dice_left'] + metrics['dice_right']) / 2
    metrics['iou_mean'] = (metrics['iou_left'] + metrics['iou_right']) / 2

    return metrics


# ============================================================
# 5. 可视化工具
# ============================================================

def plot_training_history(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    # 标记数据增强开始的epoch
    if history.get('augmentation_start_epoch', 0) > 0:
        axes[0].axvline(x=history['augmentation_start_epoch'] - 0.5,
                        color='green', linestyle='--', alpha=0.7,
                        label='Augmentation Start')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Dice 曲线
    axes[1].plot(history['train_dice'], label='Train Dice', color='blue')
    axes[1].plot(history['val_dice'], label='Val Dice', color='red')
    if history.get('augmentation_start_epoch', 0) > 0:
        axes[1].axvline(x=history['augmentation_start_epoch'] - 0.5,
                        color='green', linestyle='--', alpha=0.7,
                        label='Augmentation Start')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_title('Dice Score')
    axes[1].legend()
    axes[1].grid(True)

    # 学习率曲线
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
    """保存测试结果"""
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"测试结果")
    print(f"{'=' * 50}")
    print(f"左颈总动脉 - Dice: {test_metrics['dice_left']:.4f}, IoU: {test_metrics['iou_left']:.4f}")
    print(f"右颈总动脉 - Dice: {test_metrics['dice_right']:.4f}, IoU: {test_metrics['iou_right']:.4f}")
    print(f"平均 Dice: {test_metrics['dice_mean']:.4f}")
    print(f"平均 IoU: {test_metrics['iou_mean']:.4f}")
    print(f"{'=' * 50}")


# ============================================================
# 6. 训练器（支持分阶段数据增强 + 独立参数更新）
# ============================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-4, epochs=200, device='cuda',
                 save_dir=None, patience=7, min_delta=0.0003,
                 phase1_epochs=0):  # 新增参数：不使用数据增强的轮数
        """
        Args:
            phase1_epochs: 不使用数据增强的轮数
                          - 0: 从一开始就使用数据增强
                          - 8: 前8轮不使用数据增强，第9轮开始使用
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')
        self.patience = patience
        self.min_delta = min_delta
        self.phase1_epochs = phase1_epochs  # 不使用数据增强的轮数

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # 学习率调度器 (余弦退火 + warm-up)
        warmup_epochs = 20
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs

        # 损失函数
        self.criterion = CombinedLoss(weight_dice=1.0, weight_bce=1.0)

        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'lr': [],
            'augmentation_start_epoch': phase1_epochs  # 记录开始增强的epoch
        }

        # 早停
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_dice = 0.0
        self.patience_counter = 0

    def _should_use_augmentation(self, epoch):
        """判断当前epoch是否应该使用数据增强"""
        return epoch >= self.phase1_epochs

    def train_epoch_without_augmentation(self, epoch):
        """
        不使用数据增强的训练（每个batch只有原始数据）
        """
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Train-NoAug]')

        for batch in pbar:
            # 不使用增强时，数据格式是 {'ap', 'lat', 'mask', ...}
            x_ap = batch['ap'].to(self.device)
            x_lat = batch['lat'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x_ap, x_lat)
            loss, _ = self.criterion(output, mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            metrics = compute_metrics(output, mask)
            total_dice += metrics['dice_mean']
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice_mean"]:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return avg_loss, avg_dice

    def train_epoch_with_augmentation(self, epoch):
        """
        使用数据增强的训练（每个batch包含原始、对比度增强、亮度增强三个版本）
        采用独立参数更新方式（每个版本独立更新参数，真正扩大数据集）
        """
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_steps = 0  # 总优化步数 = batch数 × 3
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Train-Aug]')

        for batch in pbar:
            mask = batch['mask'].to(self.device)

            # 获取三个版本的数据
            ap_original = batch['ap_original'].to(self.device)
            lat_original = batch['lat_original'].to(self.device)
            ap_contrast = batch['ap_contrast'].to(self.device)
            lat_contrast = batch['lat_contrast'].to(self.device)
            ap_brightness = batch['ap_brightness'].to(self.device)
            lat_brightness = batch['lat_brightness'].to(self.device)

            batch_loss_sum = 0
            batch_dice_sum = 0

            # 对每个版本独立进行前向+反向+更新
            for version_idx, (version_name, ap, lat) in enumerate([
                ('original', ap_original, lat_original),
                ('contrast', ap_contrast, lat_contrast),
                ('brightness', ap_brightness, lat_brightness)
            ]):
                self.optimizer.zero_grad()

                output = self.model(ap, lat)
                loss, _ = self.criterion(output, mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss_sum += loss.item()
                num_steps += 1

                # 计算Dice（用于显示）
                metrics = compute_metrics(output, mask)
                batch_dice_sum += metrics['dice_mean']

                # 进度条显示当前版本的信息
                pbar.set_postfix({
                    'ver': version_name[:3],
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice_mean"]:.4f}'
                })

            # 记录batch的平均损失和平均Dice
            avg_batch_loss = batch_loss_sum / 3
            avg_batch_dice = batch_dice_sum / 3
            total_loss += avg_batch_loss
            total_dice += avg_batch_dice
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        # 打印统计信息
        print(f"\n  数据增强统计: 总优化步数 = {num_steps} (相当于 {num_steps} 个独立样本)")

        return avg_loss, avg_dice

    def validate_epoch(self, epoch, use_augmentation=False):
        """
        验证一个epoch（不使用数据增强）
        use_augmentation参数仅用于显示描述
        """
        self.model.eval()
        total_loss = 0
        total_metrics = defaultdict(float)
        num_batches = 0

        desc = f'Epoch {epoch + 1}/{self.epochs} [Val]'
        pbar = tqdm(self.val_loader, desc=desc)

        with torch.no_grad():
            for batch in pbar:
                x_ap = batch['ap'].to(self.device)
                x_lat = batch['lat'].to(self.device)
                mask = batch['mask'].to(self.device)

                output = self.model(x_ap, x_lat)

                loss, _ = self.criterion(output, mask)
                total_loss += loss.item()

                metrics = compute_metrics(output, mask)
                for k, v in metrics.items():
                    total_metrics[k] += v

                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        return avg_loss, avg_metrics

    def train(self):
        """主训练函数（支持分阶段数据增强）"""
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 病例")
        print(f"验证集: {len(self.val_loader.dataset)} 病例")

        if self.phase1_epochs == 0:
            print(f"数据增强: 从第1轮开始启用")
            print(f"  - 对比度变化: ±15% 随机变化")
            print(f"  - 亮度变化: ±15% 随机变化")
            print(f"  - 训练策略: 独立参数更新 (每个病例产生3个独立训练样本)")
        else:
            print(f"数据增强: 前 {self.phase1_epochs} 轮不使用，第 {self.phase1_epochs + 1} 轮开始启用")
            print(f"  - 阶段1 (Epoch 1-{self.phase1_epochs}): 仅使用原始数据")
            print(f"  - 阶段2 (Epoch {self.phase1_epochs + 1}-{self.epochs}): 使用数据增强")

        print(f"早停条件: 验证损失 {self.patience} 轮内下降 < {self.min_delta}")

        for epoch in range(self.epochs):
            # 学习率 warm-up
            if epoch < self.warmup_epochs:
                lr = 1e-6 + (1e-4 - 1e-6) * epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # 判断当前epoch是否使用数据增强
            use_augmentation = self._should_use_augmentation(epoch)

            # 训练
            if use_augmentation:
                train_loss, train_dice = self.train_epoch_with_augmentation(epoch)
            else:
                train_loss, train_dice = self.train_epoch_without_augmentation(epoch)

            # 验证（始终不使用数据增强）
            val_loss, val_metrics = self.validate_epoch(epoch)

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_metrics['dice_mean'])
            self.history['lr'].append(current_lr)

            # 打印结果
            aug_status = "增强模式" if use_augmentation else "普通模式"
            print(f"\nEpoch {epoch + 1}/{self.epochs} [{aug_status}]")
            print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
            print(f"  Val: Loss={val_loss:.4f}, Dice={val_metrics['dice_mean']:.4f}")
            print(f"  Left: Dice={val_metrics['dice_left']:.4f}, Right: Dice={val_metrics['dice_right']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # 保存最佳模型 (基于 Dice)
            if val_metrics['dice_mean'] > self.best_dice:
                self.best_dice = val_metrics['dice_mean']
                self.best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'metrics': val_metrics,
                    'use_augmentation_at_best': use_augmentation
                }, self.save_dir / 'best_model.pth')
                print(f"  → 保存最佳模型 (Dice: {self.best_dice:.4f})")

            # 早停检查
            if epoch >= self.warmup_epochs:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"  EarlyStopping: {self.patience_counter}/{self.patience}")

                    if self.patience_counter >= self.patience:
                        print(f"\n早停触发！最佳轮次: {self.best_epoch + 1}")
                        break

        # 保存最终模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'phase1_epochs': self.phase1_epochs
        }, self.save_dir / 'final_model.pth')

        # 绘制训练曲线
        plot_training_history(self.history, self.save_dir / 'training_curves.png')

        # 保存历史记录
        history_json = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'train_dice': [float(x) for x in self.history['train_dice']],
            'val_dice': [float(x) for x in self.history['val_dice']],
            'lr': [float(x) for x in self.history['lr']],
            'augmentation_start_epoch': self.phase1_epochs
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history_json, f, indent=2)

        return self.history

    def test(self):
        """测试最佳模型"""
        print(f"\n{'=' * 50}")
        print(f"开始测试...")
        print(f"{'=' * 50}")

        # 加载最佳模型
        checkpoint = torch.load(self.save_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (Dice: {checkpoint['best_dice']:.4f})")
        if 'use_augmentation_at_best' in checkpoint:
            print(f"最佳模型训练时是否使用增强: {checkpoint['use_augmentation_at_best']}")

        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = 0

        # 创建测试结果目录
        test_dir = self.save_dir / 'test'
        test_dir.mkdir(exist_ok=True)

        pbar = tqdm(self.test_loader, desc="Testing")

        with torch.no_grad():
            for batch in pbar:
                x_ap = batch['ap'].to(self.device)
                x_lat = batch['lat'].to(self.device)
                mask = batch['mask'].to(self.device)
                case_names = batch['case_name']

                output = self.model(x_ap, x_lat)

                # 计算指标
                metrics = compute_metrics(output, mask)
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1

                # 保存预测结果
                pred_mask = torch.sigmoid(output).cpu().numpy()

                for i, case_name in enumerate(case_names):
                    # 转换回原始形状 (D, H, W, C)
                    pred_single = pred_mask[i].transpose(1, 2, 3, 0)  # (D, H, W, 2)

                    # 保存为 nii.gz
                    nib.save(
                        nib.Nifti1Image(pred_single.astype(np.float32), np.eye(4)),
                        test_dir / f"{case_name}_pred.nii.gz"
                    )

                    # 保存 ground truth
                    mask_single = mask[i].cpu().numpy().transpose(1, 2, 3, 0)  # (D, H, W, 2)
                    nib.save(
                        nib.Nifti1Image(mask_single.astype(np.float32), np.eye(4)),
                        test_dir / f"{case_name}_gt.nii.gz"
                    )

                pbar.set_postfix({'dice': metrics['dice_mean']})

        # 计算平均指标
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        # 保存测试结果
        save_test_results(avg_metrics, test_dir)

        return avg_metrics


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("X2Shape 颈总动脉分割训练 (双通道输出，支持分阶段数据增强)")
    print("=" * 60)

    # ============================================================
    # 配置参数（可在此修改）
    # ============================================================

    # 分阶段训练配置
    PHASE1_EPOCHS = 0  # 不使用数据增强的轮数
    # PHASE1_EPOCHS = 0: 从一开始就使用数据增强
    # PHASE1_EPOCHS = 8: 前8轮不使用增强，第9轮开始使用增强

    # 训练参数
    BATCH_SIZE = 4
    NUM_EPOCHS = 60
    LEARNING_RATE = 0.0005
    PATIENCE = 7
    MIN_DELTA = 0.0003
    TRAIN_RATIO = 0.8
    SEED = 42

    # ============================================================

    print(f"训练配置:")
    print(f"  - 批量大小: {BATCH_SIZE}")
    print(f"  - 总轮数: {NUM_EPOCHS}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 早停耐心值: {PATIENCE}")
    print(f"  - 训练集比例: {TRAIN_RATIO}")

    if PHASE1_EPOCHS == 0:
        print(f"  - 数据增强: 从第1轮开始启用")
    else:
        print(f"  - 数据增强: 前 {PHASE1_EPOCHS} 轮不使用，第 {PHASE1_EPOCHS + 1} 轮开始启用")

    print(f"\n数据增强策略:")
    print(f"  - 对比度变化: ±15% 随机变化")
    print(f"  - 亮度变化: ±15% 随机变化")
    print(f"  - 训练策略: 独立参数更新 (每个病例产生3个独立训练样本)")
    print(f"  - 等效训练样本数: 原样本数 × 3")

    print(f"\n数据目录: {DATA_DIR}")
    print(f"缓存目录: {CACHE_ROOT}")
    print(f"模型目录: {MODEL_ROOT}")

    # 检查数据目录是否存在
    if not DATA_DIR.exists():
        print(f"\n错误: 数据目录不存在: {DATA_DIR}")
        print("请确保:")
        print("1. Windows 路径 D:\\med_data\\biron\\data\\train_all 存在")
        print("2. 在 WSL2 中可以通过 /mnt/d/ 访问 D 盘")
        return

    # 检查是否已有缓存
    existing_caches = [d for d in CACHE_ROOT.iterdir() if d.is_dir()] if CACHE_ROOT.exists() else []

    if existing_caches:
        print(f"\n发现已有缓存目录:")
        for cache in existing_caches:
            print(f"  - {cache.name}")
        use_existing = input("是否使用最新的缓存? (y/n): ").strip().lower()

        if use_existing == 'y':
            latest_cache = sorted(existing_caches)[-1]
            cache_dir = latest_cache

            with open(cache_dir / 'split_info.json', 'r') as f:
                split_info = json.load(f)

            print(f"使用缓存: {cache_dir}")
            train_cases = split_info['train_cases']
            val_cases = split_info['val_cases']
        else:
            cache_dir, splits = preprocess_and_cache(
                data_dir=DATA_DIR,
                cache_root=CACHE_ROOT,
                img_size=128,
                train_ratio=TRAIN_RATIO,
                seed=SEED
            )
            train_cases = splits['train']
            val_cases = splits['val']
    else:
        cache_dir, splits = preprocess_and_cache(
            data_dir=DATA_DIR,
            cache_root=CACHE_ROOT,
            img_size=128,
            train_ratio=TRAIN_RATIO,
            seed=SEED
        )
        train_cases = splits['train']
        val_cases = splits['val']

    # 创建数据集
    # 注意：训练集始终设置 augment=True，在训练过程中会根据 epoch 判断是否实际使用增强
    # 这是因为 Dataset 负责生成增强数据，Trainer 负责决定是否使用
    train_dataset = CarotidDataset(cache_dir, train_cases, augment=True)
    val_dataset = CarotidDataset(cache_dir, val_cases, augment=False)
    test_dataset = CarotidDataset(cache_dir, val_cases, augment=False)

    # 数据加载器配置
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 创建模型保存目录
    aug_suffix = "aug" if PHASE1_EPOCHS == 0 else f"aug_from_{PHASE1_EPOCHS + 1}"
    model_save_dir = MODEL_ROOT / f"{cache_dir.name}_{aug_suffix}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n模型保存目录: {model_save_dir}")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = X2Shape(
        img_size=128,
        in_chans=1,
        num_classes=2,  # 双通道输出
        dims_2d=[64, 128, 256],
        depths_2d=[1, 1, 2],
        dims_3d=[64, 128, 256],
        vbp_output_channels=64
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        device=device,
        save_dir=model_save_dir,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        phase1_epochs=PHASE1_EPOCHS  # 分阶段训练参数
    )

    # 开始训练
    history = trainer.train()

    # 测试
    test_metrics = trainer.test()

    print(f"\n训练完成！")
    print(f"最佳验证 Dice: {trainer.best_dice:.4f} (Epoch {trainer.best_epoch + 1})")
    print(f"模型保存在: {model_save_dir}")
    print(f"测试结果保存在: {model_save_dir}/test/")


if __name__ == "__main__":
    main()