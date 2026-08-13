"""
X2Shape 完整训练脚本 - 子图版本 (按病例批次，批量=4子图)
模型导入: from model_new_any.subvolume_unet import SubvolumeUNet
数据格式: 每个样本包含一个病例的所有子图 (8, C, 128, 128, 128)
批次大小: 4 个子图 (即 0.5 个病例)
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
from torch.amp import autocast, GradScaler

# 导入子图提取模块和分割模型
from model_new_any.VBP_unshuffle import VBPSubvolumeExtractor
from model_new_any.segment_mamba import SubvolumeUNet

# ============================================================
# 路径配置
# ============================================================

TRAIN_DATA_DIR = Path('/mnt/d/med_data/biron/data1/train_any')
TEST_DATA_DIR = Path('/mnt/d/med_data/biron/data1/test_any1')
CACHE_ROOT = Path('/mnt/d/med_data/biron/cache_subvolumes')
MODEL_ROOT = Path('/mnt/d/med_data/biron/model_subvolume')


# ============================================================
# 1. 数据预处理（缓存子图到 .npz 文件）
# ============================================================

def generate_cache_id():
    date_str = datetime.now().strftime("%Y%m%d")
    random_num = random.randint(100, 999)
    return f"{date_str}_{random_num}"


def parse_angle_from_folder(folder_name):
    try:
        parts = folder_name.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            angle = int(parts[1])
            if angle in [30, 50, 70, 90]:
                return angle
            return 90
    except:
        pass
    return 90


def preprocess_and_cache_subvolumes(data_dir, cache_root, split_name,
                                    volume_shape=(256, 256, 256),
                                    scale_factor=2,
                                    reverse_y=False):
    data_dir = Path(data_dir)
    cache_root = Path(cache_root)

    cache_id = generate_cache_id()
    cache_dir = cache_root / cache_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"数据预处理 - {split_name} (子图提取)")
    print(f"{'=' * 60}")
    print(f"原始数据目录: {data_dir}")
    print(f"缓存目录: {split_dir}")
    print(f"体积尺寸: {volume_shape}")
    print(f"子图尺寸: {volume_shape[0] // scale_factor}³")

    extractor = VBPSubvolumeExtractor(
        volume_shape=volume_shape,
        scale_factor=scale_factor
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.to(device)
    extractor.eval()

    case_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    case_dirs.sort()

    valid_cases = []
    case_angles = {}

    for case_dir in case_dirs:
        ap_path = case_dir / "ap.nii.gz"
        lat_path = case_dir / "lat.nii.gz"
        mask_path = case_dir / "mask.nii.gz"

        if ap_path.exists() and lat_path.exists() and mask_path.exists():
            case_name = case_dir.name
            valid_cases.append(case_name)
            case_angles[case_name] = parse_angle_from_folder(case_name)

    print(f"有效病例: {len(valid_cases)}")

    split_info = {
        'cache_id': cache_id,
        'split_name': split_name,
        'cases': valid_cases,
        'case_angles': case_angles,
        'num_cases': len(valid_cases),
        'volume_shape': volume_shape,
        'scale_factor': scale_factor,
        'subvolume_shape': [volume_shape[0] // scale_factor] * 3,
        'num_subvolumes': scale_factor ** 3
    }

    with open(split_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n处理 {split_name} 数据...")

    for case_name in tqdm(valid_cases, desc=f"{split_name} cases"):
        case_dir = data_dir / case_name

        ap = nib.load(case_dir / "ap.nii.gz").get_fdata().astype(np.float32)
        lat = nib.load(case_dir / "lat.nii.gz").get_fdata().astype(np.float32)
        mask = nib.load(case_dir / "mask.nii.gz").get_fdata().astype(np.float32)

        angle = case_angles[case_name]

        ap_tensor = torch.from_numpy(ap).unsqueeze(0).unsqueeze(0)
        lat_tensor = torch.from_numpy(lat).unsqueeze(0).unsqueeze(0)

        if ap_tensor.shape[-2:] != (volume_shape[0], volume_shape[2]):
            ap_tensor = F.interpolate(ap_tensor, size=(volume_shape[0], volume_shape[2]),
                                      mode='bilinear', align_corners=False)
            lat_tensor = F.interpolate(lat_tensor, size=(volume_shape[0], volume_shape[2]),
                                       mode='bilinear', align_corners=False)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        if mask_tensor.shape[-3:] != tuple(volume_shape):
            mask_tensor = F.interpolate(mask_tensor, size=tuple(volume_shape),
                                        mode='trilinear', align_corners=False)

        ap_tensor = ap_tensor.to(device)
        lat_tensor = lat_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            subvolumes = extractor(ap_tensor, lat_tensor, angle=angle, reverse_y=reverse_y)
            mask_subvolumes = extractor.extractor(mask_tensor)

        # 保存为单个npz文件 (包含所有子图)
        np.savez(
            split_dir / f"{case_name}.npz",
            subvolumes=subvolumes.cpu().numpy(),  # (8, 1, S, S, S)
            masks=mask_subvolumes.cpu().numpy(),  # (8, 1, S, S, S)
            angle=np.array(angle, dtype=np.int32)
        )

    print(f"\n预处理完成！")
    print(f"{split_name} 集: {len(valid_cases)} 个病例")
    print(f"每个病例: {scale_factor ** 3} 个子图")
    print(f"子图尺寸: {volume_shape[0] // scale_factor}³")

    return split_dir, valid_cases, case_angles


def preprocess_all_data(train_dir, test_dir, cache_root,
                        volume_shape=(256, 256, 256),
                        scale_factor=2, reverse_y=False):
    print(f"\n{'=' * 60}")
    print(f"开始预处理所有数据 (子图版本)")
    print(f"{'=' * 60}")

    train_cache_dir, train_cases, train_angles = preprocess_and_cache_subvolumes(
        data_dir=train_dir,
        cache_root=cache_root,
        split_name='train',
        volume_shape=volume_shape,
        scale_factor=scale_factor,
        reverse_y=reverse_y
    )

    val_cache_dir, val_cases, val_angles = preprocess_and_cache_subvolumes(
        data_dir=test_dir,
        cache_root=cache_root,
        split_name='val_test',
        volume_shape=volume_shape,
        scale_factor=scale_factor,
        reverse_y=reverse_y
    )

    total_cache_dir = train_cache_dir.parent
    total_info = {
        'cache_id': total_cache_dir.name,
        'volume_shape': volume_shape,
        'scale_factor': scale_factor,
        'subvolume_shape': [volume_shape[0] // scale_factor] * 3,
        'train_cache_dir': str(train_cache_dir),
        'val_cache_dir': str(val_cache_dir),
        'num_train': len(train_cases),
        'num_val': len(val_cases),
        'train_cases': train_cases,
        'val_cases': val_cases,
        'train_angles': train_angles,
        'val_angles': val_angles
    }

    with open(total_cache_dir / 'total_split_info.json', 'w') as f:
        json.dump(total_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"所有数据预处理完成！")
    print(f"训练集: {len(train_cases)} 个病例")
    print(f"验证集: {len(val_cases)} 个病例")
    print(f"总缓存目录: {total_cache_dir}")

    return train_cache_dir, train_cases, train_angles, val_cache_dir, val_cases, val_angles


# ============================================================
# 2. 数据集类 - 按病例组织，每次返回所有子图
# ============================================================

class SubvolumeDataset(Dataset):
    """
    按病例组织数据，每次返回一个病例的所有子图
    每个样本: (8, C, S, S, S) 和对应的 (8, C, S, S, S) mask
    """

    def __init__(self, cache_dir, case_list):
        self.cache_dir = Path(cache_dir)
        self.case_list = case_list
        self.file_paths = [self.cache_dir / f"{case}.npz" for case in case_list]

        with open(self.cache_dir / 'split_info.json', 'r') as f:
            split_info = json.load(f)
        self.subvolume_shape = split_info.get('subvolume_shape', [128, 128, 128])
        self.num_subvolumes = split_info.get('num_subvolumes', 8)

    def __len__(self):
        return len(self.case_list)  # 病例数

    def __getitem__(self, idx):
        case_name = self.case_list[idx]
        data = np.load(self.file_paths[idx])

        # 返回所有子图 (8个)
        subvolumes = data['subvolumes']  # (8, C, S, S, S)
        masks = data['masks']            # (8, C, S, S, S)
        angle = int(data['angle']) if 'angle' in data else 90

        return {
            'subvolumes': torch.from_numpy(subvolumes).float(),  # (8, 1, S, S, S)
            'masks': torch.from_numpy(masks).float(),            # (8, 1, S, S, S)
            'angle': torch.tensor(angle, dtype=torch.long),
            'case_name': case_name
        }


# ============================================================
# 3. Collate函数 - 将病例批次展平为子图批次
# ============================================================

def collate_subvolumes(batch):
    """
    将批次中的病例数据展平为子图数据

    输入: batch = [
        {'subvolumes': (8, C, S, S, S), 'masks': (8, C, S, S, S), ...},
        {'subvolumes': (8, C, S, S, S), 'masks': (8, C, S, S, S), ...},
    ]
    输出: {
        'subvolumes': (B*8, C, S, S, S),
        'masks': (B*8, C, S, S, S),
        'case_names': [...],
        'subvolume_indices': [...]
    }
    """
    subvolumes_list = []
    masks_list = []
    case_names = []
    subvolume_indices = []

    for b_idx, item in enumerate(batch):
        subvols = item['subvolumes']  # (8, C, S, S, S)
        masks = item['masks']          # (8, C, S, S, S)

        # 展平每个病例的8个子图
        for s_idx in range(subvols.shape[0]):
            subvolumes_list.append(subvols[s_idx])  # (C, S, S, S)
            masks_list.append(masks[s_idx])          # (C, S, S, S)
            case_names.append(item['case_name'])
            subvolume_indices.append(s_idx)

    return {
        'subvolumes': torch.stack(subvolumes_list, dim=0),  # (B*8, C, S, S, S)
        'masks': torch.stack(masks_list, dim=0),            # (B*8, C, S, S, S)
        'case_names': case_names,
        'subvolume_indices': subvolume_indices,
        'angle': batch[0]['angle']  # 同一批次角度相同
    }


# ============================================================
# 4. 损失函数、评估指标
# ============================================================

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
        return 1 - dice.mean()

    def focal_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (self.alpha * focal_weight * bce).mean()

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return self.weight_dice * dice_loss + self.weight_focal * focal_loss, \
               {'dice_loss': dice_loss, 'focal_loss': focal_loss}


def compute_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    target_flat = target_binary.view(target_binary.shape[0], -1)

    dice_list, iou_list, precision_list, recall_list = [], [], [], []

    for i in range(pred_flat.shape[0]):
        p = pred_flat[i]
        t = target_flat[i]
        intersection = (p * t).sum()
        pred_sum = p.sum()
        target_sum = t.sum()

        dice = 2 * intersection / (pred_sum + target_sum + 1e-6) if pred_sum + target_sum > 0 else 0.0
        dice_list.append(dice)

        union = pred_sum + target_sum - intersection
        iou = intersection / (union + 1e-6) if union > 0 else 0.0
        iou_list.append(iou)

        precision = intersection / (pred_sum + 1e-6) if pred_sum > 0 else 0.0
        precision_list.append(precision)

        recall = intersection / (target_sum + 1e-6) if target_sum > 0 else 0.0
        recall_list.append(recall)

    return {
        'dice': float(torch.tensor(dice_list).mean()),
        'iou': float(torch.tensor(iou_list).mean()),
        'precision': float(torch.tensor(precision_list).mean()),
        'recall': float(torch.tensor(recall_list).mean())
    }


# ============================================================
# 5. 训练器
# ============================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-4, epochs=200, device='cuda',
                 save_dir=None, patience=12, min_delta=0.0002,
                 lr_patience=4, min_lr=1e-6, start_reduce_epoch=10):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')

        self.patience = patience
        self.min_delta = min_delta
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.start_reduce_epoch = start_reduce_epoch

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.initial_lr = learning_rate
        self.criterion = DiceFocalLoss(weight_dice=1.0, weight_focal=4.0)
        self.scaler = GradScaler('cuda')

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'lr': []
        }

        self.best_dice = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.lr_patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Train]')

        for batch in pbar:
            # batch 已经由 collate_fn 展平
            subvolumes = batch['subvolumes'].to(self.device)  # (B*8, C, S, S, S)
            masks = batch['masks'].to(self.device)            # (B*8, C, S, S, S)

            self.optimizer.zero_grad()

            with autocast('cuda'):
                output = self.model(subvolumes)
                loss, loss_dict = self.criterion(output, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            metrics = compute_metrics(output, masks)
            total_dice += metrics['dice']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice"]:.4f}',
                'subs': subvolumes.shape[0]
            })

        return total_loss / max(num_batches, 1), total_dice / max(num_batches, 1)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.epochs} [Val]')

        with torch.no_grad():
            for batch in pbar:
                subvolumes = batch['subvolumes'].to(self.device)
                masks = batch['masks'].to(self.device)

                with autocast('cuda'):
                    output = self.model(subvolumes)
                    loss, _ = self.criterion(output, masks)

                total_loss += loss.item()
                metrics = compute_metrics(output, masks)
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice"]:.4f}'
                })

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        return avg_loss, avg_metrics

    def train(self):
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 病例 ({len(self.train_loader)} 批次/代)")
        print(f"验证集: {len(self.val_loader.dataset)} 病例 ({len(self.val_loader)} 批次/代)")
        print(f"学习率: {self.initial_lr:.2e}")
        print("=" * 60)

        for epoch in range(self.epochs):
            train_loss, train_dice = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
            print(f"  Val: Loss={val_loss:.4f}, Dice={val_metrics['dice']:.4f}")
            print(f"  Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.lr_patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'metrics': val_metrics
                }, self.save_dir / 'best_model.pth')
                print(f"  → 保存最佳模型 (Dice: {self.best_dice:.4f})")

            if epoch >= self.start_reduce_epoch:
                if val_metrics['dice'] > self.best_dice - self.min_delta:
                    self.patience_counter = 0
                    self.lr_patience_counter = 0
                else:
                    self.patience_counter += 1
                    self.lr_patience_counter += 1

                    if self.lr_patience_counter >= self.lr_patience:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        new_lr = max(current_lr * 0.5, self.min_lr)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        self.lr_patience_counter = 0
                        print(f"  → 学习率衰减: {current_lr:.2e} -> {new_lr:.2e}")

                    if self.patience_counter >= self.patience:
                        print(f"\n  → 触发早停 (连续 {self.patience} 轮未改进)")
                        break

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, self.save_dir / 'final_model.pth')

        return self.history

    def test(self):
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
            for batch in pbar:
                subvolumes = batch['subvolumes'].to(self.device)
                masks = batch['masks'].to(self.device)
                case_names = batch['case_names']
                sub_idx = batch['subvolume_indices']

                with autocast('cuda'):
                    output = self.model(subvolumes)

                metrics = compute_metrics(output, masks)
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1

                # 按病例保存
                pred_mask = torch.sigmoid(output).cpu().numpy()
                gt_mask = masks.cpu().numpy()

                for i, case_name in enumerate(case_names):
                    case_dir = test_dir / case_name
                    case_dir.mkdir(exist_ok=True)

                    idx = sub_idx[i]
                    pred_single = pred_mask[i, 0]
                    gt_single = gt_mask[i, 0]
                    pred_binary = (pred_single > 0.5).astype(np.float32)

                    nib.save(nib.Nifti1Image(pred_binary, np.eye(4)),
                            case_dir / f"pred_sub{idx:02d}.nii.gz")
                    nib.save(nib.Nifti1Image(gt_single.astype(np.float32), np.eye(4)),
                            case_dir / f"gt_sub{idx:02d}.nii.gz")

                pbar.set_postfix({'dice': metrics['dice']})

        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}

        with open(test_dir / 'test_results.json', 'w') as f:
            json.dump(avg_metrics, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"测试结果")
        print(f"{'=' * 50}")
        print(f"Dice: {avg_metrics['dice']:.4f}")
        print(f"IoU: {avg_metrics['iou']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall: {avg_metrics['recall']:.4f}")

        return avg_metrics


# ============================================================
# 6. 主函数
# ============================================================

def main():
    # ============================================================
    # 配置参数
    # ============================================================
    VOLUME_SHAPE = (256, 256, 256)
    SCALE_FACTOR = 2
    SUBVOLUME_SIZE = VOLUME_SHAPE[0] // SCALE_FACTOR
    REVERSE_Y = True
    BATCH_SIZE = 1 # 每批2个病例 = 16个子图 (您的电脑最大支持4个子图，但这是子图数)
                     # 所以病例数 = 4 // 8 = 0.5，但collate_fn会处理
    SUBVOLUME_BATCH = 2  # 实际输入模型的子图数
    EPOCHS = 60

    print("=" * 60)
    print("X2Shape 子图版本训练 (按病例批次)")
    print("=" * 60)
    print(f"体积尺寸: {VOLUME_SHAPE}")
    print(f"子图尺寸: {SUBVOLUME_SIZE}³")
    print(f"每个病例子图数: {SCALE_FACTOR ** 3}")
    print(f"批次大小: {SUBVOLUME_BATCH} 子图/批")
    print(f"Y轴方向: {'反向' if REVERSE_Y else '正向'}")

    if not TRAIN_DATA_DIR.exists() or not TEST_DATA_DIR.exists():
        print(f"\n错误: 数据目录不存在")
        return

    # 检查缓存
    existing_caches = [d for d in CACHE_ROOT.iterdir() if d.is_dir()] if CACHE_ROOT.exists() else []

    train_cache_dir = None
    val_cache_dir = None

    if existing_caches:
        print(f"\n发现已有缓存目录:")
        for cache in existing_caches:
            if (cache / 'total_split_info.json').exists():
                with open(cache / 'total_split_info.json', 'r') as f:
                    info = json.load(f)
                    if info.get('subvolume_shape', []) == [SUBVOLUME_SIZE] * 3:
                        print(f"  - {cache.name} (子图: {SUBVOLUME_SIZE}³, 训练: {info['num_train']}, 验证: {info['num_val']})")

        use_existing = input("\n是否使用最新的缓存? (y/n): ").strip().lower()

        if use_existing == 'y':
            for cache in sorted(existing_caches, reverse=True):
                if (cache / 'total_split_info.json').exists():
                    with open(cache / 'total_split_info.json', 'r') as f:
                        info = json.load(f)
                        if info.get('subvolume_shape', []) == [SUBVOLUME_SIZE] * 3:
                            train_cache_dir = Path(info['train_cache_dir'])
                            val_cache_dir = Path(info['val_cache_dir'])
                            train_cases = info['train_cases']
                            val_cases = info['val_cases']
                            print(f"\n使用缓存: {cache}")
                            break

    if train_cache_dir is None:
        train_cache_dir, train_cases, train_angles, val_cache_dir, val_cases, val_angles = preprocess_all_data(
            train_dir=TRAIN_DATA_DIR,
            test_dir=TEST_DATA_DIR,
            cache_root=CACHE_ROOT,
            volume_shape=VOLUME_SHAPE,
            scale_factor=SCALE_FACTOR,
            reverse_y=REVERSE_Y
        )

    # 创建数据集 - 按病例组织
    train_dataset = SubvolumeDataset(train_cache_dir, train_cases)
    val_dataset = SubvolumeDataset(val_cache_dir, val_cases)
    test_dataset = SubvolumeDataset(val_cache_dir, val_cases)

    # 批次大小 = 病例数 (使得子图数不超过SUBVOLUME_BATCH)
    # 每个病例8个子图，所以病例数 = SUBVOLUME_BATCH // 8
    case_batch_size = max(1, SUBVOLUME_BATCH // 8)  # 4//8=0.5 -> 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=case_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_subvolumes,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=case_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_subvolumes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=case_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_subvolumes
    )

    print(f"\n训练集: {len(train_dataset)} 病例 ({len(train_loader)} 批次/代)")
    print(f"验证集: {len(val_dataset)} 病例 ({len(val_loader)} 批次/代)")
    print(f"每批子图数: {case_batch_size * 8}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_dir = MODEL_ROOT / train_cache_dir.parent.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型
    model = SubvolumeUNet(
        in_chans=1,
        num_classes=1,
        subvolume_size=SUBVOLUME_SIZE,
        encoder_channels=[16, 32, 64, 128],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=1e-4,
        epochs=EPOCHS,
        device=device,
        save_dir=model_save_dir,
        patience=15,
        min_delta=0.0005,
        lr_patience=5,
        min_lr=1e-6,
        start_reduce_epoch=10
    )

    history = trainer.train()
    test_metrics = trainer.test()

    print(f"\n训练完成！")
    print(f"最佳验证 Dice: {trainer.best_dice:.4f} (Epoch {trainer.best_epoch + 1})")
    print(f"模型保存在: {model_save_dir}")


if __name__ == "__main__":
    main()