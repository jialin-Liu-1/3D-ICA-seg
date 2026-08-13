"""
X2Shape 完整训练脚本 - WSL2 兼容版
路径使用 /mnt/d/ 格式
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
import shutil
import json
from collections import defaultdict


# ============================================================
# 0. 路径配置 (WSL2 兼容)
# ============================================================

# Windows 路径 -> WSL2 路径映射
# 注意: 如果你的用户名不是 lenovo，请修改
PATHS = {
    'data_dir': '/mnt/d/med_data/biron/data/train_all',      # 原始数据
    'cache_root': '/mnt/d/med_data/biron/cache',             # 缓存目录
    'model_root': '/mnt/d/med_data/biron/model',             # 模型保存目录
}


# ============================================================
# 1. 数据预处理（缓存到 .npz 文件）
# ============================================================

def generate_cache_id():
    """生成缓存文件夹ID：日期_三位随机数"""
    date_str = datetime.now().strftime("%Y%m%d")
    random_num = random.randint(100, 999)
    return f"{date_str}_{random_num}"


def preprocess_and_cache(data_dir, cache_root,
                         img_size=128, train_ratio=0.8, seed=42):
    """
    预处理数据并保存为 .npz 缓存文件
    """
    data_dir = Path(data_dir)
    cache_root = Path(cache_root)

    # 创建缓存目录
    cache_id = generate_cache_id()
    cache_dir = cache_root / cache_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"数据预处理")
    print(f"{'='*60}")
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
        ap = ap.squeeze()[None, :, :]  # (1, 128, 128)
        lat = lat.squeeze()[None, :, :]  # (1, 128, 128)
        mask = np.transpose(mask, (3, 0, 1, 2))  # (3, 128, 128, 128)

        # 保存为 npz 文件
        np.savez(
            cache_dir / f"{case_name}.npz",
            ap=ap.astype(np.float32),
            lat=lat.astype(np.float32),
            mask=mask.astype(np.float32)
        )

    print("\n处理验证集...")
    for case_name in tqdm(val_cases, desc="Validation cases"):
        case_dir = data_dir / case_name

        ap = nib.load(case_dir / "ap.nii.gz").get_fdata()
        lat = nib.load(case_dir / "lat.nii.gz").get_fdata()
        mask = nib.load(case_dir / "mask.nii.gz").get_fdata()

        ap = ap.squeeze()[None, :, :]
        lat = lat.squeeze()[None, :, :]
        mask = np.transpose(mask, (3, 0, 1, 2))

        np.savez(
            cache_dir / f"{case_name}.npz",
            ap=ap.astype(np.float32),
            lat=lat.astype(np.float32),
            mask=mask.astype(np.float32)
        )

    print(f"\n预处理完成！")
    print(f"训练集: {len(train_cases)} 个病例")
    print(f"验证集: {len(val_cases)} 个病例")
    print(f"缓存目录: {cache_dir}")

    return cache_dir, splits


# ============================================================
# 2. 数据集类（流式读取）
# ============================================================

class CarotidDataset(Dataset):
    """从缓存文件读取数据"""
    def __init__(self, cache_dir, case_list):
        self.cache_dir = Path(cache_dir)
        self.case_list = case_list
        self.file_paths = [self.cache_dir / f"{case}.npz" for case in case_list]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        ap = torch.from_numpy(data['ap'])      # (1, 128, 128)
        lat = torch.from_numpy(data['lat'])    # (1, 128, 128)
        mask = torch.from_numpy(data['mask'])  # (3, 128, 128, 128)

        return {
            'x_ap': ap,
            'x_lat': lat,
            'mask': mask,
            'case_name': self.case_list[idx]
        }


# ============================================================
# 3. 损失函数（多通道二分类）
# ============================================================

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

        return 1 - dice.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss


# ============================================================
# 4. 评估指标
# ============================================================

def compute_metrics(pred, target, threshold=0.5):
    """
    计算分割指标
    pred: (B, C, D, H, W) logits
    target: (B, C, D, H, W) binary mask
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    metrics = {}

    for ch, name in [(1, 'left'), (2, 'right')]:
        pred_c = pred_binary[:, ch, ...]
        target_c = target[:, ch, ...]

        intersection = (pred_c * target_c).sum().float()
        pred_sum = pred_c.sum().float()
        target_sum = target_c.sum().float()

        if pred_sum + target_sum > 0:
            dice = 2 * intersection / (pred_sum + target_sum)
        else:
            dice = 1.0 if (pred_sum == 0 and target_sum == 0) else 0.0

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

    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
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
    """保存测试结果"""
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"测试结果")
    print(f"{'='*50}")
    print(f"左颈总动脉 - Dice: {test_metrics['dice_left']:.4f}, IoU: {test_metrics['iou_left']:.4f}")
    print(f"右颈总动脉 - Dice: {test_metrics['dice_right']:.4f}, IoU: {test_metrics['iou_right']:.4f}")
    print(f"平均 Dice: {test_metrics['dice_mean']:.4f}")
    print(f"平均 IoU: {test_metrics['iou_mean']:.4f}")
    print(f"{'='*50}")


# ============================================================
# 6. 训练器
# ============================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-4, epochs=200, device='cuda',
                 save_dir=None, patience=7, min_delta=0.0003):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')
        self.patience = patience
        self.min_delta = min_delta

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        warmup_epochs = 20
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs

        self.criterion = CombinedLoss(weight_dice=1.0, weight_bce=1.0)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'lr': []
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_dice = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')

        for batch in pbar:
            x_ap = batch['x_ap'].to(self.device)
            x_lat = batch['x_lat'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x_ap, x_lat)

            loss, loss_dict = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            metrics = compute_metrics(output, mask)
            total_dice += metrics['dice_mean']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice_mean"]:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return avg_loss, avg_dice

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]')

        with torch.no_grad():
            for batch in pbar:
                x_ap = batch['x_ap'].to(self.device)
                x_lat = batch['x_lat'].to(self.device)
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
        print(f"\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集: {len(self.train_loader.dataset)} 病例")
        print(f"验证集: {len(self.val_loader.dataset)} 病例")
        print(f"早停条件: 验证损失 {self.patience} 轮内下降 < {self.min_delta}")

        for epoch in range(self.epochs):
            if epoch < self.warmup_epochs:
                lr = 1e-6 + (1e-4 - 1e-6) * epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            train_loss, train_dice = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_metrics['dice_mean'])
            self.history['lr'].append(current_lr)

            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
            print(f"  Val: Loss={val_loss:.4f}, Dice={val_metrics['dice_mean']:.4f}")
            print(f"  Left: Dice={val_metrics['dice_left']:.4f}, Right: Dice={val_metrics['dice_right']:.4f}")

            if val_metrics['dice_mean'] > self.best_dice:
                self.best_dice = val_metrics['dice_mean']
                self.best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'metrics': val_metrics
                }, self.save_dir / 'best_model.pth')
                print(f"  → 保存最佳模型 (Dice: {self.best_dice:.4f})")

            if epoch >= self.warmup_epochs:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"  EarlyStopping: {self.patience_counter}/{self.patience}")

                    if self.patience_counter >= self.patience:
                        print(f"\n早停触发！最佳轮次: {self.best_epoch+1}")
                        break

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
        print(f"\n{'='*50}")
        print(f"开始测试...")
        print(f"{'='*50}")

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

                output = self.model(x_ap, x_lat)

                metrics = compute_metrics(output, mask)
                for k, v in metrics.items():
                    total_metrics[k] += v
                num_batches += 1

                pred_mask = torch.sigmoid(output).cpu().numpy()

                for i, case_name in enumerate(case_names):
                    pred_single = pred_mask[i].transpose(1, 2, 3, 0)
                    nib.save(
                        nib.Nifti1Image(pred_single.astype(np.float32), np.eye(4)),
                        test_dir / f"{case_name}_pred.nii.gz"
                    )

                    mask_single = mask[i].cpu().numpy().transpose(1, 2, 3, 0)
                    nib.save(
                        nib.Nifti1Image(mask_single.astype(np.float32), np.eye(4)),
                        test_dir / f"{case_name}_gt.nii.gz"
                    )

                pbar.set_postfix({'dice': metrics['dice_mean']})

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        save_test_results(avg_metrics, test_dir)

        return avg_metrics


# ============================================================
# 7. 主函数
# ============================================================

def main():
    # 使用 WSL2 兼容的路径
    data_dir = Path('/mnt/d/med_data/biron/data/train_all')
    cache_root = Path('/mnt/d/med_data/biron/cache')
    model_root = Path('/mnt/d/med_data/biron/model')

    # 验证路径是否存在
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请确保:")
        print("1. Windows 路径 D:\\med_data\\biron\\data\\train_all 存在")
        print("2. 在 WSL2 中可以通过 /mnt/d/ 访问 D 盘")
        return

    print("="*60)
    print("X2Shape 颈总动脉分割训练 (WSL2)")
    print("="*60)
    print(f"数据目录: {data_dir}")
    print(f"缓存目录: {cache_root}")
    print(f"模型目录: {model_root}")

    # 检查是否已有缓存
    existing_caches = [d for d in cache_root.iterdir() if d.is_dir()] if cache_root.exists() else []

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
                data_dir=data_dir,
                cache_root=cache_root,
                img_size=128,
                train_ratio=0.8,
                seed=42
            )
            train_cases = splits['train']
            val_cases = splits['val']
    else:
        cache_dir, splits = preprocess_and_cache(
            data_dir=data_dir,
            cache_root=cache_root,
            img_size=128,
            train_ratio=0.8,
            seed=42
        )
        train_cases = splits['train']
        val_cases = splits['val']

    # 创建数据集
    train_dataset = CarotidDataset(cache_dir, train_cases)
    val_dataset = CarotidDataset(cache_dir, val_cases)
    test_dataset = CarotidDataset(cache_dir, val_cases)

    # 数据加载器配置
    batch_size = 4
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 创建模型保存目录
    model_save_dir = model_root / cache_dir.name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n模型保存目录: {model_save_dir}")

    # 导入模型
    from model.test0 import X2Shape

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = X2Shape(
        img_size=128, in_chans=1, num_classes=2,
        dims_2d=[56, 140, 280],
        depths_2d=[1, 1, 2],
        dims_3d=[70, 140, 280],
        vbp_output_channels=70
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
        learning_rate=1e-4,
        epochs=200,
        device=device,
        save_dir=model_save_dir,
        patience=7,
        min_delta=0.0003
    )

    # 开始训练
    history = trainer.train()

    # 测试
    test_metrics = trainer.test()

    print(f"\n训练完成！")
    print(f"最佳验证 Dice: {trainer.best_dice:.4f} (Epoch {trainer.best_epoch+1})")
    print(f"模型保存在: {model_save_dir}")
    print(f"测试结果保存在: {model_save_dir}/test/")


if __name__ == "__main__":
    main()