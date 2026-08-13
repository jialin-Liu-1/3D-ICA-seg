import numpy as np
import nibabel as nib
import os
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from scipy import ndimage as ndi
from skimage import measure
import warnings
import time

# ============================================================
# GPU加速相关导入 - 增强兼容性
# ============================================================
GPU_AVAILABLE = False
GPU_BACKEND = None  # 'cupy' or 'cupyx'

try:
    import cupy as cp

    # 检查CuPy版本并兼容处理
    try:
        # 新版本CuPy
        gpu_name = cp.cuda.Device(0).name
        gpu_memory = cp.cuda.Device(0).mem_info[1] / 1024 ** 3
        GPU_AVAILABLE = True
        GPU_BACKEND = 'cupy'
        print("✅ GPU可用，启用CuPy加速")
        print(f"   GPU设备: {gpu_name}")
        print(f"   GPU内存: {gpu_memory:.1f} GB")
    except AttributeError:
        # 旧版本CuPy，尝试其他方式获取信息
        try:
            device = cp.cuda.Device(0)
            # 旧版本可能没有name属性，使用id代替
            gpu_id = device.id
            # 尝试获取内存信息
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                gpu_memory = total / 1024 ** 3
                print("✅ GPU可用，启用CuPy加速 (旧版本)")
                print(f"   GPU ID: {gpu_id}")
                print(f"   GPU内存: {gpu_memory:.1f} GB")
                GPU_AVAILABLE = True
                GPU_BACKEND = 'cupy'
            except:
                print("✅ GPU可用，启用CuPy加速 (旧版本)")
                print(f"   GPU ID: {gpu_id}")
                GPU_AVAILABLE = True
                GPU_BACKEND = 'cupy'
        except:
            # 无法获取详细信息，但CuPy可用
            print("✅ GPU可用，启用CuPy加速")
            GPU_AVAILABLE = True
            GPU_BACKEND = 'cupy'

    # 尝试导入cupyx.scipy.ndimage
    if GPU_AVAILABLE:
        try:
            from cupyx.scipy.ndimage import rotate as gpu_rotate
            from cupyx.scipy.ndimage import zoom as gpu_zoom

            GPU_BACKEND = 'cupyx'
            print("   ✅ CuPyX加速模块可用")
        except ImportError:
            print("   ⚠️  CuPyX模块不可用，使用基础CuPy功能")

except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy未安装，使用CPU模式")
    print("   安装命令: pip install cupy-cuda12x (根据CUDA版本调整)")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"⚠️  GPU初始化失败: {e}，使用CPU模式")

warnings.filterwarnings('ignore')

# ============================================================
# 设置全局精度
# ============================================================
COMPUTE_DTYPE = np.float32
SAVE_DTYPE = np.float32

# ============================================================
# GPU加速开关
# ============================================================
USE_GPU = True  # 设置为False可强制使用CPU

# ============================================================
# 参数配置
# ============================================================

RAW_NII_BASE = r"D:\med_data\biron\data1\raw_nii"
MASK_TEST_BASE = r"D:\med_data\biron\data1\slicer"
TRAIN_BASE = r"D:\med_data\biron\data2\train_new"

# ============================================================
# 分辨率参数
# ============================================================
TARGET_3D_SIZE = 256
TARGET_2D_SIZE = 256

# ============================================================
# DSA图像参数
# ============================================================
VASCULAR_ATTENUATION = np.float32(0.05)
TISSUE_ATTENUATION = np.float32(0.03)
BACKGROUND_ATTENUATION = np.float32(0.02)
MAP_TO_REAL_ATTENUATION = True

# ============================================================
# 旋转参数
# ============================================================
ROTATION_ANGLES = [0]
PROJECTION_ANGLES = [70, 85, 100, 115]

# ============================================================
# 扇形束几何参数
# ============================================================
SOURCE_TO_ISOCENTER = 500.0
SOURCE_TO_DETECTOR = 1000.0
DETECTOR_WIDTH = 400.0
NUM_DETECTOR_CHANNELS = 256

# ============================================================
# 蒙泰卡罗物理参数
# ============================================================
XRAY_TUBE_VOLTAGE = 80.0
USE_SCATTER = False
SCATTER_FRACTION = np.float32(0.3)

# ============================================================
# 二值化参数（已废弃，保留仅为兼容性）
# ============================================================
BINARY_THRESHOLD = np.float32(0.2)

# ============================================================
# 投影参数
# ============================================================
PROJECTION_SCALE_FACTOR = 100.0

# ============================================================
# 保存选项
# ============================================================
INVERT_PROJECTION = True

# ============================================================
# 重叠图参数
# ============================================================
AP_ALPHA = 0.7
LAT_ALPHA = 0.7
MASK_ALPHA = 0.4

# ============================================================
# 投影轴定义
# ============================================================
PROJECTION_AXIS = 1
LAT_ROTATION_AXES = (1, 2)

# ============================================================
# 运行模式
# ============================================================
RUN_MODE = "batch"
TEST_CASE_NUM = 0
TEST_ROTATION_IDX = 0
TEST_PROJ_IDX = 0

# ============================================================
# 血管剪枝参数（已废弃，保留仅为兼容性）
# ============================================================
ENABLE_KEEP_LARGEST = False
VOLUME_THRESHOLD = 500

print("=" * 80)
print("DSA扇形束蒙泰卡罗模拟 + 自定义夹角双平面投影系统 (GPU加速版)")
print("=" * 80)
print(f"计算精度: {COMPUTE_DTYPE}, 保存精度: {SAVE_DTYPE}")
print(f"GPU加速: {'✅ 启用' if (USE_GPU and GPU_AVAILABLE) else '❌ 禁用'}")
print(f"投影反转保存: {'是 (血管变暗)' if INVERT_PROJECTION else '否 (血管变亮)'}")
print(f"投影轴定义:")
print(f"  - AP投影轴: axis={PROJECTION_AXIS} (沿Y轴)")
print(f"  - LAT旋转轴: axis={LAT_ROTATION_AXES} (绕X轴旋转后沿Y轴投影)")
print(f"  - AP和LAT夹角: 自定义 ({PROJECTION_ANGLES}°)")
print(f"Mask来源: {MASK_TEST_BASE} (预生成，直接加载)")
print(f"DSA图像尺寸: {TARGET_3D_SIZE}³, 投影尺寸: {TARGET_2D_SIZE}²")
print(f"整体旋转角度: {ROTATION_ANGLES}")
print(f"投影夹角: {PROJECTION_ANGLES}")
print(f"总样本数: {len(ROTATION_ANGLES)} × {len(PROJECTION_ANGLES)} = {len(ROTATION_ANGLES) * len(PROJECTION_ANGLES)}")
print("=" * 80)


# ============================================================
# GPU辅助函数
# ============================================================

def to_gpu(data):
    """将数据转移到GPU"""
    if USE_GPU and GPU_AVAILABLE:
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        return data
    return data


def to_cpu(data):
    """将数据从GPU转移到CPU"""
    if USE_GPU and GPU_AVAILABLE:
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data
    return data


def gpu_rotate_wrapper(image, angle, axes, reshape=False, order=1, cval=0):
    """GPU旋转包装器，自动处理CPU/GPU切换"""
    if USE_GPU and GPU_AVAILABLE:
        try:
            # 确保数据在GPU上
            image_gpu = to_gpu(image)
            # 执行GPU旋转
            rotated_gpu = gpu_rotate(image_gpu, angle, axes=axes, reshape=reshape, order=order, cval=cval)
            return rotated_gpu
        except Exception as e:
            print(f"    GPU旋转失败，回退到CPU: {e}")
            return rotate(image, angle, axes=axes, reshape=reshape, order=order, cval=cval)
    else:
        return rotate(image, angle, axes=axes, reshape=reshape, order=order, cval=cval)


def gpu_zoom_wrapper(image, zoom_factor, order=1):
    """GPU缩放包装器"""
    if USE_GPU and GPU_AVAILABLE:
        try:
            image_gpu = to_gpu(image)
            zoomed_gpu = gpu_zoom(image_gpu, zoom_factor, order=order)
            return zoomed_gpu
        except Exception as e:
            print(f"    GPU缩放失败，回退到CPU: {e}")
            return zoom(image, zoom_factor, order=order)
    else:
        return zoom(image, zoom_factor, order=order)


def gpu_sum_wrapper(image, axis):
    """GPU求和包装器"""
    if USE_GPU and GPU_AVAILABLE:
        try:
            image_gpu = to_gpu(image)
            return cp.sum(image_gpu, axis=axis)
        except Exception as e:
            print(f"    GPU求和失败，回退到CPU: {e}")
            return np.sum(image, axis=axis)
    else:
        return np.sum(image, axis=axis)


def gpu_exp_wrapper(x):
    """GPU指数运算包装器"""
    if USE_GPU and GPU_AVAILABLE:
        try:
            x_gpu = to_gpu(x)
            return cp.exp(x_gpu)
        except Exception as e:
            print(f"    GPU指数运算失败，回退到CPU: {e}")
            return np.exp(x)
    else:
        return np.exp(x)


def gpu_clip_wrapper(x, a_min, a_max):
    """GPU裁剪包装器"""
    if USE_GPU and GPU_AVAILABLE:
        try:
            x_gpu = to_gpu(x)
            return cp.clip(x_gpu, a_min, a_max)
        except Exception as e:
            print(f"    GPU裁剪失败，回退到CPU: {e}")
            return np.clip(x, a_min, a_max)
    else:
        return np.clip(x, a_min, a_max)


def clear_gpu_memory():
    """清理GPU内存"""
    if USE_GPU and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# ============================================================
# 辅助函数
# ============================================================

def ensure_dir(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"  创建目录: {path}")


def normalize_image(image):
    """将图像归一化到0-1范围"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-8:
        return image
    normalized = (image - min_val) / (max_val - min_val)
    return normalized


def save_overlay_image(proj_image, mask_image, output_path, title1="Projection", title2="Mask",
                       cmap1='gray', cmap2='hot', alpha1=0.7, alpha2=0.4):
    """保存投影和mask的重叠图"""
    img1_norm = normalize_image(proj_image.astype(np.float32))
    img2_norm = normalize_image(mask_image.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1_norm, cmap=cmap1, interpolation='nearest', origin='lower')
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=10)
    axes[0].set_ylabel('Y', fontsize=10)
    plt.colorbar(axes[0].images[0], ax=axes[0], shrink=0.8)

    axes[1].imshow(img2_norm, cmap=cmap2, interpolation='nearest', origin='lower')
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X', fontsize=10)
    axes[1].set_ylabel('Y', fontsize=10)
    plt.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)

    cmap1_func = plt.cm.get_cmap(cmap1)
    cmap2_func = plt.cm.get_cmap(cmap2)

    img1_rgb = cmap1_func(img1_norm)[:, :, :3]
    img2_rgb = cmap2_func(img2_norm)[:, :, :3]

    overlay = img1_rgb * alpha1 + img2_rgb * alpha2
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay, interpolation='nearest', origin='lower')
    axes[2].set_title(f'Overlay (Alpha: {alpha1}/{alpha2})', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X', fontsize=10)
    axes[2].set_ylabel('Y', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# DSA图像预处理函数
# ============================================================

def dsa_to_attenuation(dsa_volume, threshold=0.1):
    """将DSA图像转换为真实的线性衰减系数"""
    dsa_float32 = dsa_volume.astype(np.float32)
    threshold_f32 = float(threshold)

    base_attenuation = np.float32(0.01)
    extra_attenuation = np.float32(0.06)

    attenuation_volume = base_attenuation + dsa_float32 * extra_attenuation

    vessel_mask = dsa_float32 >= threshold_f32
    attenuation_volume[vessel_mask] = base_attenuation + extra_attenuation * 1.5

    print(
        f"    衰减系数统计: min={attenuation_volume.min():.4f}, max={attenuation_volume.max():.4f}, mean={attenuation_volume.mean():.4f}")

    return attenuation_volume.astype(COMPUTE_DTYPE)


def process_dsa_volume(dsa_volume, target_size=256):
    """处理DSA体积，转换为衰减系数"""
    current_size = dsa_volume.shape[0]

    if current_size != target_size:
        print(f"    尺寸调整: {current_size} -> {target_size}")
        zoom_factor = target_size / current_size
        processed_float32 = zoom(dsa_volume.astype(np.float32), zoom_factor, order=1)
        processed = processed_float32.astype(COMPUTE_DTYPE)
    else:
        print(f"    尺寸匹配: {current_size}，无需调整")
        processed = dsa_volume.astype(COMPUTE_DTYPE)

    if MAP_TO_REAL_ATTENUATION:
        print(f"    映射DSA值到真实衰减系数...")
        attenuation_volume = dsa_to_attenuation(processed, threshold=BINARY_THRESHOLD)
        return attenuation_volume
    else:
        print(f"    警告: 直接使用DSA值")
        return processed


# ============================================================
# 投影模拟函数 (GPU加速版本)
# ============================================================

def compute_projection_gpu(attenuation_volume, axis, voxel_spacing=0.5):
    """
    计算投影（GPU加速版本）
    返回: 投影图像 (透射率值，范围0-1)
    """
    if USE_GPU and GPU_AVAILABLE:
        try:
            atten_gpu = to_gpu(attenuation_volume)
            line_integral = cp.sum(atten_gpu, axis=axis) * voxel_spacing
            projection = cp.exp(-line_integral)
            projection = cp.clip(projection, 0, 1)
            return projection
        except Exception as e:
            print(f"    GPU投影计算失败，回退到CPU: {e}")
            # 回退到CPU
            atten_f32 = attenuation_volume.astype(np.float32)
            line_integral = np.sum(atten_f32, axis=axis) * voxel_spacing
            projection = np.exp(-line_integral)
            projection = np.clip(projection, 0, 1)
            return projection.astype(COMPUTE_DTYPE)
    else:
        atten_f32 = attenuation_volume.astype(np.float32)
        line_integral = np.sum(atten_f32, axis=axis) * voxel_spacing
        projection = np.exp(-line_integral)
        projection = np.clip(projection, 0, 1)
        return projection.astype(COMPUTE_DTYPE)


def normalize_projections_gpu(ap_proj, lat_proj):
    """
    使用统一的归一化范围对AP和LAT进行归一化 (GPU加速版本)
    """
    if USE_GPU and GPU_AVAILABLE:
        try:
            ap_gpu = to_gpu(ap_proj)
            lat_gpu = to_gpu(lat_proj)

            global_min = cp.minimum(cp.min(ap_gpu), cp.min(lat_gpu))
            global_max = cp.maximum(cp.max(ap_gpu), cp.max(lat_gpu))

            if global_max - global_min > 1e-8:
                ap_normalized = (ap_gpu - global_min) / (global_max - global_min)
                lat_normalized = (lat_gpu - global_min) / (global_max - global_min)
            else:
                ap_normalized = cp.ones_like(ap_gpu) * 0.5
                lat_normalized = cp.ones_like(lat_gpu) * 0.5

            return ap_normalized, lat_normalized
        except Exception as e:
            print(f"    GPU归一化失败，回退到CPU: {e}")
            # 回退到CPU
            global_min = min(np.min(ap_proj), np.min(lat_proj))
            global_max = max(np.max(ap_proj), np.max(lat_proj))
            if global_max - global_min > 1e-8:
                ap_normalized = (ap_proj - global_min) / (global_max - global_min)
                lat_normalized = (lat_proj - global_min) / (global_max - global_min)
            else:
                ap_normalized = np.ones_like(ap_proj) * 0.5
                lat_normalized = np.ones_like(lat_proj) * 0.5
            return ap_normalized.astype(COMPUTE_DTYPE), lat_normalized.astype(COMPUTE_DTYPE)
    else:
        global_min = min(np.min(ap_proj), np.min(lat_proj))
        global_max = max(np.max(ap_proj), np.max(lat_proj))
        if global_max - global_min > 1e-8:
            ap_normalized = (ap_proj - global_min) / (global_max - global_min)
            lat_normalized = (lat_proj - global_min) / (global_max - global_min)
        else:
            ap_normalized = np.ones_like(ap_proj) * 0.5
            lat_normalized = np.ones_like(lat_proj) * 0.5
        return ap_normalized.astype(COMPUTE_DTYPE), lat_normalized.astype(COMPUTE_DTYPE)


def resize_2d_image_gpu(image, target_size):
    """将2D图像缩放到目标尺寸 (GPU加速版本)"""
    current_size = image.shape[0]
    if current_size == target_size:
        return image

    zoom_factor = target_size / current_size
    return gpu_zoom_wrapper(image, zoom_factor, order=1)


def save_as_nifti(data, output_path):
    """保存为NIfTI格式，自动创建目录"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    # 确保数据在CPU上
    data_cpu = to_cpu(data)
    data_save = data_cpu.astype(SAVE_DTYPE)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii_img = nib.Nifti1Image(data_save, affine)
    nib.save(nii_img, output_path)


# ============================================================
# 核心处理函数 (GPU加速版本)
# ============================================================

def process_single_combination_gpu(case_num, attenuation_volume, mask_pruned,
                                   rot_angle_deg, proj_angle_deg,
                                   rot_idx, proj_idx, output_dir_parent,
                                   generate_overlay=False):
    """
    处理单个旋转角度和投影夹角的组合 (GPU加速版本)
    """
    # 文件夹命名
    folder_name = f"{case_num}{rot_idx:1d}_{proj_angle_deg}"
    output_dir = os.path.join(output_dir_parent, folder_name)
    ensure_dir(output_dir)

    ap_path = os.path.join(output_dir, "ap.nii.gz")
    lat_path = os.path.join(output_dir, "lat.nii.gz")
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    try:
        # ========== 1. 获取衰减体积的最小值 ==========
        fill_value = np.min(attenuation_volume)
        if USE_GPU and GPU_AVAILABLE:
            fill_value = float(fill_value)

        # ========== 2. 整体旋转体积 ==========
        rotated_atten = gpu_rotate_wrapper(
            attenuation_volume,
            rot_angle_deg,
            axes=(1, 2),
            reshape=False,
            order=1,
            cval=fill_value
        )

        # ========== 3. AP投影 ==========
        ap_projection = compute_projection_gpu(rotated_atten, axis=PROJECTION_AXIS, voxel_spacing=0.5)

        # ========== 4. LAT投影 ==========
        lat_rotated = gpu_rotate_wrapper(
            rotated_atten,
            proj_angle_deg,
            axes=LAT_ROTATION_AXES,
            reshape=False,
            order=1,
            cval=fill_value
        )
        lat_projection = compute_projection_gpu(lat_rotated, axis=PROJECTION_AXIS, voxel_spacing=0.5)

        # ========== 5. 使用统一的归一化范围 ==========
        ap_normalized, lat_normalized = normalize_projections_gpu(ap_projection, lat_projection)

        # ========== 6. 缩放投影 ==========
        ap_resized = resize_2d_image_gpu(ap_normalized, TARGET_2D_SIZE)
        lat_resized = resize_2d_image_gpu(lat_normalized, TARGET_2D_SIZE)

        # ========== 7. 像素值反转 ==========
        if INVERT_PROJECTION:
            if USE_GPU and GPU_AVAILABLE:
                ap_resized = 1.0 - ap_resized
                lat_resized = 1.0 - lat_resized
            else:
                ap_resized = 1.0 - ap_resized
                lat_resized = 1.0 - lat_resized

        # ========== 8. 转换回CPU并保存 ==========
        ap_final = to_cpu(ap_resized)
        lat_final = to_cpu(lat_resized)

        # ========== 9. Mask的处理 (在CPU上执行，保持简单) ==========
        rotated_mask = rotate(
            mask_pruned.astype(np.float32),
            rot_angle_deg,
            axes=(1, 2),
            reshape=False,
            order=0,
            cval=0
        )
        rotated_mask = (rotated_mask >= 0.5).astype(np.float32)
        mask_ap_projection = np.max(rotated_mask, axis=PROJECTION_AXIS)

        lat_rotated_mask = rotate(
            rotated_mask,
            proj_angle_deg,
            axes=LAT_ROTATION_AXES,
            reshape=False,
            order=0,
            cval=0
        )
        lat_rotated_mask = (lat_rotated_mask >= 0.5).astype(np.float32)
        mask_lat_projection = np.max(lat_rotated_mask, axis=PROJECTION_AXIS)

        mask_ap_resized = zoom(mask_ap_projection, TARGET_2D_SIZE / mask_ap_projection.shape[0], order=0)
        mask_lat_resized = zoom(mask_lat_projection, TARGET_2D_SIZE / mask_lat_projection.shape[0], order=0)

        # ========== 10. 保存文件 ==========
        save_as_nifti(ap_final, ap_path)
        save_as_nifti(lat_final, lat_path)
        save_as_nifti(mask_pruned, mask_path)

        # ========== 11. 生成叠加图（仅前两个组合） ==========
        if generate_overlay:
            overlay_dir = os.path.join(output_dir, "overlays")
            ensure_dir(overlay_dir)

            ap_overlay_path = os.path.join(overlay_dir, "AP_overlay.png")
            save_overlay_image(ap_final, mask_ap_resized, ap_overlay_path,
                               title1="AP Projection", title2="AP Mask (Loaded)",
                               cmap1='gray', cmap2='hot', alpha1=AP_ALPHA, alpha2=MASK_ALPHA)

            lat_overlay_path = os.path.join(overlay_dir, "LAT_overlay.png")
            save_overlay_image(lat_final, mask_lat_resized, lat_overlay_path,
                               title1="LAT Projection", title2="LAT Mask (Loaded)",
                               cmap1='gray', cmap2='hot', alpha1=LAT_ALPHA, alpha2=MASK_ALPHA)

        # 清理GPU内存
        if USE_GPU and GPU_AVAILABLE and (rot_idx + proj_idx) % 5 == 0:
            clear_gpu_memory()

        return True

    except Exception as e:
        print(f"    ✗ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_single_case_gpu(case_num, output_base, max_combinations_with_overlay=2):
    """处理单个病例 (GPU加速版本)"""

    # ========== 1. 加载DSA图像 ==========
    dsa_file = os.path.join(RAW_NII_BASE, f"{case_num}.nii.gz")
    if not os.path.exists(dsa_file):
        print(f"错误: DSA文件不存在 {dsa_file}")
        return 0, 0

    nii_img = nib.load(dsa_file)
    original_dsa = nii_img.get_fdata().astype(np.float32)

    print(f"\n病例 {case_num}:")
    print(f"  DSA形状: {original_dsa.shape}")
    print(f"  DSA像素范围: [{original_dsa.min():.4f}, {original_dsa.max():.4f}]")

    # ========== 2. 加载预生成mask ==========
    mask_file = os.path.join(MASK_TEST_BASE, f"{case_num}.nii.gz")
    if not os.path.exists(mask_file):
        print(f"错误: Mask文件不存在 {mask_file}")
        return 0, 0

    mask_img = nib.load(mask_file)
    mask_pruned = mask_img.get_fdata().astype(np.float32)
    mask_pruned = (mask_pruned >= 0.5).astype(np.uint8)

    print(f"  加载mask形状: {mask_pruned.shape}")
    print(f"  mask体素数: {np.sum(mask_pruned)}")
    print(f"  mask像素范围: [{mask_pruned.min():.0f}, {mask_pruned.max():.0f}]")

    # ========== 3. 处理DSA体积 ==========
    print(f"  预处理DSA体积（转换为衰减系数）...")
    attenuation_volume = process_dsa_volume(original_dsa, target_size=TARGET_3D_SIZE)

    # ========== 4. 调整尺寸 ==========
    if mask_pruned.shape[0] != TARGET_3D_SIZE:
        print(f"  调整mask尺寸: {mask_pruned.shape[0]} -> {TARGET_3D_SIZE}")
        zoom_factor = TARGET_3D_SIZE / mask_pruned.shape[0]
        mask_pruned_resized = zoom(mask_pruned.astype(np.float32), zoom_factor, order=0)
        mask_pruned = (mask_pruned_resized >= 0.5).astype(np.uint8)
        print(f"  调整后mask体素数: {np.sum(mask_pruned)}")

    if original_dsa.shape[0] != TARGET_3D_SIZE:
        zoom_factor = TARGET_3D_SIZE / original_dsa.shape[0]
        attenuation_volume = zoom(attenuation_volume.astype(np.float32), zoom_factor, order=1).astype(COMPUTE_DTYPE)

    # ========== 5. 处理所有组合 ==========
    success_count = 0
    total_combinations = len(ROTATION_ANGLES) * len(PROJECTION_ANGLES)
    combo_counter = 0

    start_time = time.time()

    for rot_idx, rot_angle in enumerate(ROTATION_ANGLES):
        for proj_idx, proj_angle in enumerate(PROJECTION_ANGLES):
            generate_overlay = (combo_counter < max_combinations_with_overlay)

            if process_single_combination_gpu(case_num, attenuation_volume, mask_pruned,
                                              rot_angle, proj_angle,
                                              rot_idx, proj_idx,
                                              output_base, generate_overlay):
                success_count += 1

            combo_counter += 1

    elapsed_time = time.time() - start_time
    print(f"  处理完成，耗时: {elapsed_time:.2f} 秒")
    print(f"  平均每个组合: {elapsed_time / total_combinations:.2f} 秒")

    return success_count, total_combinations


# ============================================================
# 测试模式
# ============================================================

def test_mode_gpu():
    """测试模式 (GPU加速版本)"""
    print("\n" + "=" * 80)
    print("测试模式 - DSA扇形束蒙泰卡罗模拟 + 自定义夹角双平面投影 (GPU加速)")
    print("=" * 80)
    print(f"测试病例: {TEST_CASE_NUM}")
    print(f"整体旋转索引: {TEST_ROTATION_IDX} (对应角度: {ROTATION_ANGLES[TEST_ROTATION_IDX]}°)")
    print(f"投影夹角索引: {TEST_PROJ_IDX} (对应角度: {PROJECTION_ANGLES[TEST_PROJ_IDX]}°)")
    print(f"投影反转: {'是 (血管变暗)' if INVERT_PROJECTION else '否'}")
    print(f"Mask来源: {MASK_TEST_BASE}")
    print(f"GPU加速: {'✅ 启用' if (USE_GPU and GPU_AVAILABLE) else '❌ 禁用'}")
    print("=" * 80)

    ensure_dir(TRAIN_BASE)

    success, total = process_single_case_gpu(TEST_CASE_NUM, TRAIN_BASE, max_combinations_with_overlay=2)

    if success > 0:
        print(f"\n✓ 成功处理 {success}/{total} 个组合")
    else:
        print(f"\n✗ 处理失败")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


# ============================================================
# 批量模式
# ============================================================

def batch_mode_gpu():
    """批量模式 (GPU加速版本)"""
    print("\n" + "=" * 80)
    print("批量处理模式 (DSA + 自定义夹角双平面投影) (GPU加速)")
    print("=" * 80)
    print(f"输出目录: {TRAIN_BASE}")
    print(f"投影反转: {'是 (血管变暗)' if INVERT_PROJECTION else '否'}")
    print(f"投影轴定义:")
    print(f"  - AP投影轴: axis={PROJECTION_AXIS} (沿Y轴)")
    print(f"  - LAT旋转轴: axis={LAT_ROTATION_AXES} (绕X轴旋转后沿Y轴投影)")
    print(f"  - AP和LAT夹角: 自定义 ({PROJECTION_ANGLES}°)")
    print(f"Mask来源: {MASK_TEST_BASE}")
    print(f"整体旋转角度: {ROTATION_ANGLES}")
    print(f"投影夹角: {PROJECTION_ANGLES}")
    print(
        f"总样本数: {len(ROTATION_ANGLES)} × {len(PROJECTION_ANGLES)} = {len(ROTATION_ANGLES) * len(PROJECTION_ANGLES)}")
    print(f"GPU加速: {'✅ 启用' if (USE_GPU and GPU_AVAILABLE) else '❌ 禁用'}")
    print("=" * 80)

    nii_files = glob.glob(os.path.join(RAW_NII_BASE, "*.nii.gz"))
    cases = []
    for f in nii_files:
        match = re.search(r"(\d+)\.nii\.gz$", os.path.basename(f))
        if match:
            case_num = int(match.group(1))
            mask_file = os.path.join(MASK_TEST_BASE, f"{case_num}.nii.gz")
            if os.path.exists(mask_file):
                cases.append(case_num)
            else:
                print(f"警告: 病例 {case_num} 的mask不存在，跳过")

    cases = sorted(cases)

    if len(cases) == 0:
        print("错误：没有找到可用的病例（同时需要DSA和mask）")
        return

    print(f"\n找到 {len(cases)} 个有效病例")
    ensure_dir(TRAIN_BASE)

    total_success = 0
    total_combinations = 0
    total_time = 0

    for case_num in tqdm(cases, desc="处理病例"):
        start_time = time.time()
        success, total = process_single_case_gpu(case_num, TRAIN_BASE, max_combinations_with_overlay=2)
        elapsed = time.time() - start_time
        total_success += success
        total_combinations += total
        total_time += elapsed

        # 每个病例处理后清理GPU内存
        if USE_GPU and GPU_AVAILABLE:
            clear_gpu_memory()

    print("\n" + "=" * 80)
    print("批量处理完成！")
    print("=" * 80)
    print(f"处理病例数: {len(cases)}")
    print(f"成功处理组合: {total_success}/{total_combinations}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个病例: {total_time / len(cases):.2f} 秒")
    print(f"平均每个组合: {total_time / total_combinations:.2f} 秒")
    if USE_GPU and GPU_AVAILABLE:
        print(f"GPU加速已启用，性能提升显著")
    print("=" * 80)


# ============================================================
# 主函数
# ============================================================

def main():
    if RUN_MODE == "test":
        test_mode_gpu()
    elif RUN_MODE == "batch":
        batch_mode_gpu()
    else:
        print(f"错误: 未知的运行模式 '{RUN_MODE}'")


if __name__ == "__main__":
    main()