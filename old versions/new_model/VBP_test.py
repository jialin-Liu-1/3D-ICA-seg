import torch
import numpy as np
import nibabel as nib
import os
from torch.nn import functional as F
from scipy.ndimage import rotate, zoom
import time
from datetime import datetime


def backproject_ap(f_2d, volume_shape, reverse_y=False):
    """
    AP反投影：沿Y轴复制（AP投影的逆过程）
    f_2d: (B, C, X, Z)
    volume_shape: (X, Y, Z)
    reverse_y: 是否沿Y轴反向复制（从另一端开始）
    返回: (B, C, X, Y, Z)
    """
    B, C, X, Z = f_2d.shape
    X_dim, Y_dim, Z_dim = volume_shape

    # 确保尺寸匹配
    assert X == X_dim and Z == Z_dim, f"输入尺寸 {X}x{Z} 与体积尺寸 {X_dim}x{Z_dim} 不匹配"

    # 沿Y轴复制 (B, C, X, Z) -> (B, C, X, Y, Z)
    f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)

    # 如果需要反向，沿Y轴翻转
    if reverse_y:
        f_3d = torch.flip(f_3d, dims=[3])  # 沿Y轴翻转

    return f_3d


def backproject_lat(f_2d, volume_shape, angle_deg=90, reverse_y=False):
    """
    LAT反投影：LAT投影的逆过程
    LAT投影：先绕X轴旋转angle_deg，然后沿Y轴投影
    LAT反投影：先沿Y轴复制，然后反向旋转angle_deg

    f_2d: (B, C, X, Z)
    volume_shape: (X, Y, Z)
    angle_deg: 角度（度）- 对应正向投影中的proj_angle_deg
    reverse_y: 是否沿Y轴反向复制（从另一端开始）
    返回: (B, C, X, Y, Z)
    """
    B, C, X, Z = f_2d.shape
    X_dim, Y_dim, Z_dim = volume_shape

    # 确保尺寸匹配
    assert X == X_dim and Z == Z_dim, f"输入尺寸 {X}x{Z} 与体积尺寸 {X_dim}x{Z_dim} 不匹配"

    # 如果角度为90度，LAT与AP相同
    if angle_deg == 90:
        f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)
        if reverse_y:
            f_3d = torch.flip(f_3d, dims=[3])
        return f_3d

    # 第一步：沿Y轴复制（与AP反投影相同）
    f_3d = f_2d[:, :, :, None, :].expand(-1, -1, -1, Y_dim, -1)

    # 如果需要反向，沿Y轴翻转
    if reverse_y:
        f_3d = torch.flip(f_3d, dims=[3])

    # 第二步：反向旋转（LAT投影的逆过程）
    # 正向投影是旋转angle_deg，反向投影应该旋转 -angle_deg
    rotation_angle = -angle_deg

    # 对每个通道和batch进行旋转
    # 使用scipy.ndimage.rotate
    f_3d_np = f_3d.cpu().numpy()  # (B, C, X, Y, Z)

    # 对每个batch和channel分别旋转
    f_3d_rotated = np.zeros_like(f_3d_np)
    for b in range(B):
        for c in range(C):
            # 旋转3D体积，axes=(1,2) 对应 (Y, Z)
            f_3d_rotated[b, c] = rotate(
                f_3d_np[b, c],
                rotation_angle,
                axes=(1, 2),  # 旋转Y和Z平面，相当于绕X轴
                reshape=False,
                order=1,
                cval=0
            )

    # 转换回tensor
    f_3d_result = torch.from_numpy(f_3d_rotated).to(f_3d.device)

    return f_3d_result


def load_nifti_as_tensor(file_path):
    """加载NIFTI文件并转换为PyTorch张量"""
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    # 添加batch和channel维度
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def save_tensor_as_nifti(tensor, file_path, ref_nii_path=None):
    """将PyTorch张量保存为NIFTI文件"""
    if tensor.dim() == 5:
        data = tensor.squeeze(0).squeeze(0).cpu().numpy()  # (X, Y, Z)
    else:
        data = tensor.cpu().numpy()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if ref_nii_path and os.path.exists(ref_nii_path):
        ref_nii = nib.load(ref_nii_path)
        affine = ref_nii.affine
    else:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)
    print(f"  已保存: {file_path}")


def process_images(ap_path, lat_path, mask_path, output_dir, volume_shape=(256, 256, 256),
                   angle=90, reverse_y=False, verbose=True):
    """
    处理单组图像：基于正向投影逻辑的反投影

    Args:
        ap_path: AP图像路径
        lat_path: LAT图像路径
        mask_path: Mask图像路径（用于参考）
        output_dir: 输出目录
        volume_shape: 三维体积形状 (X, Y, Z)
        angle: LAT投影角度（度）- 对应正向投影的proj_angle_deg
        reverse_y: 是否沿Y轴反向复制（从另一端开始投影）
        verbose: 是否打印详细信息
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 时间统计 ==========
    timings = {
        'load_ap': 0.0,
        'load_lat': 0.0,
        'load_mask': 0.0,
        'resize': 0.0,
        'ap_backproject': 0.0,
        'lat_backproject': 0.0,
        'mask_crop': 0.0,
        'fusion': 0.0,
        'save': 0.0,
        'total': 0.0
    }

    start_total = time.time()

    direction_str = "反向Y轴" if reverse_y else "正向Y轴"
    if verbose:
        print(f"  ========================================")
        print(f"  反投影处理: AP基准 + LAT旋转")
        print(f"  角度: {angle}° (Y轴方向: {direction_str})")
        print(f"  ========================================")

    # ========== 加载图像 ==========
    if verbose:
        print(f"  加载AP: {ap_path}")
    start = time.time()
    ap_tensor = load_nifti_as_tensor(ap_path).to(device)
    timings['load_ap'] = time.time() - start

    if verbose:
        print(f"  加载LAT: {lat_path}")
    start = time.time()
    lat_tensor = load_nifti_as_tensor(lat_path).to(device)
    timings['load_lat'] = time.time() - start

    if verbose:
        print(f"  加载Mask: {mask_path}")
    start = time.time()
    mask_tensor = load_nifti_as_tensor(mask_path).to(device)
    timings['load_mask'] = time.time() - start

    if verbose:
        print(f"  AP形状: {ap_tensor.shape}")
        print(f"  LAT形状: {lat_tensor.shape}")
        print(f"  Mask形状: {mask_tensor.shape}")

    # ========== 调整尺寸 ==========
    start = time.time()
    # 获取输入图像的尺寸 (B, C, H, W)
    _, _, H, W = ap_tensor.shape
    X_dim, Y_dim, Z_dim = volume_shape

    # 确保输入图像尺寸与体积尺寸匹配 (H=X, W=Z)
    if H != X_dim or W != Z_dim:
        if verbose:
            print(f"  调整图像尺寸: {H}x{W} -> {X_dim}x{Z_dim}")
        ap_tensor = F.interpolate(ap_tensor, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)
        lat_tensor = F.interpolate(lat_tensor, size=(X_dim, Z_dim), mode='bilinear', align_corners=False)

    # 确保mask尺寸与输出体积匹配
    if mask_tensor.shape[-3:] != (X_dim, Y_dim, Z_dim):
        if verbose:
            print(f"  调整Mask尺寸: {mask_tensor.shape[-3:]} -> {(X_dim, Y_dim, Z_dim)}")
        mask_tensor = F.interpolate(mask_tensor, size=(X_dim, Y_dim, Z_dim),
                                    mode='trilinear', align_corners=False)
    timings['resize'] = time.time() - start

    # ========== 1. AP反投影 ==========
    if verbose:
        print("\n  [AP反投影] 沿Y轴复制...")
    start = time.time()
    ap_volume = backproject_ap(ap_tensor, volume_shape, reverse_y)
    timings['ap_backproject'] = time.time() - start
    if verbose:
        print(f"  AP体积形状: {ap_volume.shape}")
        print(f"  AP体积非零体素数: {torch.sum(ap_volume > 0).item()}")

    # ========== 2. LAT反投影 ==========
    if verbose:
        print(f"\n  [LAT反投影] 角度={angle}°, Y轴方向={direction_str}...")
    start = time.time()
    lat_volume = backproject_lat(lat_tensor, volume_shape, angle, reverse_y)
    timings['lat_backproject'] = time.time() - start
    if verbose:
        print(f"  LAT体积形状: {lat_volume.shape}")
        print(f"  LAT体积非零体素数: {torch.sum(lat_volume > 0).item()}")

    # ========== 3. 裁剪 ==========
    if verbose:
        print("\n  [裁剪] 用AP掩码裁剪LAT体积...")
    start = time.time()
    ap_mask = (ap_volume > 0).float()
    lat_masked = lat_volume * ap_mask
    timings['mask_crop'] = time.time() - start
    if verbose:
        print(f"  LAT裁剪后非零体素数: {torch.sum(lat_masked > 0).item()}")

    # ========== 4. 融合 ==========
    if verbose:
        print("\n  [融合] AP体积 + LAT贡献...")
    start = time.time()
    combined_volume = ap_volume + lat_masked
    timings['fusion'] = time.time() - start
    if verbose:
        print(f"  融合后形状: {combined_volume.shape}")

    # ========== 5. 与Mask相加 ==========
    if verbose:
        print("\n  [Mask融合] 与Mask相加...")
    combined_with_mask = combined_volume + mask_tensor

    # ========== 创建输出目录并保存 ==========
    os.makedirs(output_dir, exist_ok=True)
    dir_suffix = "reverse_y" if reverse_y else "forward_y"

    start = time.time()
    ap_output_path = os.path.join(output_dir, f"ap_volume.nii.gz")
    save_tensor_as_nifti(ap_volume, ap_output_path, mask_path)

    lat_full_path = os.path.join(output_dir, f"lat_volume_full_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(lat_volume, lat_full_path, mask_path)

    lat_masked_path = os.path.join(output_dir, f"lat_masked_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(lat_masked, lat_masked_path, mask_path)

    combined_path = os.path.join(output_dir, f"combined_volume_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(combined_volume, combined_path, mask_path)

    combined_mask_path = os.path.join(output_dir, f"combined_with_mask_{angle}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(combined_with_mask, combined_mask_path, mask_path)
    timings['save'] = time.time() - start

    timings['total'] = time.time() - start_total

    # ========== 打印时间统计 ==========
    print(f"\n  ⏱️  时间统计:")
    print(f"    - 加载AP: {timings['load_ap']:.4f}s")
    print(f"    - 加载LAT: {timings['load_lat']:.4f}s")
    print(f"    - 加载Mask: {timings['load_mask']:.4f}s")
    print(f"    - 调整尺寸: {timings['resize']:.4f}s")
    print(f"    - AP反投影: {timings['ap_backproject']:.4f}s")
    print(f"    - LAT反投影: {timings['lat_backproject']:.4f}s")
    print(f"    - 掩码裁剪: {timings['mask_crop']:.4f}s")
    print(f"    - 体积融合: {timings['fusion']:.4f}s")
    print(f"    - 保存文件: {timings['save']:.4f}s")
    print(f"    - 总耗时: {timings['total']:.4f}s")

    if verbose:
        print(f"\n  ✓ 成功保存到: {output_dir}")
        print(f"  - AP体积: ap_volume.nii.gz (基准)")
        print(f"  - LAT完整体积: lat_volume_full_{angle}deg_{dir_suffix}.nii.gz")
        print(f"  - LAT裁剪后: lat_masked_{angle}deg_{dir_suffix}.nii.gz")
        print(f"  - AP+LAT: combined_volume_{angle}deg_{dir_suffix}.nii.gz")
        print(f"  - AP+LAT+Mask: combined_with_mask_{angle}deg_{dir_suffix}.nii.gz")

    return ap_volume, lat_volume, lat_masked, combined_volume, combined_with_mask, timings


def find_available_cases(base_path):
    """
    自动查找所有可用的病例文件夹
    返回: list of (case_num, angle, folder_name) tuples
    """
    available_cases = []

    if not os.path.exists(base_path):
        print(f"  错误: 路径不存在 {base_path}")
        return available_cases

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split('_')
        if len(parts) != 2:
            continue

        try:
            case_num = int(parts[0])
            angle = int(parts[1])
        except ValueError:
            continue

        ap_path = os.path.join(folder_path, "ap.nii.gz")
        lat_path = os.path.join(folder_path, "lat.nii.gz")
        mask_path = os.path.join(folder_path, "mask.nii.gz")

        if all([os.path.exists(p) for p in [ap_path, lat_path, mask_path]]):
            available_cases.append((case_num, angle, folder_name))
            print(f"  发现病例: {folder_name} (病例{case_num}, 角度{angle}°)")
        else:
            missing = []
            if not os.path.exists(ap_path): missing.append("ap.nii.gz")
            if not os.path.exists(lat_path): missing.append("lat.nii.gz")
            if not os.path.exists(mask_path): missing.append("mask.nii.gz")
            print(f"  跳过文件夹 {folder_name}: 缺少 {', '.join(missing)}")

    return available_cases


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("反投影处理 - 基于正向投影逻辑")
    print("AP基准 + LAT旋转（Y轴方向可调）")
    print("=" * 60)

    # 配置路径
    base_path = r"/mnt/d/med_data/biron/data1/test_vbp"
    output_base = r"/mnt/d/med_data/biron/data1/VBP_no_trainable"
    volume_shape = (256, 256, 256)

    # ========== 关键参数：控制Y轴投影方向 ==========
    # True: 从Y轴另一端开始投影（反向）
    # False: 从Y轴正向开始投影
    REVERSE_Y = True  # 改为True试试

    # ========== 推理时间统计开关 ==========
    SHOW_TIMING = True  # 是否显示详细时间统计

    print(f"\n数据路径: {base_path}")
    print(f"输出路径: {output_base}")
    print(f"设备: {device}")
    print(f"体积形状: X={volume_shape[0]}, Y={volume_shape[1]}, Z={volume_shape[2]}")
    print(f"Y轴投影方向: {'反向Y轴' if REVERSE_Y else '正向Y轴'}")
    print(f"显示时间统计: {'是' if SHOW_TIMING else '否'}")
    print(f"反投影逻辑:")
    print(f"  - AP: 沿Y轴复制 ({'反向' if REVERSE_Y else '正向'})")
    print(f"  - LAT: 沿Y轴复制 ({'反向' if REVERSE_Y else '正向'}) + 反向旋转 (绕X轴)")

    print("\n正在扫描可用病例...")
    print("-" * 50)
    available_cases = find_available_cases(base_path)

    if not available_cases:
        print("\n错误: 未找到任何有效的病例文件夹!")
        exit(1)

    print("-" * 50)
    print(f"\n找到 {len(available_cases)} 个可用病例:")
    for case_num, angle, folder_name in available_cases:
        print(f"  - {folder_name} (病例 {case_num}, 角度 {angle}°)")

    print("\n开始处理...")
    print("=" * 60)

    processed_count = 0
    failed_cases = []

    # ========== 全局时间统计 ==========
    total_timings = {
        'total_time': 0.0,
        'num_cases': 0,
        'per_case_times': []
    }
    overall_start = time.time()

    for case_num, angle, folder_name in available_cases:
        print(f"\n{'=' * 60}")
        print(f"处理: {folder_name}")
        print(f"病例编号: {case_num}, 角度: {angle}°")
        print(f"{'=' * 60}")

        folder_path = os.path.join(base_path, folder_name)
        ap_path = os.path.join(folder_path, "ap.nii.gz")
        lat_path = os.path.join(folder_path, "lat.nii.gz")
        mask_path = os.path.join(folder_path, "mask.nii.gz")

        try:
            output_dir = os.path.join(output_base, folder_name)
            result = process_images(
                ap_path, lat_path, mask_path, output_dir,
                volume_shape, angle, REVERSE_Y,
                verbose=SHOW_TIMING
            )
            # result 返回: ap_volume, lat_volume, lat_masked, combined_volume, combined_with_mask, timings
            if SHOW_TIMING:
                total_timings['per_case_times'].append({
                    'case': folder_name,
                    'angle': angle,
                    'timings': result[5]
                })
            processed_count += 1

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback

            traceback.print_exc()
            failed_cases.append(folder_name)

    total_timings['total_time'] = time.time() - overall_start
    total_timings['num_cases'] = processed_count

    # ========== 打印总结时间统计 ==========
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"  成功: {processed_count}/{len(available_cases)} 个病例")
    if failed_cases:
        print(f"  失败: {len(failed_cases)} 个病例")
        print(f"  失败的病例: {', '.join(failed_cases)}")

    # ========== 打印详细时间统计 ==========
    if SHOW_TIMING and processed_count > 0:
        print("\n" + "-" * 60)
        print("⏱️  总体时间统计:")
        print("-" * 60)

        # 计算平均时间
        avg_times = {}
        for case_data in total_timings['per_case_times']:
            for key, value in case_data['timings'].items():
                if key not in avg_times:
                    avg_times[key] = []
                avg_times[key].append(value)

        print(f"\n  总处理病例数: {total_timings['num_cases']}")
        print(f"  总耗时: {total_timings['total_time']:.4f}s")
        print(f"  平均每病例耗时: {total_timings['total_time'] / total_timings['num_cases']:.4f}s")

        print("\n  各阶段平均耗时:")
        for key, values in avg_times.items():
            avg = sum(values) / len(values)
            print(f"    - {key}: {avg:.4f}s")

        # 找出最快和最慢的病例
        if len(total_timings['per_case_times']) > 1:
            sorted_cases = sorted(
                total_timings['per_case_times'],
                key=lambda x: x['timings']['total']
            )
            print(f"\n  最快病例: {sorted_cases[0]['case']} ({sorted_cases[0]['timings']['total']:.4f}s)")
            print(f"  最慢病例: {sorted_cases[-1]['case']} ({sorted_cases[-1]['timings']['total']:.4f}s)")

    print("=" * 60)