import torch
import numpy as np
import nibabel as nib
import os
from torch.nn import functional as F
from scipy.ndimage import rotate
import time
import re


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


def normalize_angle(angle):
    """
    将角度归一化到 -180° 到 180° 之间
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def parse_ap_lat_files(base_path):
    """
    解析AP和LAT文件，按病例分组
    期望的文件结构:
    base_path/
      ├── ANY_病例号_投影次数/
      │   ├── ap_角度.nii.gz
      │   └── lat_角度.nii.gz
      └── ...

    返回: list of dict
    """
    valid_cases = []

    # 遍历所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 解析文件夹名称: ANY_病例号_投影次数 或 ANY_病例号_检查序号_投影次数
        parts = folder_name.split('_')
        if len(parts) < 2 or parts[0] != 'ANY':
            continue

        # 查找ap和lat文件
        ap_files = []
        lat_files = []

        for file in os.listdir(folder_path):
            if file.endswith('.nii.gz'):
                if file.startswith('ap_'):
                    ap_files.append(file)
                elif file.startswith('lat_'):
                    lat_files.append(file)

        # 检查是否有ap和lat文件
        if not ap_files or not lat_files:
            print(f"  跳过 {folder_name}: 缺少AP或LAT文件 (AP: {len(ap_files)}, LAT: {len(lat_files)})")
            continue

        # 提取角度信息
        for ap_file in ap_files:
            # 从 ap_角度.nii.gz 提取角度
            ap_match = re.match(r'ap_(\d+)\.nii\.gz', ap_file)
            if not ap_match:
                continue
            ap_angle = int(ap_match.group(1))

            for lat_file in lat_files:
                lat_match = re.match(r'lat_(\d+)\.nii\.gz', lat_file)
                if not lat_match:
                    continue
                lat_angle = int(lat_match.group(1))

                # 解析病例信息
                # 格式: ANY_病例号_投影次数 或 ANY_病例号_检查序号_投影次数
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
                    print(f"  跳过 {folder_name}: 无法解析文件夹名称")
                    continue

                # 计算LAT相对于AP的夹角
                angle_diff = lat_angle - ap_angle
                angle_diff_norm = normalize_angle(angle_diff)

                valid_cases.append({
                    'case_id': case_id,
                    'series_num': series_num,
                    'projection': projection,
                    'folder_name': folder_name,
                    'folder_path': folder_path,
                    'ap_angle': ap_angle,
                    'lat_angle': lat_angle,
                    'angle_diff': angle_diff_norm,  # LAT相对于AP的夹角
                    'ap_path': os.path.join(folder_path, ap_file),
                    'lat_path': os.path.join(folder_path, lat_file),
                    'ap_file': ap_file,
                    'lat_file': lat_file,
                    'case_key': case_key
                })

                print(f"  发现: {folder_name} - AP角度: {ap_angle}°, LAT角度: {lat_angle}°, 夹角: {angle_diff_norm}°")

    return valid_cases


def backproject_ap(f_2d, volume_shape, reverse_y=False):
    """
    AP反投影：沿Y轴复制（AP投影的逆过程）
    AP作为基准（0°），直接沿Y轴复制

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


def backproject_lat(f_2d, volume_shape, angle_diff, reverse_y=False):
    """
    LAT反投影：LAT投影的逆过程
    LAT投影：先绕X轴旋转angle_deg，然后沿Y轴投影
    LAT反投影：先沿Y轴复制，然后反向旋转angle_diff（LAT相对于AP的夹角）

    f_2d: (B, C, X, Z)
    volume_shape: (X, Y, Z)
    angle_diff: LAT相对于AP的夹角（度）
    reverse_y: 是否沿Y轴反向复制（从另一端开始）
    返回: (B, C, X, Y, Z)
    """
    B, C, X, Z = f_2d.shape
    X_dim, Y_dim, Z_dim = volume_shape

    # 确保尺寸匹配
    assert X == X_dim and Z == Z_dim, f"输入尺寸 {X}x{Z} 与体积尺寸 {X_dim}x{Z_dim} 不匹配"

    # 如果夹角为0°，LAT与AP方向相同，直接沿Y轴复制
    if abs(angle_diff) < 1e-6:
        print(f"    夹角为0°，LAT与AP方向相同，直接沿Y轴复制")
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
    # 正向投影是旋转angle_diff，反向投影应该旋转 -angle_diff
    rotation_angle = -angle_diff

    print(f"    LAT反投影: 沿Y轴复制后绕X轴旋转 {rotation_angle}° (夹角: {angle_diff}°)")

    # 对每个通道和batch进行旋转
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


def process_images(ap_path, lat_path, output_dir, case_info,
                   volume_shape=(256, 256, 256), reverse_y=False, verbose=True):
    """
    处理单组图像：基于正向投影逻辑的反投影

    AP反投影：沿Y轴复制（作为基准）
    LAT反投影：沿Y轴复制后绕X轴旋转 -angle_diff（angle_diff = LAT角度 - AP角度）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 时间统计 ==========
    timings = {
        'load_ap': 0.0,
        'load_lat': 0.0,
        'resize': 0.0,
        'ap_backproject': 0.0,
        'lat_backproject': 0.0,
        'fusion': 0.0,
        'save': 0.0,
        'total': 0.0
    }

    start_total = time.time()

    direction_str = "反向Y轴" if reverse_y else "正向Y轴"
    if verbose:
        print(f"  ========================================")
        print(f"  反投影处理: AP基准 + LAT旋转")
        print(f"  病例: {case_info['case_id']}, 投影: {case_info['projection']}")
        print(f"  AP角度: {case_info['ap_angle']}°, LAT角度: {case_info['lat_angle']}°")
        print(f"  夹角 (LAT - AP): {case_info['angle_diff']}°")
        print(f"  Y轴方向: {direction_str}")
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
        print(f"  AP形状: {ap_tensor.shape}")
        print(f"  LAT形状: {lat_tensor.shape}")
        print(f"  AP像素范围: [{ap_tensor.min():.4f}, {ap_tensor.max():.4f}]")
        print(f"  LAT像素范围: [{lat_tensor.min():.4f}, {lat_tensor.max():.4f}]")

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
    timings['resize'] = time.time() - start

    # ========== 1. AP反投影（基准） ==========
    if verbose:
        print("\n  [AP反投影] 作为基准，沿Y轴复制...")
    start = time.time()
    ap_volume = backproject_ap(ap_tensor, volume_shape, reverse_y)
    timings['ap_backproject'] = time.time() - start
    if verbose:
        print(f"  AP体积形状: {ap_volume.shape}")
        print(f"  AP体积非零体素数: {torch.sum(ap_volume > 0).item()}")

    # ========== 2. LAT反投影（相对于AP旋转） ==========
    if verbose:
        print(f"\n  [LAT反投影] 沿Y轴复制后绕X轴旋转 -{case_info['angle_diff']}°...")
    start = time.time()
    lat_volume = backproject_lat(lat_tensor, volume_shape, case_info['angle_diff'], reverse_y)
    timings['lat_backproject'] = time.time() - start
    if verbose:
        print(f"  LAT体积形状: {lat_volume.shape}")
        print(f"  LAT体积非零体素数: {torch.sum(lat_volume > 0).item()}")

    # ========== 3. 融合 ==========
    if verbose:
        print("\n  [融合] AP体积 + LAT贡献...")
    start = time.time()
    combined_volume = ap_volume + lat_volume
    timings['fusion'] = time.time() - start
    if verbose:
        print(f"  融合后形状: {combined_volume.shape}")
        print(f"  融合后非零体素数: {torch.sum(combined_volume > 0).item()}")

    # ========== 创建输出目录并保存 ==========
    os.makedirs(output_dir, exist_ok=True)
    dir_suffix = "reverse_y" if reverse_y else "forward_y"

    start = time.time()
    ap_output_path = os.path.join(output_dir, f"ap_volume.nii.gz")
    save_tensor_as_nifti(ap_volume, ap_output_path, ap_path)

    lat_output_path = os.path.join(output_dir, f"lat_volume_rot_{case_info['angle_diff']}deg_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(lat_volume, lat_output_path, lat_path)

    combined_path = os.path.join(output_dir,
                                 f"combined_volume_ap{case_info['ap_angle']}_lat{case_info['lat_angle']}_diff{case_info['angle_diff']}_{dir_suffix}.nii.gz")
    save_tensor_as_nifti(combined_volume, combined_path, ap_path)
    timings['save'] = time.time() - start

    timings['total'] = time.time() - start_total

    # ========== 打印时间统计 ==========
    print(f"\n  ⏱️  时间统计:")
    print(f"    - 加载AP: {timings['load_ap']:.4f}s")
    print(f"    - 加载LAT: {timings['load_lat']:.4f}s")
    print(f"    - 调整尺寸: {timings['resize']:.4f}s")
    print(f"    - AP反投影: {timings['ap_backproject']:.4f}s")
    print(f"    - LAT反投影: {timings['lat_backproject']:.4f}s")
    print(f"    - 体积融合: {timings['fusion']:.4f}s")
    print(f"    - 保存文件: {timings['save']:.4f}s")
    print(f"    - 总耗时: {timings['total']:.4f}s")

    if verbose:
        print(f"\n  ✓ 成功保存到: {output_dir}")
        print(f"  - AP体积: ap_volume.nii.gz (基准，沿Y轴复制)")
        print(
            f"  - LAT体积: lat_volume_rot_{case_info['angle_diff']}deg_{dir_suffix}.nii.gz (旋转{case_info['angle_diff']}°)")
        print(
            f"  - AP+LAT: combined_volume_ap{case_info['ap_angle']}_lat{case_info['lat_angle']}_diff{case_info['angle_diff']}_{dir_suffix}.nii.gz")

    return ap_volume, lat_volume, combined_volume, timings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("反投影处理 - 以AP为基准")
    print("AP基准 + LAT旋转（Y轴方向可调）")
    print("=" * 60)

    # 配置路径
    input_base = r"/mnt/d/med_data/biron/data2/output_nifti"
    output_base = r"/mnt/d/med_data/biron/data2/VBP_test"
    volume_shape = (256, 256, 256)

    # ========== 关键参数：控制Y轴投影方向 ==========
    REVERSE_Y = True  # True: 从Y轴另一端开始投影（反向）, False: 从Y轴正向开始投影

    # ========== 推理时间统计开关 ==========
    SHOW_TIMING = True

    print(f"\n输入路径: {input_base}")
    print(f"输出路径: {output_base}")
    print(f"设备: {device}")
    print(f"体积形状: X={volume_shape[0]}, Y={volume_shape[1]}, Z={volume_shape[2]}")
    print(f"Y轴投影方向: {'反向Y轴' if REVERSE_Y else '正向Y轴'}")
    print(f"显示时间统计: {'是' if SHOW_TIMING else '否'}")
    print(f"\n反投影逻辑:")
    print(f"  - AP: 沿Y轴复制 (基准，角度0°)")
    print(f"  - LAT: 沿Y轴复制 + 绕X轴旋转 -(LAT角度 - AP角度)")

    print("\n正在扫描输入文件...")
    print("-" * 50)

    # 解析AP和LAT文件
    valid_cases = parse_ap_lat_files(input_base)

    if not valid_cases:
        print("\n错误: 未找到任何有效的AP/LAT配对!")
        print("期望的文件结构:")
        print("  base_path/")
        print("    ├── ANY_病例号_投影次数/")
        print("    │   ├── ap_角度.nii.gz")
        print("    │   └── lat_角度.nii.gz")
        print("    └── ...")
        exit(1)

    print("-" * 50)
    print(f"\n找到 {len(valid_cases)} 个有效的AP/LAT配对:")
    for case in valid_cases:
        series_info = f" (检查序号 {case['series_num']})" if case['series_num'] else ""
        print(f"  - 病例 {case['case_id']}{series_info}, 投影 {case['projection']}")
        print(f"      AP: {case['ap_file']} (角度: {case['ap_angle']}°)")
        print(f"      LAT: {case['lat_file']} (角度: {case['lat_angle']}°)")
        print(f"      夹角 (LAT - AP): {case['angle_diff']}°")

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

    for case_info in valid_cases:
        print(f"\n{'=' * 60}")
        series_info = f" (检查序号 {case_info['series_num']})" if case_info['series_num'] else ""
        print(f"处理: {case_info['case_key']}")
        print(f"病例编号: {case_info['case_id']}{series_info}, 投影: {case_info['projection']}")
        print(f"AP角度: {case_info['ap_angle']}°, LAT角度: {case_info['lat_angle']}°")
        print(f"夹角 (LAT - AP): {case_info['angle_diff']}°")
        print(f"{'=' * 60}")

        try:
            # 创建输出目录
            if case_info['series_num']:
                output_dir = os.path.join(output_base,
                                          f"ANY_{case_info['case_id']}_{case_info['series_num']}_{case_info['projection']}")
            else:
                output_dir = os.path.join(output_base, f"ANY_{case_info['case_id']}_{case_info['projection']}")

            result = process_images(
                case_info['ap_path'],
                case_info['lat_path'],
                output_dir,
                case_info,
                volume_shape,
                REVERSE_Y,
                verbose=SHOW_TIMING
            )
            # result 返回: ap_volume, lat_volume, combined_volume, timings
            if SHOW_TIMING:
                total_timings['per_case_times'].append({
                    'case': case_info['case_key'],
                    'ap_angle': case_info['ap_angle'],
                    'lat_angle': case_info['lat_angle'],
                    'angle_diff': case_info['angle_diff'],
                    'timings': result[3]
                })
            processed_count += 1

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_cases.append(case_info['case_key'])

    total_timings['total_time'] = time.time() - overall_start
    total_timings['num_cases'] = processed_count

    # ========== 打印总结 ==========
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"  成功: {processed_count}/{len(valid_cases)} 个病例")
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


if __name__ == "__main__":
    main()
