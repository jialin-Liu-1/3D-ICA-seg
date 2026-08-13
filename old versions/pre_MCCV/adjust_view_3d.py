import numpy as np
import nibabel as nib
import pydicom
import os
from scipy.ndimage import rotate as ndi_rotate

# ============================================================
# 旋转参数配置（直接在这里修改）
# ============================================================

# 3D体积旋转参数（绕X、Y、Z轴，值为0、1、2、3，对应0°、90°、180°、270°）
ROT_X = 3  # 绕X轴旋转次数: 0=0°, 1=90°, 2=180°, 3=270°
ROT_Y = 0  # 绕Y轴旋转次数: 0=0°, 1=90°, 2=180°, 3=270°
ROT_Z = 3  # 绕Z轴旋转次数: 0=0°, 1=90°, 2=180°, 3=270°

# 2D图像变换参数
ROT_2D = 0  # 旋转次数: 0=0°, 1=90°, 2=180°, 3=270°
FLIP_HORIZONTAL = False  # 水平翻转（左右）
FLIP_VERTICAL = False  # 垂直翻转（上下）

# 输入输出路径
MASK_INPUT = r"D:\med_data\biron\data1\mask\0_mask.nii.gz"
PROJECTION_INPUT = r"D:\med_data\biron\data1\raw_dcm\0\contrast_vessel_image.dat_0000.dcm"
OUTPUT_DIR = r"D:\med_data\biron\data1\test"

# 是否同时处理两个文件（True=同时处理mask和投影，False=只处理mask）
PROCESS_BOTH = True


# ============================================================
# 核心处理函数（保持原始尺寸）
# ============================================================

def rotate_3d_volume_preserve_size(volume, rot_x=0, rot_y=0, rot_z=0):
    """
    对3D体积进行绕X、Y、Z轴的旋转（90°的倍数），保持原始尺寸
    使用np.rot90配合axes参数，但会重新排列维度，需要通过转置来保持空间方向

    参数:
        volume: 3D numpy数组
        rot_x: 绕X轴旋转次数（0,1,2,3）
        rot_y: 绕Y轴旋转次数（0,1,2,3）
        rot_z: 绕Z轴旋转次数（0,1,2,3）

    返回:
        旋转后的体积（保持原始尺寸）
    """
    result = volume.copy()

    # 注意：对于90°旋转，图像尺寸在某些维度会交换
    # 例如：绕Z轴旋转90°，X和Y维度会交换
    # 这是正确的，因为视野方向改变了

    # 绕X轴旋转（在YZ平面旋转）
    if rot_x % 4 != 0:
        result = np.rot90(result, k=rot_x % 4, axes=(1, 2))

    # 绕Y轴旋转（在XZ平面旋转）
    if rot_y % 4 != 0:
        result = np.rot90(result, k=rot_y % 4, axes=(0, 2))

    # 绕Z轴旋转（在XY平面旋转）
    if rot_z % 4 != 0:
        result = np.rot90(result, k=rot_z % 4, axes=(0, 1))

    return result


def update_affine_for_rotation(original_affine, rot_x=0, rot_y=0, rot_z=0):
    """
    根据旋转参数更新仿射变换矩阵
    保持空间坐标系的一致性

    参数:
        original_affine: 原始4x4仿射矩阵
        rot_x, rot_y, rot_z: 旋转次数

    返回:
        更新后的仿射矩阵
    """
    affine = original_affine.copy()

    # 对于90°旋转，需要重新排列仿射矩阵的对应行/列
    # 简化处理：重新排列方向向量

    # 获取方向向量（3x3旋转部分）
    direction = affine[:3, :3]
    origin = affine[:3, 3]

    # 根据旋转重新排列方向向量
    # 绕X轴旋转：Y和Z方向交换并可能反转
    if rot_x % 4 != 0:
        if rot_x % 4 == 1:  # 90°
            direction = direction[:, [0, 2, 1]]
            direction[1] = -direction[1]  # Y方向反转
        elif rot_x % 4 == 2:  # 180°
            direction = direction[:, [0, 2, 1]]
            direction[1] = -direction[1]
            direction[2] = -direction[2]
        elif rot_x % 4 == 3:  # 270°
            direction = direction[:, [0, 2, 1]]
            direction[2] = -direction[2]

    # 绕Y轴旋转：X和Z方向交换并可能反转
    if rot_y % 4 != 0:
        if rot_y % 4 == 1:  # 90°
            direction = direction[:, [2, 1, 0]]
            direction[0] = -direction[0]
        elif rot_y % 4 == 2:  # 180°
            direction = direction[:, [2, 1, 0]]
            direction[0] = -direction[0]
            direction[2] = -direction[2]
        elif rot_y % 4 == 3:  # 270°
            direction = direction[:, [2, 1, 0]]
            direction[2] = -direction[2]

    # 绕Z轴旋转：X和Y方向交换并可能反转
    if rot_z % 4 != 0:
        if rot_z % 4 == 1:  # 90°
            direction = direction[:, [1, 0, 2]]
            direction[0] = -direction[0]
        elif rot_z % 4 == 2:  # 180°
            direction = direction[:, [1, 0, 2]]
            direction[0] = -direction[0]
            direction[1] = -direction[1]
        elif rot_z % 4 == 3:  # 270°
            direction = direction[:, [1, 0, 2]]
            direction[1] = -direction[1]

    # 更新仿射矩阵
    affine[:3, :3] = direction
    affine[:3, 3] = origin

    return affine


def rotate_2d_image_preserve_size(image, rot=0, flip_h=False, flip_v=False):
    """
    对2D图像进行旋转和翻转，保持原始尺寸

    参数:
        image: 2D numpy数组
        rot: 旋转次数（0,1,2,3）
        flip_h: 是否水平翻转
        flip_v: 是否垂直翻转

    返回:
        变换后的图像（保持原始尺寸）
    """
    result = image.copy()

    # 旋转（会交换高度和宽度）
    if rot % 4 != 0:
        result = np.rot90(result, k=rot % 4)

    # 水平翻转（左右）
    if flip_h:
        result = np.fliplr(result)

    # 垂直翻转（上下）
    if flip_v:
        result = np.flipud(result)

    return result


def save_3d_volume(volume, output_path, original_affine=None, description="",
                   rot_x=0, rot_y=0, rot_z=0):
    """
    保存3D体积为NIfTI格式，更新仿射变换
    """
    if original_affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    else:
        # 更新仿射变换以匹配旋转
        affine = update_affine_for_rotation(original_affine, rot_x, rot_y, rot_z)

    nii_img = nib.Nifti1Image(volume, affine)
    if description:
        nii_img.header['descrip'] = description[:80]

    # 更新header中的像素间距信息
    zooms = nii_img.header.get_zooms()
    nii_img.header.set_zooms(zooms)

    nib.save(nii_img, output_path)
    print(f"  ✓ 已保存: {output_path}")
    print(f"    体积形状: {volume.shape}")
    print(f"    体素尺寸: {zooms}")


def save_2d_image(image, output_path, original_shape=None):
    """
    保存2D图像为NIfTI格式
    """
    # 将2D图像作为3D保存（第三维为1）
    if len(image.shape) == 2:
        image_3d = image.reshape(image.shape[0], image.shape[1], 1)
    else:
        image_3d = image

    # 创建标准仿射矩阵
    # 对于2D投影，像素间距保持不变
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    nii_img = nib.Nifti1Image(image_3d, affine)

    # 设置像素间距（假设原始像素间距为0.546875mm，即280mm/512）
    pixel_spacing = 280.0 / 512  # 从原始代码中获取
    nii_img.header.set_zooms((pixel_spacing, pixel_spacing, 1.0))

    nib.save(nii_img, output_path)
    print(f"  ✓ 已保存: {output_path}")
    print(f"    图像形状: {image.shape}")
    print(f"    像素间距: {pixel_spacing:.4f}mm")


def process_3d_mask():
    """
    处理3D mask文件
    """
    print("\n" + "=" * 80)
    print("处理3D Mask文件")
    print("=" * 80)
    print(f"输入文件: {MASK_INPUT}")
    print(f"旋转参数: X轴={ROT_X * 90}°, Y轴={ROT_Y * 90}°, Z轴={ROT_Z * 90}°")
    print(f"真实参数: ROT_X={ROT_X}, ROT_Y={ROT_Y}, ROT_Z={ROT_Z}")

    # 检查输入文件
    if not os.path.exists(MASK_INPUT):
        print(f"错误：文件不存在 {MASK_INPUT}")
        return False

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print(f"\n加载文件...")
    nii_img = nib.load(MASK_INPUT)
    original_volume = nii_img.get_fdata()
    original_affine = nii_img.affine
    original_zooms = nii_img.header.get_zooms()

    print(f"  原始形状: {original_volume.shape}")
    print(f"  原始体素尺寸: {original_zooms}")
    print(f"  数值范围: [{np.min(original_volume):.3f}, {np.max(original_volume):.3f}]")
    print(f"  数据类型: {original_volume.dtype}")

    # 执行旋转
    print(f"\n执行旋转...")
    rotated_volume = rotate_3d_volume_preserve_size(original_volume, ROT_X, ROT_Y, ROT_Z)
    print(f"  旋转后形状: {rotated_volume.shape}")
    print(f"  旋转后体素总数: {rotated_volume.size} (原始: {original_volume.size})")

    # 验证体素总数是否保持不变
    if rotated_volume.size != original_volume.size:
        print(f"  警告：体素总数发生变化！")

    # 保存结果
    print(f"\n保存结果...")
    output_filename = f"mask_rotated_X{ROT_X * 90}Y{ROT_Y * 90}Z{ROT_Z * 90}.nii.gz"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_3d_volume(rotated_volume, output_path, original_affine,
                   f"Rotated X={ROT_X * 90} Y={ROT_Y * 90} Z={ROT_Z * 90}",
                   ROT_X, ROT_Y, ROT_Z)

    # 保存参数文件
    param_file = os.path.join(OUTPUT_DIR, "rotation_params_3d.txt")
    with open(param_file, 'w') as f:
        f.write(f"ROT_X={ROT_X}\n")
        f.write(f"ROT_Y={ROT_Y}\n")
        f.write(f"ROT_Z={ROT_Z}\n")
        f.write(f"Description: X={ROT_X * 90}deg, Y={ROT_Y * 90}deg, Z={ROT_Z * 90}deg\n")
        f.write(f"Original shape: {original_volume.shape}\n")
        f.write(f"Rotated shape: {rotated_volume.shape}\n")
        f.write(f"Original voxel size: {original_zooms}\n")
    print(f"  ✓ 参数已保存: {param_file}")

    print("\n" + "=" * 80)
    print("3D Mask处理完成！")
    print("=" * 80)
    return True


def process_2d_projection():
    """
    处理2D投影图像
    """
    print("\n" + "=" * 80)
    print("处理2D投影图像")
    print("=" * 80)
    print(f"输入文件: {PROJECTION_INPUT}")
    print(f"旋转参数: {ROT_2D * 90}°")
    print(f"水平翻转: {FLIP_HORIZONTAL}")
    print(f"垂直翻转: {FLIP_VERTICAL}")
    print(f"真实参数: ROT={ROT_2D}, FLIP_H={FLIP_HORIZONTAL}, FLIP_V={FLIP_VERTICAL}")

    # 检查输入文件
    if not os.path.exists(PROJECTION_INPUT):
        print(f"错误：文件不存在 {PROJECTION_INPUT}")
        return False

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print(f"\n加载文件...")
    ds = pydicom.dcmread(PROJECTION_INPUT)
    original_image = ds.pixel_array.astype(np.float32)
    original_shape = original_image.shape

    print(f"  原始形状: {original_shape}")
    print(f"  数值范围: [{np.min(original_image):.3f}, {np.max(original_image):.3f}]")
    print(f"  数据类型: {original_image.dtype}")

    # 执行变换
    print(f"\n执行变换...")
    transformed_image = rotate_2d_image_preserve_size(original_image, ROT_2D, FLIP_HORIZONTAL, FLIP_VERTICAL)
    print(f"  变换后形状: {transformed_image.shape}")

    # 保存结果
    print(f"\n保存结果...")
    flip_str = ""
    if FLIP_HORIZONTAL:
        flip_str += "H"
    if FLIP_VERTICAL:
        flip_str += "V"
    if flip_str:
        flip_str = f"_flip{flip_str}"

    output_filename = f"projection_rot{ROT_2D * 90}{flip_str}.nii.gz"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    save_2d_image(transformed_image, output_path, original_shape)

    # 保存参数文件
    param_file = os.path.join(OUTPUT_DIR, "transform_params_2d.txt")
    with open(param_file, 'w') as f:
        f.write(f"ROT={ROT_2D}\n")
        f.write(f"FLIP_H={FLIP_HORIZONTAL}\n")
        f.write(f"FLIP_V={FLIP_VERTICAL}\n")
        f.write(f"Description: Rotate={ROT_2D * 90}deg, H-Flip={FLIP_HORIZONTAL}, V-Flip={FLIP_VERTICAL}\n")
        f.write(f"Original shape: {original_shape}\n")
        f.write(f"Transformed shape: {transformed_image.shape}\n")
    print(f"  ✓ 参数已保存: {param_file}")

    print("\n" + "=" * 80)
    print("2D投影处理完成！")
    print("=" * 80)
    return True


def verify_rotation():
    """
    验证旋转是否正确（加载并检查保存的文件）
    """
    print("\n" + "=" * 80)
    print("验证旋转结果")
    print("=" * 80)

    # 验证3D mask
    mask_output = os.path.join(OUTPUT_DIR, f"mask_rotated_X{ROT_X * 90}Y{ROT_Y * 90}Z{ROT_Z * 90}.nii.gz")
    if os.path.exists(mask_output):
        print(f"\n验证3D Mask: {os.path.basename(mask_output)}")
        nii_img = nib.load(mask_output)
        data = nii_img.get_fdata()
        zooms = nii_img.header.get_zooms()
        print(f"  形状: {data.shape}")
        print(f"  体素尺寸: {zooms}")
        print(f"  数值范围: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"  有效二值: {np.all((data == 0) | (data == 1))}")

    # 验证2D投影
    flip_str = ""
    if FLIP_HORIZONTAL:
        flip_str += "H"
    if FLIP_VERTICAL:
        flip_str += "V"
    if flip_str:
        flip_str = f"_flip{flip_str}"

    proj_output = os.path.join(OUTPUT_DIR, f"projection_rot{ROT_2D * 90}{flip_str}.nii.gz")
    if os.path.exists(proj_output):
        print(f"\n验证2D投影: {os.path.basename(proj_output)}")
        nii_img = nib.load(proj_output)
        data = nii_img.get_fdata()
        zooms = nii_img.header.get_zooms()
        print(f"  形状: {data.shape}")
        print(f"  体素尺寸: {zooms}")
        print(f"  数值范围: [{np.min(data):.3f}, {np.max(data):.3f}]")


def main():
    """
    主函数
    """
    print("\n" + "=" * 80)
    print("图像旋转工具（保持原始尺寸和医学有效性）")
    print("=" * 80)
    print("注意：旋转会交换维度以保持空间方向正确性")
    print("例如：绕Z轴旋转90°后，X和Y维度会交换")
    print("这是正确的，因为视野方向改变了")
    print("=" * 80)

    print(f"\n当前配置:")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  处理3D Mask: {PROCESS_BOTH or True}")
    print(f"  处理2D投影: {PROCESS_BOTH}")

    if PROCESS_BOTH:
        # 处理3D mask
        success_3d = process_3d_mask()

        # 处理2D投影
        success_2d = process_2d_projection()

        if success_3d and success_2d:
            print("\n" + "=" * 80)
            print("所有文件处理完成！")
            print(f"输出目录: {OUTPUT_DIR}")
            print("=" * 80)

            # 验证结果
            verify_rotation()
        else:
            print("\n" + "=" * 80)
            print("部分文件处理失败，请检查错误信息")
            print("=" * 80)
    else:
        # 只处理3D mask
        success_3d = process_3d_mask()

        if success_3d:
            print("\n" + "=" * 80)
            print("3D Mask处理完成！")
            print(f"输出目录: {OUTPUT_DIR}")
            print("=" * 80)

            # 验证结果
            verify_rotation()


if __name__ == "__main__":
    main()