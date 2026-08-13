import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_nrrd_to_nifti_batch(source_dir, target_dir, batch_size=10, start_idx=0, end_idx=None):
    """
    批量将.nrrd文件转换为.nii.gz文件（使用SimpleITK）

    Parameters:
    -----------
    source_dir : str
        源文件夹路径，包含.nrrd文件
    target_dir : str
        目标文件夹路径，保存.nii.gz文件
    batch_size : int
        每批处理的文件数量
    start_idx : int
        起始索引（用于断点续传）
    end_idx : int or None
        结束索引，None表示处理到最后
    """

    # 创建目标文件夹
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有.nrrd文件
    nrrd_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.nrrd')])

    if not nrrd_files:
        print(f"在 {source_dir} 中没有找到.nrrd文件")
        return

    # 过滤已存在的文件（跳过已转换的）
    existing_files = set(os.listdir(target_dir))
    nrrd_files = [f for f in nrrd_files if f.replace('.nrrd', '.nii.gz') not in existing_files]

    if not nrrd_files:
        print("所有文件已转换完成！")
        return

    # 应用索引范围
    if end_idx is None:
        end_idx = len(nrrd_files)
    nrrd_files = nrrd_files[start_idx:end_idx]

    total_files = len(nrrd_files)
    print(f"找到 {total_files} 个待转换的.nrrd文件")
    print(f"批次大小: {batch_size}")
    print(f"总批次数: {(total_files + batch_size - 1) // batch_size}")
    print("-" * 60)

    # 分批处理
    success_count = 0
    fail_count = 0
    failed_files = []

    for batch_num, i in enumerate(range(0, total_files, batch_size)):
        batch_files = nrrd_files[i:i + batch_size]

        # 使用tqdm显示进度
        with tqdm(total=len(batch_files), desc=f"批次 {batch_num + 1}", unit="file") as pbar:
            for filename in batch_files:
                try:
                    # 构建完整的文件路径
                    source_path = os.path.join(source_dir, filename)

                    # 生成目标文件名（保持数字编号不变）
                    base_name = filename.replace('.nrrd', '')
                    target_filename = base_name + '.nii.gz'
                    target_path = os.path.join(target_dir, target_filename)

                    # 检查目标文件是否已存在（双重检查）
                    if os.path.exists(target_path):
                        pbar.update(1)
                        pbar.set_postfix({"状态": "跳过", "成功": success_count, "失败": fail_count})
                        continue

                    # 使用SimpleITK读取.nrrd文件
                    img = sitk.ReadImage(source_path)

                    # 获取图像信息（可选，用于验证）
                    # print(f"  图像尺寸: {img.GetSize()}")
                    # print(f"  像素类型: {img.GetPixelIDTypeAsString()}")
                    # print(f"  方向: {img.GetDirection()}")

                    # 保存为.nii.gz文件
                    # 使用SetCompression启用压缩
                    sitk.WriteImage(img, target_path, useCompression=True)

                    success_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"状态": "成功", "成功": success_count, "失败": fail_count})

                except Exception as e:
                    fail_count += 1
                    failed_files.append(filename)
                    pbar.update(1)
                    pbar.set_postfix({"状态": "失败", "成功": success_count, "失败": fail_count})
                    print(f"\n  ❌ 转换文件 {filename} 时出错: {str(e)}")
                    continue

        # 批次处理完成后的统计信息
        print(
            f"批次 {batch_num + 1} 完成 - ✅ 成功: {success_count}, ❌ 失败: {fail_count}, 总进度: {success_count + fail_count}/{total_files}")
        print("-" * 60)

    # 最终统计
    print(f"\n{'=' * 60}")
    print("转换完成统计:")
    print(f"✅ 成功转换: {success_count} 个文件")
    print(f"❌ 转换失败: {fail_count} 个文件")
    if failed_files:
        print(f"\n失败的文件列表:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"📁 转换后的文件保存在: {target_dir}")
    print("=" * 60)


def verify_conversion(source_dir, target_dir):
    """
    验证转换结果，检查是否有文件丢失或损坏
    """
    source_files = set([f.replace('.nrrd', '') for f in os.listdir(source_dir) if f.endswith('.nrrd')])
    target_files = set([f.replace('.nii.gz', '') for f in os.listdir(target_dir) if f.endswith('.nii.gz')])

    missing = source_files - target_files
    extra = target_files - source_files

    print(f"\n{'=' * 60}")
    print("验证结果:")
    if missing:
        print(f"⚠️  以下文件未转换: {missing}")
        print(f"未转换数量: {len(missing)}")
    else:
        print("✅ 所有源文件都已成功转换！")

    if extra:
        print(f"⚠️  目标文件夹中有额外文件: {extra}")
        print(f"额外文件数量: {len(extra)}")

    # 验证文件完整性（随机检查几个文件）
    if target_files:
        sample_size = min(3, len(target_files))
        sample_files = sorted(list(target_files))[:sample_size]
        print(f"\n验证文件完整性（抽样检查 {sample_size} 个文件）:")
        for base_name in sample_files:
            target_path = os.path.join(target_dir, base_name + '.nii.gz')
            try:
                img = sitk.ReadImage(target_path)
                print(f"  ✅ {base_name}.nii.gz - 尺寸: {img.GetSize()}, 类型: {img.GetPixelIDTypeAsString()}")
            except Exception as e:
                print(f"  ❌ {base_name}.nii.gz - 读取失败: {str(e)}")

    print("=" * 60)
    return missing


def get_file_info(source_dir, target_dir):
    """
    获取文件信息用于验证
    """
    print(f"\n{'=' * 60}")
    print("文件信息:")
    print(f"📂 源文件夹: {source_dir}")
    nrrd_files = [f for f in os.listdir(source_dir) if f.endswith('.nrrd')]
    nrrd_count = len(nrrd_files)
    print(f"  .nrrd文件数量: {nrrd_count}")

    if nrrd_count > 0:
        sample_files = sorted(nrrd_files)[:5]
        print(f"  示例文件: {sample_files}")

    print(f"\n📂 目标文件夹: {target_dir}")
    nifti_files = [f for f in os.listdir(target_dir) if f.endswith('.nii.gz')]
    nifti_count = len(nifti_files)
    print(f"  .nii.gz文件数量: {nifti_count}")

    if nifti_count > 0:
        sample_files = sorted(nifti_files)[:5]
        print(f"  示例文件: {sample_files}")
    print("=" * 60)


def get_sample_nrrd_info(source_dir, sample_file=None):
    """
    获取一个样本nrrd文件的信息，用于调试
    """
    if sample_file is None:
        nrrd_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.nrrd')])
        if not nrrd_files:
            print("没有找到.nrrd文件")
            return
        sample_file = nrrd_files[0]

    source_path = os.path.join(source_dir, sample_file)
    try:
        img = sitk.ReadImage(source_path)
        print(f"\n样本文件信息 ({sample_file}):")
        print(f"  尺寸: {img.GetSize()}")
        print(f"  像素类型: {img.GetPixelIDTypeAsString()}")
        print(f"  方向: {img.GetDirection()}")
        print(f"  原点: {img.GetOrigin()}")
        print(f"  间距: {img.GetSpacing()}")
        print(f"  像素数量: {img.GetNumberOfPixels()}")
    except Exception as e:
        print(f"读取样本文件失败: {str(e)}")


# 主程序
if __name__ == "__main__":
    # 设置路径
    source_dir = r"D:\med_data\biron\data1\seg"
    target_dir = r"D:\med_data\biron\data1\mask_seg"

    # 参数设置
    BATCH_SIZE = 4  # 每批处理10个文件，可根据内存情况调整
    START_IDX = 0  # 起始索引，用于断点续传
    END_IDX = None  # 结束索引，None表示处理全部

    # 是否显示样本信息
    SHOW_SAMPLE_INFO = True

    print("=" * 60)
    print("🗂️  NRRD 转 NIfTI 批量转换工具 (SimpleITK版本)")
    print("=" * 60)

    # 显示文件信息
    get_file_info(source_dir, target_dir)

    # 显示样本信息（用于调试）
    if SHOW_SAMPLE_INFO:
        get_sample_nrrd_info(source_dir)

    print("\n开始转换...")
    print("=" * 60)

    # 执行转换
    try:
        convert_nrrd_to_nifti_batch(
            source_dir=source_dir,
            target_dir=target_dir,
            batch_size=BATCH_SIZE,
            start_idx=START_IDX,
            end_idx=END_IDX
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断了转换过程")
        print("提示: 重新运行程序将自动跳过已转换的文件")
    except Exception as e:
        print(f"\n❌ 转换过程出现错误: {str(e)}")

    # 验证转换结果
    print("\n验证转换结果...")
    verify_conversion(source_dir, target_dir)

    # 再次显示文件信息
    get_file_info(source_dir, target_dir)

    print("\n" + "=" * 60)
    print("✅ 程序执行完成！")
    print("=" * 60)
