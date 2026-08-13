import os
import shutil
from pathlib import Path
from tqdm import tqdm


def copy_folders_by_last_digit_advanced(source_dir, target_dir, target_digits=[0, 3, 5, 8], show_progress=True):
    """
    增强版：根据文件夹名称最后一位数字筛选并复制文件夹

    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        target_digits: 需要筛选的数字列表
        show_progress: 是否显示进度条
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    # 获取所有文件夹
    folders = [f for f in source_path.iterdir() if f.is_dir()]

    # 筛选匹配的文件夹
    matched_folders = []
    for folder in folders:
        folder_name = folder.name
        last_char = folder_name[-1]
        if last_char.isdigit() and int(last_char) in target_digits:
            matched_folders.append(folder)

    print(f"找到 {len(matched_folders)} 个匹配的文件夹")
    print(f"目标目录: {target_dir}")
    print("=" * 60)

    if not matched_folders:
        print("没有找到匹配的文件夹")
        return

    # 复制文件夹
    copied_count = 0
    total_size = 0

    iterator = tqdm(matched_folders, desc="复制进度") if show_progress else matched_folders

    for folder in iterator:
        folder_name = folder.name
        src_path = folder
        dst_path = target_path / folder_name

        try:
            # 如果目标文件夹已存在，先删除
            if dst_path.exists():
                shutil.rmtree(dst_path)

            # 复制整个文件夹
            shutil.copytree(src_path, dst_path)

            # 计算复制的大小
            size = sum(f.stat().st_size for f in dst_path.rglob('*') if f.is_file())
            total_size += size
            copied_count += 1

            if not show_progress:
                print(f"  ✓ {folder_name} ({size / 1024:.2f} KB)")

        except Exception as e:
            print(f"  ✗ {folder_name} 复制失败: {str(e)}")

    # 输出统计信息
    print("=" * 60)
    print(f"复制完成! 共复制 {copied_count} 个文件夹")
    print(f"总大小: {total_size / 1024 / 1024:.2f} MB")

    # 显示复制的文件夹列表
    print("\n已复制的文件夹:")
    for folder in sorted([f.name for f in matched_folders]):
        print(f"  - {folder}")


def main():
    # 配置路径
    source_dir = r"D:\med_data\biron\data1\train"
    target_dir = r"D:\med_data\biron\data1\train2"

    # 筛选条件：最后一位数字为 0, 3, 5, 8
    target_digits = [0, 3, 5, 8]

    # 执行复制
    copy_folders_by_last_digit_advanced(source_dir, target_dir, target_digits)

    # 验证
    print("\n" + "=" * 60)
    print("验证:")
    target_path = Path(target_dir)
    if target_path.exists():
        copied = [f.name for f in target_path.iterdir() if f.is_dir()]
        print(f"目标目录中包含 {len(copied)} 个文件夹")
        print(f"期望复制: {len([0, 3, 5, 8])} 个数字的文件夹")
    else:
        print("目标目录不存在")


if __name__ == "__main__":
    main()