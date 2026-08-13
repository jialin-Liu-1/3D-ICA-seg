import nibabel as nib
import nibabel as nib
import numpy as np

img = nib.load('D:/med_data/biron/Totalsegmentator_dataset_v201/s0000/ct.nii.gz')
data = img.get_fdata()

print(f"数据类型: {data.dtype}")
print(f"数据形状: {data.shape}")
print(f"最小值: {data.min()}")
print(f"最大值: {data.max()}")
print(f"是否全零: {np.all(data == 0)}")
print(f"非零像素数量: {np.count_nonzero(data)}")
# 将 qform 和 sform 都设置为与当前仿射矩阵一致
img.set_qform(img.affine, code=1)  # 1 代表 NIFTI_XFORM_SCANNER_ANAT
img.set_sform(img.affine, code=1)
nib.save(img, 'D:/med_data/biron/Totalsegmentator_dataset_v201/s0000/ct_fixed.nii.gz')