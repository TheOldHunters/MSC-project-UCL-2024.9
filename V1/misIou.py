import os
import numpy as np
import nibabel as nib


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0  # 避免除以零
    else:
        return intersection / union


def load_mask(filepath):
    nii_img = nib.load(filepath)
    mask = nii_img.get_fdata()  # 加载3D体积数据
    return mask


def compute_misaligned_iou_within_nii(directory):
    iou_scores_all_files = {}
    file_list = sorted([f for f in os.listdir(directory) if f.endswith('.nii')])  # 只处理.nii文件

    total_iou_scores = []

    for nii_file in file_list:
        file_path = os.path.join(directory, nii_file)
        mask = load_mask(file_path)
        num_slices = mask.shape[-1]

        iou_scores = []
        for slice_idx in range(num_slices - 1):
            current_slice = mask[..., slice_idx]
            next_slice = mask[..., slice_idx + 1]

            # 使用实际的前景像素值来提取前景区域
            current_foreground_mask = np.logical_or.reduce(
                (current_slice == 1.0, current_slice == 2.0, current_slice == 3.0, current_slice == 4.0))
            next_foreground_mask = np.logical_or.reduce(
                (next_slice == 1.0, next_slice == 2.0, next_slice == 3.0, next_slice == 4.0))

            if np.sum(current_foreground_mask) == 0 and np.sum(next_foreground_mask) == 0:
                continue

            iou = calculate_iou(current_foreground_mask, next_foreground_mask)
            iou_scores.append(iou)

        average_iou = np.mean(iou_scores) if iou_scores else 0.0
        iou_scores_all_files[nii_file] = average_iou
        total_iou_scores.extend(iou_scores)  # 将所有 IOU 值添加到总列表中
        print(f'Average Foreground IOU within {nii_file}: {average_iou}')

    overall_average_iou = np.mean(total_iou_scores) if total_iou_scores else 0.0
    print(f'Overall Average Foreground IOU across all NII files: {overall_average_iou}')

    return iou_scores_all_files, overall_average_iou


# 选取哪个weights来算错位iou
directory_path = './weights4/exp0/result'
iou_scores_within_files, overall_average_iou = compute_misaligned_iou_within_nii(directory_path)
