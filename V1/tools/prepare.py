import copy
import os

import SimpleITK
from PIL import Image
import numpy as np

if __name__ == '__main__':
    fold_data_path = 'F:/Final Program/data/folds_new'
    unlabel_data_path = 'F:/Final Program/data/Images_5FPS'
    path_out = 'F:/video seg/data_prepare/data_prepare'
    for i in range(5):
        os.makedirs(path_out + '/fold' + str(i), exist_ok=True)
        os.makedirs(path_out + '/fold' + str(i) + '/images', exist_ok=True)
        os.makedirs(path_out + '/fold' + str(i) + '/masks', exist_ok=True)
        fold_path = fold_data_path + '/fold' + str(i)
        img_path = fold_path + '/images'
        files = os.listdir(img_path)
        flag = np.ones(shape=(len(files, )))
        while flag.sum() != 0:
            print(1 - np.sum(flag) / len(flag))
            index = np.where(flag > 0)[0][0]
            flag[index] = 0
            file = files[index]
            name = file.split('_')[0]
            patients = [img_path + '/' + file]
            for j in range(len(files)):
                if flag[j] == 0:
                    continue
                if files[j].split('_')[0] == name:
                    patients.append(img_path + '/' + files[j])
                    flag[j] = 0
            mask_file = copy.deepcopy(patients)
            for ufile in os.listdir(unlabel_data_path):
                if ufile.split('_')[0] == name and img_path + '/' + ufile not in patients:
                    patients.append(unlabel_data_path + '/' + ufile)
            patients.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
            imgs = []
            masks = []
            for k, file in enumerate(patients):
                img = np.array(Image.open(file))
                imgs.append(img)
                if file in mask_file:
                    mask = np.array(Image.open(file.replace('/images/', '/masks/')))
                    masks.append(mask)
                else:
                    masks.append(np.zeros(shape=img.shape[:2]))

            imgs = np.array(imgs)
            masks = np.array(masks)
            os.makedirs(path_out + '/fold' + str(i) + '/images/' + name, exist_ok=True)
            os.makedirs(path_out + '/fold' + str(i) + '/masks/' + name, exist_ok=True)
            for ii in range(0, len(imgs), 8):
                z_min = ii
                z_max = ii + 8

                if z_max >= len(imgs):
                    z_max = len(imgs) - 1
                    z_min = z_max - 8
                z_img = imgs[z_min: z_max]
                z_mask = masks[z_min: z_max]
                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(z_img),
                                     path_out + '/fold' + str(i) + '/images/' + name + '/' + str(z_min) + '.nii')

                SimpleITK.WriteImage(SimpleITK.GetImageFromArray(z_mask),
                                     path_out + '/fold' + str(i) + '/masks/' + name + '/' + str(z_min) + '.nii')
