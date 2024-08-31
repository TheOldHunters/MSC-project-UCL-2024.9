import os
import SimpleITK
import albumentations as album
import numpy as np
import torch
from torch.utils import data


def get_data_train_path(dirs, fold):
    imgs, masks = [], []
    for i in range(5):
        if i == fold:
            continue
        img, mask = get_data_path(dirs, i)
        imgs.extend(img)
        masks.extend(mask)
    return imgs, masks


def get_data_path(dirs, fold):
    img_path = dirs + '/fold' + str(fold) + '/images'
    label_path = dirs + '/fold' + str(fold) + '/masks'
    img_files = os.listdir(img_path)
    imgs, masks = [], []
    for file in img_files:
        for name in os.listdir(img_path + '/' + file):
            imgs.append(img_path + '/' + file + '/' + name)
            masks.append(label_path + '/' + file + '/' + name)
    return imgs, masks


class Dataset(data.Dataset):
    def __init__(self, imgs, masks, train=True):
        self.imgs = imgs
        self.is_train = train
        self.masks = masks
        if self.masks is not None:
            assert len(self.imgs) == len(self.masks)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.transform = album.Compose([
            album.Resize(height=512, width=512),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.OneOf([
                album.IAAAdditiveGaussianNoise(),
                album.GaussNoise(),
            ], p=0.2),
            album.OneOf([
                album.MotionBlur(p=0.2),
                album.MedianBlur(blur_limit=3, p=0.1),
                album.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            album.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            album.RandomBrightnessContrast(p=0.2),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        image = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(img_path))
        if self.masks is not None:
            mask_path = self.masks[index]
            mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(mask_path))

            mask[mask == 64] = 1  # right carotid
            mask[mask == 128] = 2  # left carotid
            mask[mask == 192] = 3  # sella
            mask[mask == 255] = 4  # clival
            keep = np.sum(mask, axis=(1, 2)) > 0
            imgs_resize, mask_resize = [], []
            for z in range(len(image)):
                sample = self.transform(image=image[z], mask=mask[z])
                imgs_resize.append(sample['image'])
                mask_resize.append(sample['mask'])
            mask_resize = np.array(mask_resize)
            imgs_resize = np.array(imgs_resize)
            mask_resize = torch.from_numpy(mask_resize).float()
            imgs_resize = (imgs_resize / 255.0 - self.mean) / self.std
            imgs_resize = torch.from_numpy(imgs_resize.transpose(0, 3, 1, 2)).float()
            # return imgs_resize, mask_resize
            keep = torch.from_numpy(keep).long()
            return {'source': imgs_resize, 'target': mask_resize, 'mask': keep}


if __name__ == '__main__':
    from tools.config import config

    opt = config
    dataset_train = Dataset(*get_data_train_path(opt.Data_config.data_dir,
                                                 opt.Run_config.fold), True)
    dataset_val = Dataset(*get_data_path(opt.Data_config.data_dir,
                                         opt.Run_config.fold), False)

    train_loader = data.DataLoader(dataset_train, batch_size=config.Run_config.batch_size,
                                   shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset_val, batch_size=config.Run_config.batch_size,
                                 shuffle=False, num_workers=4)
    for i, batch in enumerate(train_loader):
        print(batch)
