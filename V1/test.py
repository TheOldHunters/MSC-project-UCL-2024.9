import os
import sys
from tqdm import tqdm
from tools import eva
import SimpleITK
import albumentations as album
import numpy as np
import torch
from prettytable import PrettyTable


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size


if __name__ == '__main__':
    device = 'cuda:1'
    path = 'weights0/exp1'
    weights_path = path + '/weights.pth'
    sys.path.append(path)
    from config import config as opt
    from Src_use.vacs import VACSNet

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    transform = album.Compose([
        album.Resize(height=512, width=512),
    ])

    transform_infer = album.Compose([
        album.Resize(height=720, width=1280),
    ])

    result_path = path + '/result'
    os.makedirs(result_path, exist_ok=True)
    opt.Run_config.device = device
    model = VACSNet(opt.Data_config.num_classes)
    model = model.to(opt.Run_config.device)
    EVA = eva.Evaluator(5)
    EVA1 = eva.Evaluator(5)
    key = model.load_state_dict(torch.load(weights_path))
    # model.eval()
    print(key)
    test_path = opt.Data_config.data_dir + '/fold' + str(opt.Run_config.fold)
    print(test_path)

    img_path = test_path + '/images'

    for name in tqdm(os.listdir(img_path)):
        print(name)
        index = []
        pres = []
        labels = []
        for file in os.listdir(img_path + '/' + name):
            index.append(int(file.split('.')[0]))
            image = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(img_path + '/' + name + '/' + file))
            label = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(img_path.replace('/images', '/masks') + '/'
                                                                    + name + '/' + file))
            labels.append(label)
            imgs_resize = []
            for z in range(len(image)):
                sample = transform(image=image[z])
                imgs_resize.append(sample['image'])
            imgs_resize = np.array(imgs_resize)

            imgs_resize = (imgs_resize / 255.0 - mean) / std
            imgs_resize = torch.from_numpy(imgs_resize.transpose(0, 3, 1, 2)).float()
            with torch.no_grad():
                imgs_resize = imgs_resize.to(device)
                pred = model(imgs_resize[None])['seg_final'][0]
            pred = torch.softmax(pred, dim=1).cpu().numpy()
            pres.append(pred)
        max_z = np.max(index) + 8
        pres_out = np.zeros(shape=(max_z, 5, 512, 512))
        f = np.zeros(shape=(max_z, 1, 512, 512))
        labels_out = np.zeros(shape=(max_z, *labels[0].shape[1:]))
        for j, i in enumerate(index):
            pres_out[i: i + 8] += pres[j]
            f[i: i + 8] += 1
            labels_out[i: i + 8] = labels[j]
        pres_out = pres_out / f
        z = np.where(labels_out > 0)[0]
        z_use = np.unique(z)
        print(z_use - 1)
        label_use = labels_out[z_use]
        label_use[label_use == 64] = 1
        label_use[label_use == 128] = 2
        label_use[label_use == 192] = 3
        label_use[label_use == 255] = 4
        pres_out = np.argmax(pres_out, axis=1).astype(np.uint8)
        pres_resize = []
        for i in range(len(pres_out)):
            pres_resize.append(transform_infer(image=pres_out[i])['image'])
        pres_resize = np.array(pres_resize)
        pres_resize_use_v1 = pres_resize[z_use - 1]
        pres_resize_use = pres_resize[z_use]
        EVA.add_batch(label_use, pres_resize_use)
        EVA1.add_batch(label_use, pres_resize_use_v1)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(pres_resize), result_path + '/' + name + '.nii')
    table = PrettyTable(['fold: ' + str(opt.Run_config.fold), 'Dice', 'Recall', 'Precision', 'F1', 'IoU'])
    for i in range(len(EVA.Dice())):
        table.add_row([str(i),
                       str(EVA.Dice()[i]),
                       str(EVA.Recall()[i]),
                       str(EVA.Precision()[i]),
                       str(EVA.F1()[i]),
                       str(EVA.Intersection_over_Union()[i]),
                       ])
    table.add_row(['mean',
                   str(EVA.Dice().mean()),
                   str(EVA.Recall().mean()),
                   str(EVA.Precision().mean()),
                   str(EVA.F1().mean()),
                   str(EVA.Intersection_over_Union().mean())
                   ])
    print(table)

    table = PrettyTable(['consistency fold: ' + str(opt.Run_config.fold), 'Dice', 'Recall', 'Precision', 'F1', 'IoU'])
    for i in range(len(EVA1.Dice())):
        table.add_row([str(i),
                       str(EVA1.Dice()[i]),
                       str(EVA1.Recall()[i]),
                       str(EVA1.Precision()[i]),
                       str(EVA1.F1()[i]),
                       str(EVA1.Intersection_over_Union()[i]),
                       ])
    table.add_row(['mean',
                   str(EVA1.Dice().mean()),
                   str(EVA1.Recall().mean()),
                   str(EVA1.Precision().mean()),
                   str(EVA1.F1().mean()),
                   str(EVA1.Intersection_over_Union().mean())
                   ])
    print(table)
    total = getModelSize(model)
    print("Number of parameter: %.2f / MB" % total)
