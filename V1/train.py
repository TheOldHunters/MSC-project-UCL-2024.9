import os
import shutil
import torch
import tqdm
from Src.vacs import VACSNet
import data_loader
from torch.utils import data
from matplotlib import pyplot as plt
from torch import optim
from tools.config import config
from tools.losses import loss_function


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr


def run(model, dataset, opt, epoch, optimizer=None):
    epoch_loss = 0
    loss_func = loss_function
    dt_size = len(dataset.dataset)
    if optimizer is not None:
        pbar = tqdm.tqdm(
            total=dt_size // config.Run_config.batch_size,
            desc=f'Epoch {epoch + 1} / {config.Run_config.num_epochs}',
            postfix=dict,
            miniters=.3
        )
    else:
        pbar = tqdm.tqdm(
            total=dt_size // config.Run_config.batch_size,
            desc=f'Val Epoch {epoch + 1} / {config.Run_config.num_epochs}',
            postfix=dict,
            miniters=.3
        )
    for i, batch in enumerate(dataset):
        source_img = batch['source'].to(opt.Run_config.device)
        target_img = batch['target'].to(opt.Run_config.device)
        mask_img = batch['mask'].to(opt.Run_config.device)
        # print(source_img.shape)
        if optimizer is not None:
            optimizer.zero_grad()
            output = model(source_img)

        else:
            with torch.no_grad():
                output = model(source_img)
        loss = loss_func(output, target_img, mask_img)
        epoch_loss += loss.item()

        if optimizer is not None:
            loss.backward()
            clip_gradient(optimizer, config.Run_config.clip)
            optimizer.step()
        if optimizer is not None:
            pbar.set_postfix(**{
                'train_loss': epoch_loss / (i + 1),
            })
        else:
            pbar.set_postfix(**{
                'val_loss': epoch_loss / (i + 1),
            })
        pbar.update(1)
    pbar.close()
    return epoch_loss


def main(config):
    opt = config
    os.makedirs(config.Run_config.weights_path, exist_ok=True)
    shutil.copyfile('./tools/config.py', config.Run_config.weights_path + '/config.py')
    shutil.copytree('./Src', config.Run_config.weights_path + '/Src_use')

    dataset_train = data_loader.Dataset(*data_loader.get_data_train_path(opt.Data_config.data_dir,
                                                                         opt.Run_config.fold), True)
    dataset_val = data_loader.Dataset(*data_loader.get_data_path(opt.Data_config.data_dir,
                                                                 opt.Run_config.fold), False)

    train_loader = data.DataLoader(dataset_train, batch_size=config.Run_config.batch_size,
                                   shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset_val, batch_size=config.Run_config.batch_size,
                                 shuffle=False, num_workers=4)

    model = VACSNet(opt.Data_config.num_classes)
    model = model.to(config.Run_config.device)
    if config.Run_config.pre_weights is not None:
        model.load_state_dict(torch.load(config.Run_config.pre_weights), strict=False)
    weights_path = config.Run_config.weights_path
    train_loss = open(weights_path + '/train_loss.txt', 'w')
    val_loss = open(weights_path + '/val_loss.txt', 'w')

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config.Run_config.lr, weight_decay=1e-4)

    train_loss_all, val_loss_all = [], []
    for epoch in range(config.Run_config.num_epochs):
        cur_lr = adjust_lr(optimizer, config.Run_config.lr, epoch,
                           config.Run_config.decay_rate, config.Run_config.decay_epoch)

        model.train()
        epoch_loss = run(model, train_loader, opt, epoch, optimizer)
        train_loss.write(str(epoch_loss / len(train_loader.dataset)))
        train_loss.write('\n')
        train_loss_all.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        epoch_loss = run(model, val_loader, opt, epoch)
        val_loss.write(str(epoch_loss / len(val_loader.dataset)))
        val_loss.write('\n')
        val_loss_all.append(epoch_loss / len(val_loader.dataset))

        plt.figure(figsize=(12, 12))
        plt.plot(train_loss_all, label='train loss')
        plt.plot(val_loss_all, label='vali loss')
        plt.legend()
        plt.savefig(weights_path + '/loss.png')
        if config.Run_config.save_loss_min > epoch_loss / len(val_loader.dataset):
            config.Run_config.save_loss_min = epoch_loss / len(val_loader.dataset)
            torch.save(model.state_dict(), weights_path + '/weights.pth')


def updata(config):
    config.Run_config.pre_weights = None
    config.Run_config.weights_path = './weights' + str(config.Run_config.fold) + '/exp'
    i = 0
    while os.path.exists(config.Run_config.weights_path + str(i)):
        if os.path.exists(config.Run_config.weights_path + str(i) + '/weights.pth'):
            config.Run_config.pre_weights = config.Run_config.weights_path + str(i) + '/weights.pth'
        i += 1
    config.Run_config.weights_path += str(i)
    return config


if __name__ == '__main__':
    for i in range(1, 5):
        config.Run_config.fold = i
        config.Run_config.save_loss_min = 100
        config_use = updata(config)
        main(config_use)
