import os


class config:
    class Data_config:
        data_dir = 'F:/video seg/data_prepare/data_prepare'
        num_classes = 5

    class Run_config:
        # test on fold 0
        fold = 4
        device = 'cuda:0'
        weight_decay = 4e-3
        lr = 1e-4
        batch_size = 8
        num_epochs = 120
        save_loss_min = 100
        clip = 0.5
        decay_rate = 0.1
        decay_epoch = 50
