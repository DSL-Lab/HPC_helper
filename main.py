# Reference:
# https://github.com/olehb/pytorch_ddp_tutorial/blob/main/ddp_tutorial_multi_gpu.py


import os
import logging
import pdb
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn, optim
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision import datasets, transforms

from utils.arg_parser import parse_arguments, set_seed_and_logger, backup_code
from utils.dist_training import DistributedHelper, get_ddp_save_flag
from utils.learning_utils import count_model_params


def init_basics():
    """
    Initialization
    """
    args = parse_arguments()
    dist_helper = DistributedHelper(args.dp, args.ddp, args.ddp_gpu_ids, args.ddp_init_method)
    writer = set_seed_and_logger(args.seed, args.logdir, args.log_level, args.comment, dist_helper)
    backup_code(args.logdir)
    return args, dist_helper, writer


def init_model(dist_helper):
    """
    Initialize model and training necessities.
    """
    # model, we use an unnecessarily heavy model to showcase the GPU profiling
    model = getattr(torchvision.models, 'resnet50')(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes to predict
    model = model.to(dist_helper.device)

    param_string, total_params, total_trainable_params = count_model_params(model)
    logging.info(f"Parameters: \n{param_string}")
    logging.info(f"Parameters Count: {total_params:,}, Trainable: {total_trainable_params:,}.")

    # adapt to distributed training
    model = dist_helper.dist_adapt_model(model)

    # optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def init_dataloader(batch_size, dist_helper):
    """
    Get dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize(128)  # resize to larger image to showcase the use of GPU profiling
    ])
    dataset_loc = './mnist_data'

    train_dataset = datasets.MNIST(dataset_loc, download=True, train=True, transform=transform)

    # For final evaluation, it is advised not to use distributed sampler due to possibly incorrect results.
    # But we are using it now to accelerate evaluation during training.
    # Ref: https://github.com/pytorch/pytorch/issues/25162
    test_dataset = datasets.MNIST(dataset_loc, download=True, train=False, transform=transform)

    logging.info("Training set size: {:d}, testing set size: {:d}".format(len(train_dataset), len(test_dataset)))

    if dist_helper.is_ddp:
        batch_size_per_gpu = max(1, batch_size // dist.get_world_size())
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(dataset=train_dataset, sampler=sampler, batch_size=batch_size_per_gpu,
                                  pin_memory=True, num_workers=min(6, os.cpu_count()))

        sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, sampler=sampler, batch_size=batch_size_per_gpu,
                                 pin_memory=True, num_workers=min(6, os.cpu_count()))
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True, num_workers=min(6, os.cpu_count()))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, pin_memory=True, num_workers=min(6, os.cpu_count()))

    return train_loader, test_loader


def go_training(epochs, model, optimizer, criterion, dist_helper, train_loader, test_loader, writer, logdir):
    """
    Training loop.
    """

    # init
    time_train_ls, time_val_ls = [], []
    epoch_when_snapshot = list(range(0, epochs, epochs // 5))

    # epoch-wise training
    for i_epoch in range(epochs):
        # train the model for one epoch
        if dist_helper.is_ddp:
            train_loader.sampler.set_epoch(i_epoch)

        time_epoch = time.time()
        train_loss = 0
        pbar = tqdm(train_loader)
        model.train()
        for x, y in pbar:
            x = x.to(dist_helper.device, non_blocking=True)
            y = y.to(dist_helper.device, non_blocking=True)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = criterion(y_hat, y)
            batch_loss.backward()
            optimizer.step()
            batch_loss_scalar = batch_loss.item()
            train_loss += batch_loss_scalar / x.shape[0]
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')
        time_training = time.time() - time_epoch

        # calculate validation loss
        time_val = time.time()
        val_loss = 0.0
        pbar = tqdm(test_loader)
        model.eval()
        with torch.no_grad():
            for x, y in pbar:
                x = x.to(dist_helper.device, non_blocking=True)
                y = y.to(dist_helper.device, non_blocking=True)
                y_hat = model(x)
                batch_loss = criterion(y_hat, y)
                batch_loss_scalar = batch_loss.item()
                val_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')
        time_val = time.time() - time_val

        logging.info(f"Epoch={i_epoch}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        logging.info("Training time: {:.3f}s, Validation time: {:.3f}s".format(time_training, time_val))
        time_train_ls.append(time_training)
        time_val_ls.append(time_val)

        if get_ddp_save_flag():
            writer.add_scalar("train/loss", train_loss, i_epoch)
            writer.add_scalar("test/loss", val_loss, i_epoch)
            writer.flush()

        if i_epoch in epoch_when_snapshot and get_ddp_save_flag():
            model_path = os.path.join(logdir, 'model_epoch_{:03d}_{:s}_{:d}.pt'.format(
                i_epoch, datetime.now().strftime("%Y%m%d-%H%M%S"), os.getpid()))
            torch.save(model.state_dict(), model_path)
            logging.info("Saving model to {:s}".format(model_path))
        dist_helper.ddp_sync()

    # Count overall training efficiency
    logging.info("{:s} Overall timing results {:s}".format('-' * 10, '-' * 10))
    logging.info("Total training time: {:.3f}s, total validation time: {:.3f}s".format(
        np.sum(time_train_ls), np.sum(time_val_ls)))
    for i_epoch, time_training, time_val in zip(range(epochs), time_train_ls, time_val_ls):
        logging.info("Epoch: {:d}, Training time: {:.3f}s, Validation time: {:.3f}s.".format(
            i_epoch, time_training, time_val))


def main():
    """
    Main training loop
    """

    """Initialization basics"""
    args, dist_helper, writer = init_basics()

    """Get network"""
    model, optimizer, criterion = init_model(dist_helper)

    """Get dataloader"""
    train_loader, test_loader = init_dataloader(args.batch_size, dist_helper)

    """Go training"""
    go_training(args.epoch, model, optimizer, criterion, dist_helper, train_loader, test_loader, writer, args.logdir)

    """Distributed training cleanup"""
    dist_helper.clean_up()

    
if __name__ == '__main__':
    main()
