# Reference:
# https://github.com/olehb/pytorch_ddp_tutorial/blob/main/ddp_tutorial_multi_gpu.py


import os
import argparse
from typing import Tuple
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import time
from datetime import datetime


def create_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_loc = './mnist_data'

    train_dataset = datasets.MNIST(dataset_loc,
                                   download=True,
                                   train=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # This is not necessary to use distributed sampler for the test or validation sets.
    test_dataset = datasets.MNIST(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader


def create_model():
    # create model architecture
    model = nn.Sequential(
        nn.Linear(28*28, 128),  # MNIST images are 28x28 pixels
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10, bias=False)  # 10 classes to predict
    )
    return model


def main(epochs: int,
         model: nn.Module,
         train_loader: DataLoader,
         test_loader: DataLoader) -> nn.Module:

    device = torch.device("cuda")
    model = nn.DataParallel(model)
    model = model.to(device)

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    # train the model
    time_training_ls, time_val_ls = [], []
    for i in range(epochs):
        time_epoch = time.time()
        model.train()

        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = loss(y_hat, y)
            batch_loss.backward()
            optimizer.step()
            batch_loss_scalar = batch_loss.item()
            epoch_loss += batch_loss_scalar / x.shape[0]
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')
        time_training = time.time() - time_epoch

        # calculate validation loss
        time_val = time.time()
        with torch.no_grad():
            model.eval()
            val_loss = 0
            pbar = tqdm(test_loader)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                batch_loss_scalar = batch_loss.item()

                val_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')
        time_val = time.time() - time_val

        _print_and_log(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}", f_log)
        _print_and_log("Training time: {:.3f}s, Validation time: {:.3f}s".format(time_training, time_val),
                       f_log)
        time_training_ls.append(time_training)
        time_val_ls.append(time_val)

    _print_and_log("{:s} Overall timing results {:s}".format('-' * 10, '-' * 10), f_log)
    for i, time_training, time_val in zip(range(epochs), time_training_ls, time_val_ls):
        _print_and_log("Epoch: {:d}, Training time: {:.3f}s, Validation time: {:.3f}s.".format(
            i, time_training, time_val), f_log)

    return model.module


def _print_and_log(in_str, log_file):
    assert isinstance(in_str, str)
    print(in_str, flush=True)
    log_file.write(in_str + '\n')
    log_file.flush()


if __name__ == '__main__':

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    batch_size = 256
    epochs = 50

    num_gpus = torch.cuda.device_count()

    file_path = "logger_dp_{:s}_{:d}.txt".format(current_time, os.getpid())
    f_log = open(file_path, 'w')
    _print_and_log("Log saved at {:s}".format(file_path), f_log)
    _print_and_log("DP is ON. Process PID: {}.".format(os.getpid()), f_log)
    _print_and_log("#GPUs: {:d}.".format(num_gpus), f_log)

    train_loader, test_loader = create_data_loaders(batch_size)
    model = main(epochs=epochs,
                 model=create_model(),
                 train_loader=train_loader,
                 test_loader=test_loader)

    # if rank == 0:
    #     torch.save(model.state_dict(), 'model.pt')
