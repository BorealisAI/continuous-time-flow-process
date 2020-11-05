# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pickle

import numpy as np
import torch
from torch.utils import data

TRAIN_SPLIT_PERCENTAGE = 0.7
VAL_SPLIT_PERCENTAGE = 0.8


def get_test_dataset(args, test_batch_size):
    """
    Function for getting the dataset for testing

    Parameters:
        args: the arguments from parse_arguments in ctfp_tools
        test_batch_size: batch size used for data

    Returns:
        test_loader: the dataloader for testing
    """
    test_set = BMSequence(data_path=args.data_path, split=args.test_split)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return test_loader


def get_dataset(args):
    """
    Function for getting the dataset for training and validation

    Parameters:
        args: the arguments from parse_arguments in ctfp_tools
        return the dataloader for training and validation

    Returns:
        train_loader: data loader of training data
        val_loader: data loader of validation data
    """
    train_set = BMSequence(data_path=args.data_path, split="train")
    val_set = BMSequence(data_path=args.data_path, split="val")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader


class BMSequence(data.dataset.Dataset):
    """
    Dataset class for observations on irregular time grids of synthetic continuous
    time stochastic processes
    data_path: path to a pickle file storing the data
    split: split of the data, train, val, or test
    """

    def __init__(self, data_path, split="train"):
        super(BMSequence, self).__init__()
        f = open(data_path, "rb")
        self.data = pickle.load(f)
        f.close()
        self.max_length = 0
        for item in self.data:
            self.max_length = max(len(item), self.max_length)
        total_length = len(self.data)
        train_split = int(total_length * TRAIN_SPLIT_PERCENTAGE)
        val_split = int(total_length * VAL_SPLIT_PERCENTAGE)
        if split == "train":
            self.data = self.data[:train_split]
        elif split == "val":
            self.data = self.data[train_split:val_split]
        elif split == "test":
            self.data = self.data[val_split:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = np.array(self.data[index])
        item_len = item.shape[0]
        item_times = item[:, 0]
        item_times_shift = np.zeros_like(item_times)
        item_times_shift[1:] = item_times[:-1]
        item_values = item[:, 1]
        padded_times = torch.zeros(self.max_length)
        ## Pad all the sequences to the max length with value of 100
        ## Any value greater than zero can be used
        padded_values = torch.zeros(self.max_length) + 100
        masks = torch.ByteTensor(self.max_length).zero_()
        padded_times[:item_len] = torch.Tensor(item_times).type(torch.FloatTensor)
        padded_values[:item_len] = torch.Tensor(item_values).type(torch.FloatTensor)
        masks[:item_len] = 1
        padded_variance = torch.ones(self.max_length)
        padded_variance[:item_len] = torch.Tensor(item_times - item_times_shift).type(
            torch.FloatTensor
        )
        return (
            padded_values.unsqueeze(1),
            padded_times.unsqueeze(1),
            padded_variance,
            masks,
        )
