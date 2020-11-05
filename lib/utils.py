# MIT License
#
# Copyright (c) 2019 Yulia Rubanova
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Link: https://github.com/YuliaRubanova/latent_ode

import logging
import math
import os
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def optimizer_factory(args, parameter_list):
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            parameter_list,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError
    num_params = sum(p.numel() for p in parameter_list if p.requires_grad)
    return optimizer, num_params


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert start.size() == end.size()
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res, torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def check_mask(data, mask):
    # check that "mask" argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.0).cpu().numpy()
    n_ones = torch.sum(mask == 1.0).cpu().numpy()

    # mask should contain only zeros and ones
    assert (n_zeros + n_ones) == np.prod(list(mask.size()))

    # all masked out elements should be zeros
    assert torch.sum(data[mask == 0.0] != 0.0) == 0


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)
    d = torch.distributions.normal.Normal(
        torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device)
    )
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(
        logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpt-%04d.pth" % epoch)
    torch.save(state, filename)


def isnan(tensor):
    return tensor != tensor


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
