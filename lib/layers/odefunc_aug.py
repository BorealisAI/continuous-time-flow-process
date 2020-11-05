# Copyright (c) 2019-present Royal Bank of Canada
# Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# This code is based on ffjord project which can be found at https://github.com/rtqichen/ffjordimport copy

import copy

import numpy as np
import torch
import torch.nn as nn

from . import diffeq_layers
from .odefunc import NONLINEARITIES, sample_gaussian_like, sample_rademacher_like
from .squeeze import squeeze, unsqueeze

__all__ = ["AugODEnet", "AugODEfunc"]


def divergence_bf_aug(dx, y, effective_dim, **unused_kwargs):
    """
    The function for computing the exact log determinant of jacobian for augmented ode

    Parameters
        dx: Output of the neural ODE function
        y: input to the neural ode function
        effective_dim (int): the first n dimension of the input being transformed
                             by normalizing flows to compute log determinant
    Returns:
        sum_diag: determin
    """
    sum_diag = 0.0
    assert effective_dim <= y.shape[1]
    for i in range(effective_dim):
        sum_diag += (
            torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0]
                .contiguous()[:, i]
                .contiguous()
        )
    return sum_diag.contiguous()


def divergence_approx_aug(f, y, effective_dim, e=None):
    """
    The function for estimating log determinant of jacobian
    for augmented ode using Hutchinson's Estimator

    Parameters
        f: Output of the neural ODE function
        y: input to the neural ode function
        effective_dim (int): the first n dimensions of the input being transformed
                             by normalizing flows to compute log determinant

    Returns:
        sum_diag: estimate log determinant of the df/dy
    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


class AugODEnet(nn.Module):
    """
    Class to make neural nets for use in augmented continuous normalizing flows
    Only consider one-dimensional data yet

    Parameters:
        hidden_dims (list): the hidden dimensions of the neural ODE function
        aug_dim (int): dimension along which the input is augmented
        effective_shape (int): the size of input to be transformed
        aug_mapping (int): True or False determine whether the augmented dimension will be
                    fed into a network
        aug_hidden_dims (list): list of hiddem dimensions for the network of the augmented input
    """

    def __init__(
            self,
            hidden_dims,
            input_shape,
            effective_shape,
            strides,
            conv,
            layer_type="concat",
            nonlinearity="softplus",
            num_squeeze=0,
            aug_dim=0,
            aug_mapping=True,
            aug_hidden_dims=None,
    ):

        super(AugODEnet, self).__init__()
        self.aug_mapping = aug_mapping
        self.num_squeeze = num_squeeze
        self.effective_shape = effective_shape
        if conv:
            raise NotImplementedError
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "concat_v2": diffeq_layers.ConcatLinear_v2,
                "concatsquash": diffeq_layers.ConcatSquashLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        if self.aug_mapping:
            aug_layers = []
            aug_activation_fns = []
            aug_hidden_shape = list(copy.copy(input_shape))
            aug_hidden_shape[aug_dim] = input_shape[aug_dim] - effective_shape
            if aug_hidden_dims is None:
                aug_hidden_dims = copy.copy(hidden_dims)

        for dim_out, stride in zip(hidden_dims + (effective_shape,), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError("Unsupported stride: {}".format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] // 2,
                    hidden_shape[2] // 2,
                )
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] * 2,
                    hidden_shape[2] * 2,
                )
        if self.aug_mapping:
            for dim_out, stride in zip(
                    aug_hidden_dims + (input_shape[aug_dim] - effective_shape,), strides
            ):
                if stride is None:
                    layer_kwargs = {}
                elif stride == 1:
                    layer_kwargs = {
                        "ksize": 3,
                        "stride": 1,
                        "padding": 1,
                        "transpose": False,
                    }
                elif stride == 2:
                    layer_kwargs = {
                        "ksize": 4,
                        "stride": 2,
                        "padding": 1,
                        "transpose": False,
                    }
                elif stride == -2:
                    layer_kwargs = {
                        "ksize": 4,
                        "stride": 2,
                        "padding": 1,
                        "transpose": True,
                    }
                else:
                    raise ValueError("Unsupported stride: {}".format(stride))

                layer = base_layer(aug_hidden_shape[0], dim_out, **layer_kwargs)
                aug_layers.append(layer)
                aug_activation_fns.append(NONLINEARITIES[nonlinearity])

                aug_hidden_shape = list(copy.copy(aug_hidden_shape))
                aug_hidden_shape[0] = dim_out
                if stride == 2:
                    aug_hidden_shape[1], aug_hidden_shape[2] = (
                        aug_hidden_shape[1] // 2,
                        aug_hidden_shape[2] // 2,
                    )
                elif stride == -2:
                    aug_hidden_shape[1], aug_hidden_shape[2] = (
                        aug_hidden_shape[1] * 2,
                        aug_hidden_shape[2] * 2,
                    )

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        if self.aug_mapping:
            self.aug_layers = nn.ModuleList(aug_layers)
            self.aug_activation_fns = nn.ModuleList(aug_activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        # squeeze
        aug = y[:, self.effective_shape:]
        # aug = y[:, self]
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)

        if self.aug_mapping:
            for l, layer in enumerate(self.aug_layers):
                aug = layer(t, aug)
                if l < len(self.aug_layers) - 1:
                    aug = self.aug_activation_fns[l](aug)
        else:
            aug = torch.zeros_like(aug)

        dx = torch.cat([dx, aug], dim=1)
        return dx


class AugODEfunc(nn.Module):
    """
    Wrapper to make neural nets for use in augmented continuous normalizing flows
    """

    def __init__(
            self,
            diffeq,
            divergence_fn="approximate",
            residual=False,
            rademacher=False,
            effective_shape=None,
    ):
        super(AugODEfunc, self).__init__()
        ## effective_dim is the effective dimension for likelihood estimation
        ## It's either an integer or a list of integers
        assert divergence_fn in ("brute_force", "approximate")

        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf_aug
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx_aug

        self.register_buffer("_num_evals", torch.tensor(0.0))
        assert effective_shape is not None
        self.effective_shape = effective_shape

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.zeros_like(y)
            if isinstance(self.effective_shape, int):
                sample_like = y[:, : self.effective_shape]
            else:
                sample_like = y
                for dim, size in enumerate(self.effective_shape):
                    sample_like = sample_like.narrow(dim + 1, 0, size)

            if self.rademacher:
                sample = sample_rademacher_like(sample_like)
            else:
                sample = sample_gaussian_like(sample_like)
            if isinstance(self.effective_shape, int):
                self._e[:, : self.effective_shape] = sample
            else:
                pad_size = []
                for idx in self.effective_shape:
                    pad_size.append(0)
                    pad_size.append(y.shape[-idx - 1] - self.effective_shape[-idx - 1])
                pad_size = tuple(pad_size)
                self._e = torch.functional.padding(sample, pad_size, mode="constant")
            ## pad zeros

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf_aug(dy, y, self.effective_shape).view(
                    batchsize, 1
                )
            else:
                divergence = self.divergence_fn(
                    dy, y, self.effective_shape, e=self._e
                ).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            if isinstance(self.effective_dim, int):
                divergence -= (
                        torch.ones_like(divergence)
                        * torch.tensor(
                    np.prod(y.shape[1:]) * self.effective_shape / y.shape[1],
                    dtype=torch.float32,
                ).to(divergence)
                )
            else:
                divergence -= (
                        torch.ones_like(divergence)
                        * torch.tensor(
                    np.prod(self.effective_shape),
                    dtype=torch.float32,
                ).to(divergence)
                )
        return tuple(
            [dy, -divergence]
            + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        )
