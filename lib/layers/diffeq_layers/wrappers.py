# MIT License
#
# Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl
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
# Link: https://github.com/rtqichen/ffjord

from inspect import signature

import torch.nn as nn

__all__ = ["diffeq_wrapper", "reshape_wrapper"]


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module
        if len(signature(self.module.forward).parameters) == 1:
            self.diffeq = lambda t, y: self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            self.diffeq = self.module
        else:
            raise ValueError(
                "Differential equation needs to either take (t, y) or (y,) as input."
            )

    def forward(self, t, y):
        return self.diffeq(t, y)

    def __repr__(self):
        return self.diffeq.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class ReshapeDiffEq(nn.Module):
    def __init__(self, input_shape, net):
        super(ReshapeDiffEq, self).__init__()
        assert (
                len(signature(net.forward).parameters) == 2
        ), "use diffeq_wrapper before reshape_wrapper."
        self.input_shape = input_shape
        self.net = net

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


def reshape_wrapper(input_shape, layer):
    return ReshapeDiffEq(input_shape, layer)
