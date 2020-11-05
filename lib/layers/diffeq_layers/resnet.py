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

import torch.nn as nn

from . import basic
from . import container

NGROUPS = 16


class ResNet(container.SequentialDiffEq):
    def __init__(self, dim, intermediate_dim, n_resblocks, conv_block=None):
        super(ResNet, self).__init__()

        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d

        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.n_resblocks = n_resblocks

        layers = []
        layers.append(
            conv_block(dim, intermediate_dim, ksize=3, stride=1, padding=1, bias=False)
        )
        for _ in range(n_resblocks):
            layers.append(BasicBlock(intermediate_dim, conv_block))
        layers.append(nn.GroupNorm(NGROUPS, intermediate_dim, eps=1e-4))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv_block(intermediate_dim, dim, ksize=1, bias=False))

        super(ResNet, self).__init__(*layers)

    def __repr__(self):
        return "{name}({dim}, intermediate_dim={intermediate_dim}, n_resblocks={n_resblocks})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim, conv_block=None):
        super(BasicBlock, self).__init__()

        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d

        self.norm1 = nn.GroupNorm(NGROUPS, dim, eps=1e-4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_block(dim, dim, ksize=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(NGROUPS, dim, eps=1e-4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_block(dim, dim, ksize=3, stride=1, padding=1, bias=False)

    def forward(self, t, x):
        residual = x

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)

        out += residual

        return out
