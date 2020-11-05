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

import math

import torch
import torch.nn as nn

_DEFAULT_ALPHA = 1e-6


class ZeroMeanTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            x = x + 0.5
            if logpx is None:
                return x
            return x, logpx
        else:
            x = x - 0.5
            if logpx is None:
                return x
            return x, logpx


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=_DEFAULT_ALPHA, effective_shape=None):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.effective_shape = effective_shape

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return _sigmoid(x, logpx, self.alpha, self.effective_shape)
        else:
            return _logit(x, logpx, self.alpha, self.effective_shape)


class SigmoidTransform(nn.Module):
    """Reverse of LogitTransform."""

    def __init__(self, alpha=_DEFAULT_ALPHA, effective_shape=None):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.effective_shape = effective_shape

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return _logit(x, logpx, self.alpha, self.effective_shape)
        else:
            return _sigmoid(x, logpx, self.alpha, self.effective_shape)


def _logit(x, logpx=None, alpha=_DEFAULT_ALPHA, effective_shape=None):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if logpx is None:
        return y

    if self.effective_shape is None:
        return y, logpx - _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)
    return (
        y,
        logpx
        - _logdetgrad(x.view(x.size(0), -1)[:, :effective_shape], alpha)
            .view(x.size(0), -1)
            .sum(1, keepdim=True),
    )


def _sigmoid(y, logpy=None, alpha=_DEFAULT_ALPHA, effective_shape=None):
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    if logpy is None:
        return x
    if self.effective_shape is None:
        return x, logpy + _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)
    return (
        x,
        logpy
        + _logdetgrad(x.view(x.size(0), -1)[:, :effective_shape], alpha)
            .view(x.size(0), -1)
            .sum(1, keepdim=True),
    )


def _logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad
