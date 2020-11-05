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

import torch
import torch.nn as nn

from .wrappers import diffeq_wrapper


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers."""

    def __init__(self, *layers):
        super(SequentialDiffEq, self).__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


class MixtureODELayer(nn.Module):
    """Produces a mixture of experts where output = sigma(t) * f(t, x).
    Time-dependent weights sigma(t) help learn to blend the experts without resorting to a highly stiff f.
    Supports both regular and diffeq experts.
    """

    def __init__(self, experts):
        super(MixtureODELayer, self).__init__()
        assert len(experts) > 1
        wrapped_experts = [diffeq_wrapper(ex) for ex in experts]
        self.experts = nn.ModuleList(wrapped_experts)
        self.mixture_weights = nn.Linear(1, len(self.experts))

    def forward(self, t, y):
        dys = []
        for f in self.experts:
            dys.append(f(t, y))
        dys = torch.stack(dys, 0)
        weights = self.mixture_weights(t).view(-1, *([1] * (dys.ndimension() - 1)))

        dy = torch.sum(dys * weights, dim=0, keepdim=False)
        return dy
