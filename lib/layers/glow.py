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
import torch.nn.functional as F


class BruteForceLayer(nn.Module):
    def __init__(self, dim):
        super(BruteForceLayer, self).__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, x, logpx=None, reverse=False):

        if not reverse:
            y = F.linear(x, self.weight)
            if logpx is None:
                return y
            else:
                return y, logpx - self._logdetgrad.expand_as(logpx)

        else:
            y = F.linear(x, self.weight.double().inverse().float())
            if logpx is None:
                return y
            else:
                return y, logpx + self._logdetgrad.expand_as(logpx)

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight.double()))).float()
