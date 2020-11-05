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
from torch.autograd import grad


class PlanarFlow(nn.Module):
    def __init__(self, nd=1):
        super(PlanarFlow, self).__init__()
        self.nd = nd
        self.activation = torch.tanh

        self.register_parameter("u", nn.Parameter(torch.randn(self.nd)))
        self.register_parameter("w", nn.Parameter(torch.randn(self.nd)))
        self.register_parameter("b", nn.Parameter(torch.randn(1)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nd)
        self.u.data.uniform_(-stdv, stdv)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.fill_(0)
        self.make_invertible()

    def make_invertible(self):
        u = self.u.data
        w = self.w.data
        dot = torch.dot(u, w)
        m = -1 + math.log(1 + math.exp(dot))
        du = (m - dot) / torch.norm(w) * w
        u = u + du
        self.u.data = u

    def forward(self, z, logp=None, reverse=False):
        """Computes f(z) and log q(f(z))"""

        assert not reverse, "Planar normalizing flow cannot be reversed."

        logp - torch.log(self._detgrad(z) + 1e-8)
        h = self.activation(torch.mm(z, self.w.view(self.nd, 1)) + self.b)
        z = z + self.u.expand_as(z) * h

        f = self.sample(z)
        if logp is not None:
            qf = self.log_density(z, logp)
            return f, qf
        else:
            return f

    def sample(self, z):
        """Computes f(z)"""
        h = self.activation(torch.mm(z, self.w.view(self.nd, 1)) + self.b)
        output = z + self.u.expand_as(z) * h
        return output

    def _detgrad(self, z):
        """Computes |det df/dz|"""
        with torch.enable_grad():
            z = z.requires_grad_(True)
            h = self.activation(torch.mm(z, self.w.view(self.nd, 1)) + self.b)
            psi = grad(
                h,
                z,
                grad_outputs=torch.ones_like(h),
                create_graph=True,
                only_inputs=True,
            )[0]
        u_dot_psi = torch.mm(psi, self.u.view(self.nd, 1))
        detgrad = 1 + u_dot_psi
        return detgrad

    def log_density(self, z, logqz):
        """Computes log density of the flow given the log density of z"""
        return logqz - torch.log(self._detgrad(z) + 1e-8)
