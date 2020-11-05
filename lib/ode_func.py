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

###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
# Link: https://github.com/YuliaRubanova/latent_ode
###########################

import lib.utils as utils
import torch
import torch.nn as nn


#####################################################################################################


class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)


#####################################################################################################


class ODEFunc_w_Poisson(ODEFunc):
    def __init__(
            self,
            input_dim,
            latent_dim,
            ode_func_net,
            lambda_net,
            device=torch.device("cpu"),
    ):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc_w_Poisson, self).__init__(
            input_dim, latent_dim, ode_func_net, device
        )

        self.latent_ode = ODEFunc(
            input_dim=input_dim,
            latent_dim=latent_dim,
            ode_func_net=ode_func_net,
            device=device,
        )

        self.latent_dim = latent_dim
        self.lambda_net = lambda_net
        # The computation of poisson likelihood can become numerically unstable.
        # The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
        # Exponent of lambda can also take large values
        # So we divide lambda by the constant and then multiply the integral of lambda by the constant
        self.const_for_lambda = torch.Tensor([100.0]).to(device)

    def extract_poisson_rate(self, augmented, final_result=True):
        y, log_lambdas, int_lambda = None, None, None

        assert augmented.size(-1) == self.latent_dim + self.input_dim
        latent_lam_dim = self.latent_dim // 2

        if len(augmented.size()) == 3:
            int_lambda = augmented[:, :, -self.input_dim:]
            y_latent_lam = augmented[:, :, : -self.input_dim]

            log_lambdas = self.lambda_net(y_latent_lam[:, :, -latent_lam_dim:])
            y = y_latent_lam[:, :, :-latent_lam_dim]

        elif len(augmented.size()) == 4:
            int_lambda = augmented[:, :, :, -self.input_dim:]
            y_latent_lam = augmented[:, :, :, : -self.input_dim]

            log_lambdas = self.lambda_net(y_latent_lam[:, :, :, -latent_lam_dim:])
            y = y_latent_lam[:, :, :, :-latent_lam_dim]

        # Multiply the intergral over lambda by a constant
        # only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
        if final_result:
            int_lambda = int_lambda * self.const_for_lambda

        # Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
        assert y.size(-1) == latent_lam_dim

        return y, log_lambdas, int_lambda, y_latent_lam

    def get_ode_gradient_nn(self, t_local, augmented):
        y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(
            augmented, final_result=False
        )
        dydt_dldt = self.latent_ode(t_local, y_latent_lam)

        log_lam = log_lam - torch.log(self.const_for_lambda)
        return torch.cat((dydt_dldt, torch.exp(log_lam)), -1)
