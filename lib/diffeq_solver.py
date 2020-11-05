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

import torch
import torch.nn as nn

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint


#####################################################################################################


class DiffeqSolver(nn.Module):
    def __init__(
            self,
            input_dim,
            ode_func,
            method,
            latents,
            odeint_rtol=1e-4,
            odeint_atol=1e-5,
            device=torch.device("cpu"),
    ):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards=False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]

        pred_y = odeint(
            self.ode_func,
            first_point,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        pred_y = pred_y.permute(1, 2, 0, 3)

        assert torch.mean(pred_y[:, :, 0, :] - first_point) < 0.001
        assert pred_y.size()[0] == n_traj_samples
        assert pred_y.size()[1] == n_traj

        return pred_y

    def sample_traj_from_prior(
            self, starting_point_enc, time_steps_to_predict, n_traj_samples=1
    ):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(
            func,
            starting_point_enc,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y
