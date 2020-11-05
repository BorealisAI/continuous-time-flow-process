# Copyright (c) 2019-present Royal Bank of Canada
# Copyright (c) 2019 Yulia Rubanova
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import lib.utils as utils
# This code is based on latent ODE project which can be found at: https://github.com/YuliaRubanova/latent_ode copy
import torch.nn as nn
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Encoder_z0_ODE_RNN
from lib.ode_func import ODEFunc


def create_ode_rnn_encoder(args, device):
    """
    This function create the ode-rnn model as an encoder
    args: the arguments from parse_arguments in ctfp_tools
    device: cpu or gpu to run the model
    return an ode-rnn model 
    """
    enc_input_dim = args.input_size * 2  ## concatenate the mask with input

    ode_func_net = utils.create_net(
        args.rec_size,
        args.rec_size,
        n_layers=args.rec_layers,
        n_units=args.units,
        nonlinear=nn.Tanh,
    )

    rec_ode_func = ODEFunc(
        input_dim=enc_input_dim,
        latent_dim=args.rec_size,
        ode_func_net=ode_func_net,
        device=device,
    ).to(device)

    z0_diffeq_solver = DiffeqSolver(
        enc_input_dim,
        rec_ode_func,
        "euler",
        args.latent_size,
        odeint_rtol=1e-3,
        odeint_atol=1e-4,
        device=device,
    )

    encoder_z0 = Encoder_z0_ODE_RNN(
        args.rec_size,
        enc_input_dim,
        z0_diffeq_solver,
        z0_dim=args.latent_size,
        n_gru_units=args.gru_units,
        device=device,
    ).to(device)
    return encoder_z0
