# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time

import lib.utils as utils
import numpy as np
import torch

from bm_sequential import get_test_dataset as get_dataset
from ctfp_tools import build_augmented_model_tabular
from ctfp_tools import parse_arguments
from ctfp_tools import run_ctfp_model as run_model
from train_misc import (
    create_regularization_fns,
)
from train_misc import set_cnf_options

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    args = parse_arguments()
    logger = utils.get_logger(
        logpath=os.path.join(args.save, "logs_test"), filepath=os.path.abspath(__file__)
    )

    if args.layer_type == "blend":
        logger.info(
            "!! Setting time_length from None to 1.0 due to use of Blend layers."
        )
        args.time_length = 1.0

    logger.info(args)
    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.use_cpu:
        device = torch.device("cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    test_loader = get_dataset(args, args.test_batch_size)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    aug_model = build_augmented_model_tabular(
        args,
        args.aug_size + args.effective_shape,
        regularization_fns=regularization_fns,
    )
    set_cnf_options(args, aug_model)
    logger.info(aug_model)

    # restore parameters
    itr = 0
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        aug_model.load_state_dict(checkpt["state_dict"])

    if torch.cuda.is_available() and not args.use_cpu:
        aug_model = torch.nn.DataParallel(aug_model).cuda()

    best_loss = float("inf")
    aug_model.eval()
    with torch.no_grad():
        logger.info("Testing...")
        losses = []
        num_observes = []
        for _, x in enumerate(test_loader):
            ## x is a tuple of (values, times, stdv, masks)
            start = time.time()
            # cast data and move to device
            x = map(cvt, x)
            values, times, vars, masks = x
            loss = run_model(args, aug_model, values, times, vars, masks)
            losses.append(loss.data.cpu().numpy())
            num_observes.append(torch.sum(masks).data.cpu().numpy())
        loss = np.sum(np.array(losses) * np.array(num_observes)) / np.sum(num_observes)
        logger.info("Bit/dim {:.4f}".format(loss))
