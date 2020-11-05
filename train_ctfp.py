# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import os.path as osp
import time

import lib.utils as utils
import numpy as np
import torch
from lib.utils import optimizer_factory

from bm_sequential import get_dataset
from ctfp_tools import build_augmented_model_tabular
from ctfp_tools import run_ctfp_model as run_model, parse_arguments
from train_misc import (
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
)
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time

RUNNINGAVE_PARAM = 0.7
torch.backends.cudnn.benchmark = True


def save_model(args, aug_model, optimizer, epoch, itr, save_path):
    """
    save CTFP model's checkpoint during training

    Parameters:
        args: the arguments from parse_arguments in ctfp_tools
        aug_model: the CTFP Model
        optimizer: optimizer of CTFP model
        epoch: training epoch
        itr: training iteration
        save_path: path to save the model
    """
    torch.save(
        {
            "args": args,
            "state_dict": aug_model.module.state_dict()
            if torch.cuda.is_available() and not args.use_cpu
            else aug_model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "last_epoch": epoch,
            "iter": itr,
        },
        save_path,
    )


if __name__ == "__main__":
    args = parse_arguments()
    # logger
    utils.makedirs(args.save)
    logger = utils.get_logger(
        logpath=os.path.join(args.save, "logs"), filepath=os.path.abspath(__file__)
    )

    if args.layer_type == "blend":
        logger.info(
            "!! Setting time_length from None to 1.0 due to use of Blend layers."
        )
        args.time_length = 1.0
    logger.info(args)
    if not args.no_tb_log:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(osp.join(args.save, "tb_logs"))
        writer.add_text("args", str(args))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get deivce
    if args.use_cpu:
        device = torch.device("cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_loader, val_loader = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)

    aug_model = build_augmented_model_tabular(
        args,
        args.aug_size + args.effective_shape,
        regularization_fns=regularization_fns,
    )

    set_cnf_options(args, aug_model)
    logger.info(aug_model)

    logger.info(
        "Number of trainable parameters: {}".format(count_parameters(aug_model))
    )

    # optimizer
    parameter_list = list(aug_model.parameters())
    optimizer, num_params = optimizer_factory(args, parameter_list)
    print("Num of Parameters: %d" % num_params)

    # restore parameters
    itr = 0
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        aug_model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
        if "iter" in checkpt.keys():
            itr = checkpt["iter"]
        if "last_epoch" in checkpt.keys():
            args.begin_epoch = checkpt["last_epoch"] + 1

    if torch.cuda.is_available() and not args.use_cpu:
        aug_model = torch.nn.DataParallel(aug_model).cuda()

    # For visualization.

    time_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    loss_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    steps_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    grad_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    tt_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)

    best_loss = float("inf")
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        aug_model.train()
        for temp_idx, x in enumerate(train_loader):
            ## x is a tuple of (values, times, stdv, masks)
            start = time.time()
            optimizer.zero_grad()

            # cast data and move to device
            x = map(cvt, x)
            values, times, vars, masks = x
            # compute loss
            loss = run_model(args, aug_model, values, times, vars, masks)

            total_time = count_total_time(aug_model)
            ## Assume the base distribution be Brownian motion

            if regularization_coeffs:
                reg_states = get_regularization(aug_model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff
                    for reg_state, coeff in zip(reg_states, regularization_coeffs)
                    if coeff != 0
                )
                loss = loss + reg_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                aug_model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(aug_model))
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)

            if not args.no_tb_log:
                writer.add_scalar("train/NLL", loss.cpu().data.item(), itr)

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time "
                    "{:.2f}({:.2f})".format(
                        itr,
                        time_meter.val,
                        time_meter.avg,
                        loss_meter.val,
                        loss_meter.avg,
                        steps_meter.val,
                        steps_meter.avg,
                        grad_meter.val,
                        grad_meter.avg,
                        tt_meter.val,
                        tt_meter.avg,
                    )
                )
                if regularization_coeffs:
                    log_message = append_regularization_to_log(
                        log_message, regularization_fns, reg_states
                    )
                logger.info(log_message)

            itr += 1

        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                num_observes = []
                aug_model.eval()
                for temp_idx, x in enumerate(val_loader):
                    ## x is a tuple of (values, times, stdv, masks)
                    start = time.time()

                    # cast data and move to device
                    x = map(cvt, x)
                    values, times, vars, masks = x
                    loss = run_model(args, aug_model, values, times, vars, masks)
                    # compute loss
                    losses.append(loss.data.cpu().numpy())
                    num_observes.append(torch.sum(masks).data.cpu().numpy())

                loss = np.sum(np.array(losses) * np.array(num_observes)) / np.sum(
                    num_observes
                )
                if not args.no_tb_log:
                    writer.add_scalar("val/NLL", loss, epoch)
                logger.info(
                    "Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(
                        epoch, time.time() - start, loss
                    )
                )

                save_model(
                    args,
                    aug_model,
                    optimizer,
                    epoch,
                    itr,
                    os.path.join(args.save, "checkpt_last.pth"),
                )
                save_model(
                    args,
                    aug_model,
                    optimizer,
                    epoch,
                    itr,
                    os.path.join(args.save, "checkpt_%d.pth") % (epoch),
                )

                if loss < best_loss:
                    best_loss = loss
                    save_model(
                        args,
                        aug_model,
                        optimizer,
                        epoch,
                        itr,
                        os.path.join(args.save, "checkpt_best.pth"),
                    )
