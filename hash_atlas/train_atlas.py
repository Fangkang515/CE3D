from implicit_neural_networks import IMLP
import cv2
import torch
import torch.optim as optim
import numpy as np

from evaluate import evaluate_model
from datetime import datetime
from loss_utils import (
    get_gradient_loss,
    get_rigidity_loss,
    get_optical_flow_loss,
    get_optical_flow_alpha_loss,
)
from unwrap_utils import get_tuples, pre_train_mapping, load_input_data, save_mask_flow
import sys
import os

from torch.utils.tensorboard import SummaryWriter

import logging

# import json
import commentjson as json
import glob
from pathlib import Path
from time import time
import random
from raft_wrapper import RAFTWrapper
from tqdm import tqdm

global begin_time
begin_time = time()


try:
    import tinycudann as tcnn
except ImportError:
    print("This sample requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def backup_codes(path):
    # XXX TODO: model_zoos codes should be also backup...
    os.system(f"cp *.py {path}")


def get_ori_img_size(data_folder):
    paths = (
        glob.glob(f"{data_folder}/*.png")
        + glob.glob(f"{data_folder}/*.jpg")
        + glob.glob(f"{data_folder}/*.JPG")
    )
    frame = cv2.imread(paths[0])
    h, w, _ = frame.shape
    return np.int64(w), np.int64(h)


def getting_optical_flow(vid_frames_path):
    vid_path = Path(vid_frames_path)
    files = sorted(
        list(vid_path.glob("*.jpg"))
        + list(vid_path.glob("*.png"))
        + list(vid_path.glob("*.JPG"))
    )
    vid_root = vid_path.parent / "work_space" / "atlas_space"
    out_flow_dir = vid_root / "flow"
    out_flow_dir.mkdir(exist_ok=True)
    current_dir = os.path.dirname(__file__)
    raft_wrapper = RAFTWrapper(
        model_path=os.path.join(current_dir, "raft-things.pth"),
        max_long_edge=800,
    )
    for i, file1 in enumerate(tqdm(files, desc="computing flow")):
        if i < len(files) - 1:
            file2 = files[i + 1]
            fn1 = file1.name
            fn2 = file2.name
            out_flow12_fn = out_flow_dir / f"{fn1}_{fn2}.npy"
            out_flow21_fn = out_flow_dir / f"{fn2}_{fn1}.npy"

            overwrite = False
            if not out_flow12_fn.exists() and not out_flow21_fn.exists() or overwrite:
                im1, im2 = raft_wrapper.load_images(str(file1), str(file2))
                flow12 = raft_wrapper.compute_flow(im1, im2)
                flow21 = raft_wrapper.compute_flow(im2, im1)
                np.save(out_flow12_fn, flow12)
                np.save(out_flow21_fn, flow21)


def main(config):
    seed_everything(2023)
    if True:  # parse parameters
        maximum_number_of_frames = config["maximum_number_of_frames"]
        resx, resy = get_ori_img_size(config["data_folder"])
        iters_num = config["iters_num"]

        # batch size:
        samples = config["samples_batch"]

        # evaluation frequency (in terms of iterations number)
        evaluate_every = np.int64(config["evaluate_every"])

        # optionally it is possible to load a checkpoint
        load_checkpoint = config[
            "load_checkpoint"
        ]  # set to true to continue from a checkpoint
        checkpoint_path = config["checkpoint_path"]

        # a data folder that contains images
        data_folder = Path(config["data_folder"])
        add_to_experiment_folder_name = config["add_to_experiment_folder_name"]

        # boolean variables for determining if a pretraining is used:
        pretrain_mapping1 = config["pretrain_mapping1"]
        pretrain_mapping2 = config["pretrain_mapping2"]
        pretrain_iter_number = config["pretrain_iter_number"]

        # the scale of the atlas uv coordinates relative to frame's xy coordinates  default=0.8
        uv_mapping_scale = config["uv_mapping_scale"]

        # bootstrapping configuration:
        alpha_bootstrapping_factor = config[
            "alpha_bootstrapping_factor"
        ]  # default=2000
        stop_bootstrapping_iteration = config[
            "stop_bootstrapping_iteration"
        ]  # default=10000

        # coefficients for the different loss terms
        rgb_coeff = config["rgb_coeff"]  # coefficient for rgb loss term:
        alpha_flow_factor = config["alpha_flow_factor"]
        sparsity_coeff = config["sparsity_coeff"]
        # optical flow loss term coefficient (beta_f in the paper):
        optical_flow_coeff = config["optical_flow_coeff"]
        use_gradient_loss = config["use_gradient_loss"]
        gradient_loss_coeff = config["gradient_loss_coeff"]
        rigidity_coeff = config[
            "rigidity_coeff"
        ]  # coefficient for the rigidity loss term
        derivative_amount = config[
            "derivative_amount"
        ]  # For finite differences gradient computation:
        # for using global (in addition to the current local) rigidity loss:
        include_global_rigidity_loss = config["include_global_rigidity_loss"]
        # Finite differences parameters for the global rigidity terms:
        global_rigidity_derivative_amount_fg = config[
            "global_rigidity_derivative_amount_fg"
        ]
        global_rigidity_derivative_amount_bg = config[
            "global_rigidity_derivative_amount_bg"
        ]
        global_rigidity_coeff_fg = config["global_rigidity_coeff_fg"]
        global_rigidity_coeff_bg = config["global_rigidity_coeff_bg"]
        stop_global_rigidity = config["stop_global_rigidity"]
        use_propainter_loss = config["use_propainter_loss"]
        propainter_coeff = config["propainter_coeff"]

    use_optical_flow = True

    getting_optical_flow(data_folder)

    vid_root = data_folder.parent / "work_space" / "atlas_space"

    results_folder = vid_root / f"training_results{add_to_experiment_folder_name}"

    results_folder.mkdir(parents=True, exist_ok=True)
    code_folder = results_folder / "codes"
    code_folder.mkdir(parents=True, exist_ok=True)
    backup_codes(str(code_folder))
    with open("%s/atlas_config.json" % results_folder, "w") as json_file:
        json.dump(config, json_file, indent=4)
    logging.basicConfig(
        filename="%s/log.log" % results_folder,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info("Started")
    writer = SummaryWriter(log_dir=str(results_folder))
    (
        optical_flows_mask,
        video_frames,
        propainter_frames,
        optical_flows_reverse_mask,
        mask_frames,
        video_frames_dx,
        video_frames_dy,
        optical_flows_reverse,
        optical_flows,
    ) = load_input_data(
        resy, resx, maximum_number_of_frames, data_folder, True, True, vid_root
    )
    number_of_frames = video_frames.shape[3]
    # save a video showing the masked part of the forward optical flow:s
    save_mask_flow(optical_flows_mask, video_frames, results_folder)

    config_hash = {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Tanh",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    }

    model_F_mapping = IMLP(  # foreground ori_img coordinates[x,y,t] --> [u,v]
        input_dim=3,
        output_dim=5,  # [u1, v1, u2, v2, alpha]
        hidden_dim=256,
        use_positional=False,
        num_layers=8,
        skip_layers=[3, 6],
    ).to(device)

    model_F_atlas = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=3,
        encoding_config=config_hash["encoding"],
        network_config=config_hash["network"],
    ).to(device)

    optimizer_all = optim.AdamW(
        [
            {"params": list(model_F_mapping.parameters()), "lr": 0.001},
            {"params": list(model_F_atlas.parameters()), "lr": 0.01},
        ]
    )
    final_lr = 1e-4  # final_lr=init_lr * (gamma**iters_num)  --> gamma = (final_lr / init_lr) ** (1/iters_num)
    gamma = (final_lr / 1e-3) ** (1 / iters_num)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=gamma)

    # number_of_frames_backup = number_of_frames
    larger_dim = np.maximum(resx, resy)
    start_iteration = 0
    if not load_checkpoint:  # pretrain
        if pretrain_mapping1:
            model_F_mapping = pre_train_mapping(
                model_F_mapping,
                number_of_frames,
                uv_mapping_scale,
                resx=resx,
                resy=resy,
                larger_dim=larger_dim,
                device=device,
                pretrain_iters=pretrain_iter_number,
            )
    else:
        init_file = torch.load(checkpoint_path)
        model_F_atlas.load_state_dict(init_file["F_atlas_state_dict"])
        model_F_mapping.load_state_dict(init_file["model_F_mapping_state_dict"])
        optimizer_all.load_state_dict(init_file["optimizer_all_state_dict"])
        start_iteration = init_file["iteration"]

    # The purpose of this function is to extract the positional information of non-zero pixels from video frame data. The shape is [3, non-zero h * w * f], where 3 represents [column coordinates, row coordinates, and frame index]
    jif_all = get_tuples(number_of_frames, video_frames)

    # Start training!
    print_every = 100
    start_time = time()
    for i in range(start_iteration, iters_num):
        if i > stop_bootstrapping_iteration:
            alpha_bootstrapping_factor = 0
        if i > stop_global_rigidity:
            global_rigidity_coeff_fg = 0
            global_rigidity_coeff_bg = 0

        # randomly choose indices for the current batch
        inds_foreground = torch.randint(jif_all.shape[1], (np.int64(samples), 1))
        jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)
        rgb_GT = (
            video_frames[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]]
            .squeeze(1)
            .to(device)
        )
        rgb_propainter_background_GT = (
            propainter_frames[
                jif_current[1, :], jif_current[0, :], :, jif_current[2, :]
            ]
            .squeeze(1)
            .to(device)
        )
        # the correct alpha according to the precomputed mask
        alpha_mask = (
            mask_frames[jif_current[1, :], jif_current[0, :], jif_current[2, :]]
            .squeeze(1)
            .to(device)
            .unsqueeze(-1)
        )

        # normalize coordinates to be in [-1,1]
        xyt_current = torch.cat(
            (
                jif_current[0, :] / larger_dim * 2 - 1,
                jif_current[1, :] / larger_dim * 2 - 1,
                jif_current[2, :] / number_of_frames * 2.0 - 1,
            ),
            dim=1,
        ).to(
            device
        )  # size (batch, 3)

        # get the atlas UV coordinates from the mapping networks;
        mapping_outs = model_F_mapping(xyt_current)
        uv_foreground1 = mapping_outs[..., :2]
        uv_foreground2 = mapping_outs[..., 2:4]
        alpha = mapping_outs[..., -1:]
        # map tanh output of the alpha network to the range (0,1) :
        alpha = 0.5 * (alpha + 1.0)
        if True or i <= stop_bootstrapping_iteration:
            alpha = alpha * 0.99
            alpha = alpha + 0.001

        # The input of hash grid should be in range [0, 1] Tom94:tiny-cuda-nn's encodings all expect inputs within [0,1]^n by convention
        # Therefore, Foreground colors are sampled from [0.5,1] coordinates and background colors are sampled from [0,0.5] coordinates
        # if concat uv_foreground1 and uv_foreground2, then fed into model_F_atlas, the speed will solwer, dont konw why...
        rgb_output1 = (
            model_F_atlas(uv_foreground1 * 0.25 + 0.75) + 1.0
        ) * 0.5  # [-1,1]*0.25+0.75 = [0.5,1]; ([0,1]+1)*0.5=[0.5,1]
        rgb_output2 = (
            model_F_atlas(uv_foreground2 * 0.25 + 0.25) + 1.0
        ) * 0.5  # [-1,1]*0.5+0.25 = [0,0.5];([-1,0]+1)*0.5=[0,0.5]

        rgb_output_pred = rgb_output1 * alpha + rgb_output2 * (1.0 - alpha)
        # =========================  RGB and its sparsity loss ========================= #
        rgb_loss = (torch.norm(rgb_output_pred - rgb_GT, dim=1) ** 2).mean()
        if use_propainter_loss:
            rgb_loss_propainter = (
                torch.norm(rgb_output2 - rgb_propainter_background_GT, dim=1) ** 2
            ).mean()
        else:
            rgb_loss_propainter = torch.tensor(0.0)

        #  without the sparsity, There will be ghosting in the foreground (multiple foreground targets appear on Atlas).
        rgb_output_pred_not = rgb_output1 * (1.0 - alpha)
        rgb_loss_sparsity = (torch.norm(rgb_output_pred_not, dim=1) ** 2).mean()
        rgb_loss_sparsity = rgb_loss_sparsity  # + 0.01 * (torch.norm(rgb_output_background_not, dim=1) ** 2).mean()

        # =========================  RGB gradient loss ========================= #
        if (
            use_gradient_loss
        ):  # XXX TODO default not use, since the trade-off between of performance and speed
            gradient_loss = get_gradient_loss(
                video_frames,
                video_frames_dx,
                video_frames_dy,
                jif_current,
                model_F_mapping,
                model_F_atlas,
                rgb_output_pred,
                device,
                resx,
                number_of_frames,
            )
        else:
            gradient_loss, rgb_loss2 = torch.tensor(0.0), torch.tensor(0.0)

        # =========================  rigidity motion loss ========================= #
        rigidity_loss1 = get_rigidity_loss(
            jif_current,
            derivative_amount,
            larger_dim,
            number_of_frames,
            model_F_mapping,
            uv_foreground1,
            device,
            uv_mapping_scale=uv_mapping_scale,
            is_foreground=True,
        )
        rigidity_loss2 = get_rigidity_loss(
            jif_current,
            derivative_amount,
            larger_dim,
            number_of_frames,
            model_F_mapping,
            uv_foreground2,
            device,
            uv_mapping_scale=uv_mapping_scale,
            is_foreground=False,
        )
        if include_global_rigidity_loss and i <= stop_global_rigidity:
            global_rigidity_loss1 = get_rigidity_loss(
                jif_current,
                global_rigidity_derivative_amount_fg,
                larger_dim,
                number_of_frames,
                model_F_mapping,
                uv_foreground1,
                device,
                uv_mapping_scale=uv_mapping_scale,
                is_foreground=True,
            )
            global_rigidity_loss2 = get_rigidity_loss(
                jif_current,
                global_rigidity_derivative_amount_bg,
                larger_dim,
                number_of_frames,
                model_F_mapping,
                uv_foreground2,
                device,
                uv_mapping_scale=uv_mapping_scale,
                is_foreground=False,
            )
        else:
            global_rigidity_loss1, global_rigidity_loss2 = torch.tensor(
                0.0
            ), torch.tensor(0.0)

        # ========================= optical flow loss between different views ========================= #
        # The flow loss encourage same point map into same UV points, which ensures coherence between perspectives (i.e. no distortion or sudden changes)
        flow_loss1 = get_optical_flow_loss(
            jif_current,
            uv_foreground1,
            optical_flows_reverse,
            optical_flows_reverse_mask,
            larger_dim,
            number_of_frames,
            model_F_mapping,
            optical_flows,
            optical_flows_mask,
            uv_mapping_scale,
            device,
            use_alpha=True,
            alpha=alpha,
            is_foreground=True,
        )
        flow_loss2 = get_optical_flow_loss(
            jif_current,
            uv_foreground2,
            optical_flows_reverse,
            optical_flows_reverse_mask,
            larger_dim,
            number_of_frames,
            model_F_mapping,
            optical_flows,
            optical_flows_mask,
            uv_mapping_scale,
            device,
            use_alpha=True,
            alpha=1 - alpha,
            is_foreground=False,
        )
        flow_alpha_loss = get_optical_flow_alpha_loss(
            model_F_mapping,
            jif_current,
            alpha,
            optical_flows_reverse,
            optical_flows_reverse_mask,
            larger_dim,
            number_of_frames,
            optical_flows,
            optical_flows_mask,
            device,
        )
        # using mask for initial GT of alpha
        if i > stop_bootstrapping_iteration:
            alpha_bootstrapping_loss = torch.tensor(0.0)
        else:
            alpha_bootstrapping_loss = torch.mean(
                -alpha_mask * torch.log(alpha) - (1 - alpha_mask) * torch.log(1 - alpha)
            )

        loss = (
            rigidity_coeff * (rigidity_loss1 + rigidity_loss2)
            + global_rigidity_coeff_fg * global_rigidity_loss1
            + global_rigidity_coeff_bg * global_rigidity_loss2
            + rgb_loss * rgb_coeff
            + rgb_loss_propainter * propainter_coeff
            + optical_flow_coeff * (flow_loss1 + flow_loss2)
            + alpha_bootstrapping_loss * alpha_bootstrapping_factor
            + flow_alpha_loss * alpha_flow_factor
            + rgb_loss_sparsity * sparsity_coeff
            + gradient_loss * gradient_loss_coeff
        )  # + rgb_loss2 * gradient_loss_coeff

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()
        scheduler.step()
        lr_mapping_network = optimizer_all.param_groups[0]["lr"]
        lr_atlas_network = optimizer_all.param_groups[1]["lr"]

        try:
            if use_optical_flow and i % print_every == 0:
                print(
                    "of_loss1:%.2f" % flow_loss1.detach(),
                    "of_loss2:%.2f" % flow_loss2.detach(),
                )
                logging.info("of_loss1:%f" % flow_loss1.detach())
                logging.info("of_loss2:%f" % flow_loss2.detach())
                writer.add_scalar("Loss/train_of1", flow_loss1.detach(), i)
                writer.add_scalar("Loss/train_of2", flow_loss2.detach(), i)
        except:
            pass

        if i % print_every == 0:
            using_time = time() - start_time
            print(
                f"Iteration {i}  using time: {using_time:.2f}  lr_mapping: {lr_mapping_network:.5f}  lr_atlas: {lr_atlas_network:.5f}"
            )
            logging.info("Iteration %d" % i)
            print("gradient_loss: %.3f" % gradient_loss.cpu().item())
            print(
                "flow alpha loss: %.3f" % flow_alpha_loss.cpu().item(),
                "alpha_balancing_loss %.3f" % alpha_bootstrapping_loss.cpu().item(),
            )
            print(
                "rgb_loss:%.3f" % rgb_loss.detach(),
                "rgb_loss_negative %.3f" % rgb_loss_sparsity.detach(),
            )
            print(
                "rigidity_loss1:%.3f" % rigidity_loss1.detach(),
                "rigidity_loss2:%.3f" % rigidity_loss2.detach(),
            )
            print("total_loss:%.3f" % loss.detach())
            print(
                "alpha_mean:%.3f" % alpha.mean().detach(),
                "Big0.5_mean:%.3f" % alpha[alpha > 0.5].mean().detach(),
                "Low0.5_mean:%.3f" % alpha[alpha < 0.5].mean().detach(),
            )
            print(f"------------{results_folder}------------------")
            logging.info("time: %f s", using_time)
            logging.info("flow_alpha_loss: %f", flow_alpha_loss.detach())
            logging.info("rgb_loss:%f" % rgb_loss.detach())
            logging.info("total_loss:%f" % loss.detach())
            logging.info("rigidity_loss1:%f" % rigidity_loss1.detach())
            logging.info("rigidity_loss2:%f" % rigidity_loss2.detach())
            logging.info("rgb_loss_negative %f" % rgb_loss_sparsity.detach())
            logging.info("-------------------------------")
            logging.info("alpha_mean:%f" % alpha.mean().detach())
            logging.info("alphaBig0_mean:%f" % alpha[alpha > 0.5].mean().detach())
            logging.info("alphaLow0_mean:%f" % alpha[alpha < 0.5].mean().detach())
            writer.add_scalar("Loss/alpha_mean", alpha.mean().detach(), i)
            writer.add_scalar("Loss/rgb_loss", rgb_loss.detach(), i)
            writer.add_scalar("Loss/rigidity_loss1", rigidity_loss1.detach(), i)
            writer.add_scalar("Loss/rigidity_loss2", rigidity_loss2.detach(), i)

        if (
            (i % evaluate_every == 0 and i > start_iteration)
            or (i == iters_num - 1)
            or (pretrain_mapping1 and pretrain_mapping2 and i == 1000)
        ):
            psnrs, ssim, lpips = evaluate_model(
                model_F_atlas,
                resx,
                resy,
                number_of_frames,
                model_F_mapping,
                video_frames,
                results_folder,
                i,
                mask_frames,
                optimizer_all,
                writer,
                derivative_amount,
                uv_mapping_scale,
                optical_flows,
                optical_flows_mask,
                data_folder,
                device,
            )
            metric_path = str(results_folder).split("training_results")[0]
            # scene_name = str(results_folder).split("/")[6]
            # print(metric_path, scene_name)
            using_time = time() - begin_time
            with open(
                f"{metric_path}/metric.txt", "a"
            ) as file:
                file.write(
                    f"i:{i} psnr:{psnrs.mean():.2f} ssim:{ssim.mean():.2f} lpips:{lpips.mean():.2f} t:{using_time/60:.2f}min\n"
                )
                if i == iters_num - 1:
                    file.write("\n\n")

            rgb_img = video_frames[:, :, :, 0].numpy()
            writer.add_image("Input/rgb_0", rgb_img, i, dataformats="HWC")
            model_F_atlas.train()
            model_F_mapping.train()


config_atlas = {
    "data_folder": "Your/dataset_zoos/llff/qq11/images_2",
    "maximum_number_of_frames": 200,
    "iters_num": 60000,
    "evaluate_every": 20000,
    "samples_batch": 10000,
    "optical_flow_coeff": 5.0,
    "rgb_coeff": 5000,
    "rigidity_coeff": 1.0,
    "uv_mapping_scale": 0.9,
    "pretrain_mapping1": True,
    "pretrain_mapping2": True,
    "alpha_bootstrapping_factor": 2000.0,
    "stop_bootstrapping_iteration": 30000,
    "alpha_flow_factor": 49.0,
    "positional_encoding_num_alpha": 5,
    "gradient_loss_coeff": 1000.0,
    "use_gradient_loss": True,
    "sparsity_coeff": 1000.0,
    "pretrain_iter_number": 1000,
    "derivative_amount": 1,
    "include_global_rigidity_loss": True,
    "global_rigidity_derivative_amount_fg": 100,
    "global_rigidity_derivative_amount_bg": 100,
    "global_rigidity_coeff_fg": 5.0,
    "global_rigidity_coeff_bg": 50.0,
    "stop_global_rigidity": 5000,
    "use_propainter_loss": True,
    "propainter_coeff": 200,
    "load_checkpoint": False,
    "checkpoint_path": "",
    "add_to_experiment_folder_name": "",
}


def train_atlas(data_folder):
    config_atlas["data_folder"] = data_folder
    main(config_atlas)


"""
if __name__ == "__main__":

    begin_time = time()
    data_folder = (
        f"../datasets/flower_test"
    )
    assert os.path.isdir(data_folder)
    config_atlas["data_folder"] = data_folder
    main(config_atlas)
"""
