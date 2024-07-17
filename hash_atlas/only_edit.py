"""
when given edited atlases, to reconstruct the frames
"""
from implicit_neural_networks import IMLP
from IPython import embed
import matplotlib.image as mpimg
import glob
import time
from tqdm import tqdm
from scipy.interpolate import griddata
import torch
import numpy as np
import sys
import imageio
import cv2
from PIL import Image
import argparse
from evaluate import get_high_res_texture, get_colors, get_mapping_area
import os
import json

from pathlib import Path
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

ReconstructResolution = 512


def PIL_rgba2rgb(img):
    if img.shape[-1] == 4:
        img = np.array(img).astype(np.float32)
        img = img[..., :3] * (img[..., 3:] / 255)
        img = img.astype(np.uint8)
        img = imageio.core.util.Array(img)
    return img


def apply_edit(model_F_atlas, resx, resy, number_of_frames, model_F_mapping,
               video_frames,
               output_folder_final, mask_frames,
               evaluate_all_pixels=False, texture_edit_im1=None, texture_edit_im2=None,
               alpha_im1=None, alpha_im2=None, input_files='', frontend_save_name='./', mask_of_new_obj=None):
    imageio.imwrite(
        "%s/user_provided_edited_atlas_foreground.png" % output_folder_final, (np.concatenate((texture_edit_im1, alpha_im1[:, :, np.newaxis]), axis=2)*255).astype(np.uint8))
    imageio.imwrite(
        "%s/user_provided_edited_atlas_background.png" % output_folder_final, (np.concatenate((texture_edit_im2, alpha_im2[:, :, np.newaxis]), axis=2)*255).astype(np.uint8))

    larger_dim = np.maximum(resx, resy)

    # get relevant working crops from the atlases for atlas discretization
    # XXX TODO
    minx = 0
    miny = 0
    edge_size = 1
    _, texture_orig1 = get_high_res_texture(ReconstructResolution, 0, 1, 0, 1, model_F_atlas, device)
    maxx2, minx2, maxy2, miny2, edge_size2 = get_mapping_area(model_F_mapping, mask_frames <= 0.5, larger_dim,
                                                              number_of_frames,
                                                              torch.tensor([-0.5, -0.5]), device, invert_alpha=True, is_foreground=False)
    _, texture_orig2 = get_high_res_texture(ReconstructResolution, minx2, minx2 + edge_size2, miny2, miny2 + edge_size2, model_F_atlas, device)

    edited_tex1_only_edit = torch.from_numpy(texture_edit_im1)
    edited_tex1 = torch.from_numpy(1 - alpha_im1).unsqueeze(-1) * texture_orig1 + torch.from_numpy(alpha_im1).unsqueeze(-1) * texture_edit_im1

    edited_tex1_only_edit = torch.cat((edited_tex1_only_edit, torch.from_numpy(alpha_im1).unsqueeze(-1)), dim=-1)
    edited_tex2_only_edit = torch.from_numpy(texture_edit_im2)
    edited_tex2 = torch.from_numpy(1 - alpha_im2).unsqueeze(-1) * texture_orig2 + torch.from_numpy(alpha_im2).unsqueeze(-1) * texture_edit_im2

    edited_tex2_only_edit = torch.cat((edited_tex2_only_edit, torch.from_numpy(alpha_im2).unsqueeze(-1)), dim=-1)
    alpha_reconstruction = np.zeros((resy, resx, number_of_frames))

    masks1 = np.zeros((edited_tex1.shape[0], edited_tex1.shape[1]))
    masks2 = np.zeros((edited_tex2.shape[0], edited_tex2.shape[1]))

    only_mapped_texture = np.zeros((resy, resx, 4, number_of_frames))
    only_mapped_texture2 = np.zeros((resy, resx, 4, number_of_frames))
    rgb_edit_video = np.zeros((resy, resx, 3, number_of_frames))

    with torch.no_grad():
        begain_time = time.time()
        for f in tqdm(range(number_of_frames), 'Reconstruct the Frames from Edited Atlas'):
            if evaluate_all_pixels:
                relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)
            else:
                relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)

            relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 51200))
            reljsa = np.array_split(reljs_i.numpy(), np.ceil(relis_i.shape[0] / 51200))

            for i in range(len(relisa)):
                relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (larger_dim / 2) - 1
                reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (larger_dim / 2) - 1

                mapping_outs = model_F_mapping(
                    torch.cat((reljs, relis,
                               (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                              dim=1).to(device))
                uv_temp1 = mapping_outs[..., :2]
                uv_temp2 = mapping_outs[..., 2:4]
                alpha = 0.5 * (mapping_outs[..., -1:] + 1)
                alpha = alpha * 0.99
                alpha = alpha + 0.001

                uv_temp1 = uv_temp1.detach().cpu()
                uv_temp2 = uv_temp2.detach().cpu()
                # sample the edit colors from the edited atlases in the relevant uv coordinates
                rgb_only_edit, pointsx1, pointsy1, relevant1_only_edit = get_colors(ReconstructResolution, minx, minx + edge_size, miny,
                                                                                    miny + edge_size,
                                                                                    uv_temp1[:, 0] * 0.5 + 0.5,
                                                                                    uv_temp1[:, 1] * 0.5 + 0.5,
                                                                                    edited_tex1_only_edit)

                rgb_only_edit2, pointsx2, pointsy2, relevant2_only_edit = get_colors(ReconstructResolution,
                                                                                     minx2, minx2 + edge_size2, miny2,
                                                                                     miny2 + edge_size2,
                                                                                     uv_temp2[:, 0] * 0.5 - 0.5,
                                                                                     uv_temp2[:, 1] * 0.5 - 0.5,
                                                                                     edited_tex2_only_edit)

                try:
                    masks2[np.ceil(pointsy2).astype((np.int64)), np.ceil(pointsx2).astype((np.int64))] = 1
                    masks2[np.floor(pointsy2).astype((np.int64)), np.floor(pointsx2).astype((np.int64))] = 1
                    masks2[np.floor(pointsy2).astype((np.int64)), np.ceil(pointsx2).astype((np.int64))] = 1
                    masks2[np.ceil(pointsy2).astype((np.int64)), np.floor(pointsx2).astype((np.int64))] = 1
                except Exception:
                    pass

                try:
                    masks1[np.ceil(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))] = 1
                    masks1[np.floor(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))] = 1
                    masks1[np.floor(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))] = 1
                    masks1[np.ceil(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))] = 1
                except Exception:
                    pass
                alpha_reconstruction[relisa[i], reljsa[i], f] = alpha[:, 0].detach().cpu().numpy()

                # save the video pixels of the edits from the two atlases
                only_mapped_texture[relisa[i][relevant1_only_edit], reljsa[i][relevant1_only_edit], :, f] = rgb_only_edit

                only_mapped_texture2[relisa[i][relevant2_only_edit], reljsa[i][relevant2_only_edit], :, f] = rgb_only_edit2
            # see details in Section 3.4 in the paper
            foreground_edit_cur = only_mapped_texture[:, :, :3, f]  # denoted in the paper by c_{ef}
            foreground_edit_cur_alpha = only_mapped_texture[:, :, 3, f][:, :, np.newaxis]  # denoted by \alpha_{ef}

            background_edit_cur = only_mapped_texture2[:, :, :3, f]  # denoted in the paper by c_{eb}
            background_edit_cur_alpha = only_mapped_texture2[:, :, 3, f][:, :, np.newaxis]  # denoted in the paper by \alpha_{eb}

            video_frame_cur = video_frames[:, :, :, f].cpu().clone().numpy()  # denoted in the paper by \bar{c}_{b}

            # Equation (15):
            video_frame_cur_edited1 = foreground_edit_cur * (foreground_edit_cur_alpha) + video_frame_cur * (1 - foreground_edit_cur_alpha)  # \bar{c}_b
            video_frame_cur_edited2 = background_edit_cur * (background_edit_cur_alpha) + video_frame_cur * (1 - background_edit_cur_alpha)  # \bar{c}_f

            cur_alpha = alpha_reconstruction[:, :, f][:, :, np.newaxis]


            # For values in the foreground that are greater than 0, the original alpha value is retained. Otherwise, alpha is set to 0 (in this case, the pixels on the frame mainly come from back-atlas)
            if mask_of_new_obj:
                gray_fore = cv2.cvtColor((video_frame_cur_edited1*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                cur_alpha = cur_alpha * (gray_fore[:, :, None] > 0).astype(np.float32)
            final_edited_frame_output = video_frame_cur_edited1 * cur_alpha + (1 - cur_alpha) * video_frame_cur_edited2

            rgb_edit_video[:, :, :, f] = final_edited_frame_output

        using_time = time.time() - begain_time
        print(f"\n {output_folder_final}\n using_time:{using_time} FPS:{number_of_frames/using_time} inference time one image:{using_time/number_of_frames}s\n")

    mpimg.imsave("%s/atlas_ori_foreground.png" % output_folder_final, (masks1[:, :, np.newaxis] * texture_orig1.numpy() * (255)).astype(np.uint8))
    mpimg.imsave("%s/atlas_ori_background.png" % output_folder_final, (masks2[:, :, np.newaxis] * texture_orig2.numpy() * (255)).astype(np.uint8))
    writer_edit = imageio.get_writer("%s/results_based_on_edited_atlas.mp4" % (output_folder_final), fps=10)

    # Save the edit video and frames
    frame_base_path = os.path.join(output_folder_final, 'frames')
    os.makedirs(frame_base_path, exist_ok=True)
    for i in tqdm(range(number_of_frames), 'Save Frames'):
        img = (rgb_edit_video[:, :, :, i] * (255)).astype(np.uint8)
        writer_edit.append_data(img)
        cv2.imwrite(f'{frame_base_path}/{input_files[i].name}', img[..., ::-1])
        if i == 0:
            cv2.imwrite(frontend_save_name, img[..., ::-1])
    writer_edit.close()


def get_ori_img_size(data_folder):
    frame = cv2.imread(glob.glob(f'{data_folder}/*.png')[0])
    h, w, _ = frame.shape
    return np.int64(w), np.int64(h)


def Atlas2frames(training_folder, frames_folder, mask_folder, edit_tex1_file, edit_tex2_file, output_folder, frontend_save_name='./frontend.png', mask_of_new_obj=None):
    # mask_of_new_obj: When true, the alpha of points with pixel values>0 remains unchanged, while the rest are set to 0; When false, it means using the original alpha
    # Read the config of the trained model
    with open("%s/atlas_config.json" % training_folder) as f:
        config = json.load(f)

    config['data_folder'] = frames_folder

    maximum_number_of_frames = config["maximum_number_of_frames"]
    resx, resy = get_ori_img_size(config['data_folder'])
    data_folder = Path(frames_folder)
    mask_dir = Path(mask_folder)
    input_files = sorted(list(data_folder.glob('*.jpg')) + list(data_folder.glob('*.png')))
    mask_files = sorted(list(mask_dir.glob('*.jpg')) + list(mask_dir.glob('*.png')))

    number_of_frames = np.minimum(maximum_number_of_frames, len(input_files))
    video_frames = torch.zeros((resy, resx, 3, number_of_frames))
    mask_frames = torch.zeros((resy, resx, number_of_frames))

    for i in range(number_of_frames):
        file1 = input_files[i]
        im = np.array(Image.open(str(file1))).astype(np.float64) / 255.
        mask = np.array(Image.open(str(mask_files[i]))).astype(np.float64) / 255.
        mask = cv2.resize(mask, (resx, resy), cv2.INTER_NEAREST)
        mask_frames[:, :, i] = torch.from_numpy(mask)
        video_frames[:, :, :, i] = torch.from_numpy(cv2.resize(im[:, :, :3], (resx, resy)))

    config_hash = {
            "encoding": {
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 15,
                    "base_resolution": 16,
                    "per_level_scale": 1.5
            },
            "network": {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Tanh",
                    "n_neurons": 64,
                    "n_hidden_layers": 2
            }
    }

    model_F_mapping = IMLP(  # foreground ori_img coordinates[x,y,t] --> [u,v]
        input_dim=3,
        output_dim=5,  # [u1, v1, u2, v2, alpha]
        hidden_dim=256,
        use_positional=False,
        num_layers=8,
        skip_layers=[3, 6]).to(device)

    model_F_atlas = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=3,
        encoding_config=config_hash["encoding"],
        network_config=config_hash["network"]).to(device)

    checkpoint = torch.load("%s/checkpoint" % training_folder)

    model_F_atlas.load_state_dict(checkpoint["F_atlas_state_dict"])
    model_F_atlas.eval()
    model_F_atlas.to(device)

    model_F_mapping.load_state_dict(checkpoint["model_F_mapping_state_dict"])
    model_F_mapping.eval()
    model_F_mapping.to(device)

    edit_im1 = PIL_rgba2rgb(imageio.imread(edit_tex1_file))[:, :, :3] / 255.0
    ori_edit_im1 = PIL_rgba2rgb(imageio.imread(edit_tex1_file))
    if ori_edit_im1.shape[2] == 3:
        new_image = np.zeros((ori_edit_im1.shape[0], ori_edit_im1.shape[1], 4), dtype=np.uint8)
        new_image[:, :, :3] = ori_edit_im1  # copy RGB channels
        new_image[:, :, 3] = 255  # set Alpha channel 255
        alpha_im1 = new_image[:, :, 3] / 255.0
    else:
        assert False
        alpha_im1 = imageio.imread(edit_tex1_file)[:, :, 3] / 255.0

    edit_im2 = PIL_rgba2rgb(imageio.imread(edit_tex2_file))[:, :, :3] / 255.0
    original_image = PIL_rgba2rgb(imageio.imread(edit_tex2_file))
    if original_image.shape[2] == 3:
        new_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
        new_image[:, :, :3] = original_image
        new_image[:, :, 3] = 255
        alpha_im2 = new_image[:, :, 3] / 255.
    else:
        assert False
        alpha_im2 = imageio.imread(edit_tex2_file)[:, :, 3] / 255.0

    output_folder_final = output_folder

    Path(output_folder_final).mkdir(parents=True, exist_ok=True)

    apply_edit(model_F_atlas, resx, resy, number_of_frames, model_F_mapping,
               video_frames, output_folder_final, mask_frames,
               texture_edit_im1=edit_im1,
               texture_edit_im2=edit_im2, alpha_im1=alpha_im1, alpha_im2=alpha_im2, input_files=input_files, frontend_save_name=frontend_save_name, mask_of_new_obj=mask_of_new_obj)


if __name__ == "__main__":
    training_folder = 'Your/dataset_zoos/work_space/atlas_space/training_results'
    frames_folder = 'Your/dataset_zoos/CE3D_collect/linkoping_stones_phone/images'
    mask_folder = 'Your/dataset_zoos/scece/work_space/atlas_space/mask'
    edit_tex1_file = 'Your/atlas_edited_foreground.png'
    edit_tex2_file = 'Your/atlas_edited_background.png'
    output_folder = './edited_atlas2frames'
    frontend_save_name = './frontend.png'

    Atlas2frames(training_folder, frames_folder, mask_folder, edit_tex1_file, edit_tex2_file, output_folder, frontend_save_name='./frontend.png', mask_of_new_obj=0)

