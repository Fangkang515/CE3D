# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding: utf-8
import os
import argparse
import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from copy import deepcopy


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import wget

from datetime import datetime

import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../model_zoos/ProPainter")
from model_zoos.ProPainter.inference_propainter_for_3D_editing import (
    inference_propainter,
    config_parser_ProPainter,
)

import urllib3, socket
from urllib3.connection import HTTPConnection

from train_atlas import train_atlas


HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000),
]


def pad_numpy_edge(mask, padding):
    true_indices = np.argwhere(mask)
    mask_array = np.zeros_like(mask, dtype=bool)
    for idx in true_indices:
        padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
        mask_array[padded_slice] = True
    new_mask = (mask_array * 255).astype(np.uint8)
    return new_mask


class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        try:
            raw_image = Image.open(image_path).convert("RGB").resize((512, 512))
        except FileNotFoundError:
            raw_image = (
                Image.open(os.path.join("./frontend_images", image_path))
                .convert("RGB")
                .resize((512, 512))
            )
        inputs = self.processor(raw_image, question, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
            f"Output Answer: {answer}"
        )
        return answer


class Segmenting:
    def __init__(self, device):
        print(f"Inintializing Segmentation to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.model_checkpoint_path = os.path.join("../checkpoints", "sam")

        self.download_parameters()
        self.sam = build_sam(checkpoint=self.model_checkpoint_path).to(device)
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        self.saved_points = []
        self.saved_labels = []

    def download_parameters(self):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)

    def show_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        random_color: bool = False,
        transparency=1,
    ) -> np.ndarray:
        """Visualize a mask on top of an image.
        Args:
            mask (np.ndarray): A 2D array of shape (H, W).
            image (np.ndarray): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Outputs:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
            transparenccy: the transparency of the segmentation mask
        """

        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        image = cv2.addWeighted(image, 0.7, mask_image.astype("uint8"), transparency, 0)

        return image

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
        ax.text(x0, y0, label)

    def get_mask_with_boxes(self, image_pil, image, boxes_filt):
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks

    def segment_image_with_boxes(self, image_pil, image_path, boxes_filt, pred_phrases):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)

        # draw output image

        for mask in masks:
            image = self.show_mask(
                mask[0].cpu().numpy(), image, random_color=True, transparency=0.3
            )

        updated_image_path = get_new_image_name(image_path, func_name="segmentation")

        new_image = Image.fromarray(image)
        # new_image.save('./seg.png')
        new_image.save(updated_image_path)

        return updated_image_path

    def set_image(self, img) -> None:
        """Set the image for the predictor."""
        with torch.cuda.amp.autocast():
            self.sam_predictor.set_image(img)

    def show_points(
        self, coords: np.ndarray, labels: np.ndarray, image: np.ndarray
    ) -> np.ndarray:
        """Visualize points on top of an image.

        Args:
            coords (np.ndarray): A 2D array of shape (N, 2).
            labels (np.ndarray): A 1D array of shape (N,).
            image (np.ndarray): A 3D array of shape (H, W, 3).
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the points
            visualized on top of the image.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        for p in pos_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(0, 255, 0), thickness=-1
            )
        for p in neg_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(255, 0, 0), thickness=-1
            )
        return image


class Text2Box:
    def __init__(self, device):
        print(f"Initializing ObjectDetection to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.model_checkpoint_path = os.path.join("../checkpoints", "groundingdino")
        self.model_config_path = os.path.join("../checkpoints", "grounding_config.py")
        self.download_parameters()
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.grounding = (self.load_model()).to(self.device)

    def download_parameters(self):
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        if not os.path.exists(self.model_config_path):
            wget.download(config_url, out=self.model_config_path)

    def load_image(self, image_path):
        # from IPython import embed; embed()
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([512], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_boxes(self, image, caption, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.grounding(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.grounding.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases


class GetMask:  # example prompt for removing objecting in 3D scene = 'please remove the xx from the 3D scene in path path/of/3d/scene'
    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting):
        print("Initializing 3D remove")
        self.sam = Segmenting
        self.grounding = Text2Box

    def pad_edge(self, mask, padding):  # mask Tensor [H,W]
        mask = mask.numpy()
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        return new_mask

    def get_mask_by_sam(self, inputs, mask_save_path):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        print(f"image_path={image_path}, to_be_replaced_txt={to_be_replaced_txt}")

        image_pil, image = self.grounding.load_image(image_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(
            image, to_be_replaced_txt
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu()  # tensor
        mask = self.pad_edge(mask, padding=1)  # numpy
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_save_path)
        print(f"save mask in: {mask_save_path}")


global_vqa = VisualQuestionAnswering(device="cuda:0")
get_mask = GetMask(Text2Box(device="cuda:0"), Segmenting(device="cuda:0"))


def run_workspace_init(folder_path_3D):
    """
    - auto makedirs of the workspace / editing space / frontend / backend
    - auto generate the foreground and background Atlas
    |-images_8
    |-work_space
    |  |atlas_space
    |  |  |flow
    |  |  |mask
    |  |  |propainter
    |  |  |training_results
    |  |final_atlas
    |  |  | atlas_ori_foreground.png
    |  |  | atlas_ori_background.png
    |  |editing_space  # edited 3D scene
    |  |  |turn_into_autumn
    |  |  |remove_flower
    |-frontend_images  # for interface displayed images
    |  |abc.png
    |-backend_images  # for real edited images of atlas
    |  |abc_foreground_atlas.png
    |  |abc_background_atlas.png
    """

    folder_path_3D_list = os.listdir(folder_path_3D)
    ori_img_path_name = "images_8" if "images_8" in folder_path_3D_list else "images_2"
    frames_folder = os.path.join(folder_path_3D, ori_img_path_name)
    ori_img_names = sorted(os.listdir(frames_folder))
    ori_frame_paths = [os.path.join(frames_folder, p) for p in ori_img_names]
    global_display_original_frames = deepcopy(ori_frame_paths)

    atlas_space = os.path.join(folder_path_3D, "work_space", "atlas_space")
    mask_folder = os.path.join(
        atlas_space, "mask"
    )  # train Atlas： mask foreground and background
    flow_folder = os.path.join(
        atlas_space, "flow"
    )  # train Atlas: optical flow for consistent frames
    training_atlas_results = os.path.join(
        atlas_space, "training_results"
    )  # train Atlas: optical flow for consistent frames
    propainter_folder = os.path.join(
        atlas_space, "propainter"
    )  # train Atlas: one of the GT for background Atlas (Mainly for subsequent remove purpose)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(flow_folder, exist_ok=True)
    os.makedirs(propainter_folder, exist_ok=True)
    os.makedirs(training_atlas_results, exist_ok=True)

    # using for save edited [atlas / frontend_img_displayed_to_user / all_edited_frames] results according to user's chat prompt
    final_atlas_path = os.path.join(folder_path_3D, "work_space", "final_atlas")
    os.makedirs(final_atlas_path, exist_ok=True)

    editing_space = os.path.join(folder_path_3D, "work_space", "editing_space")
    os.makedirs(os.path.join(editing_space, "frames"), exist_ok=True)
    global_display_edited_frames = [
        p.replace(ori_img_path_name, "work_space/editing_space/frames")
        for p in global_display_original_frames
    ]

    #  using VQA to get the foreground and background
    foreground = global_vqa.inference(
        f"{ori_frame_paths[0]},what are the main objects of this image?"
    )  # .split(' ')[-1]
    background = global_vqa.inference(
        f"{ori_frame_paths[0]},what are the background of this image?"
    )  # .split(' ')[-1]
    print(
        f"\n======>the scene's foreground is {foreground}, and background is {background}"
    )

    if not os.path.isfile(os.path.join(final_atlas_path, "atlas_ori_foreground.png")):
        # step1: get the foreground mask and save to mask_folder
        for i, image_path in enumerate(ori_frame_paths):
            mask_save_path = os.path.join(mask_folder, ori_img_names[i])
            get_mask.get_mask_by_sam(
                f"{image_path},{foreground},background", mask_save_path
            )

        # step2: get the propainter results and save to propainter_folder
        inference_propainter(
            video_path=frames_folder,
            mask_path=mask_folder,
            save_path=propainter_folder,
            args=args,
            use_half=False,
        )

        # step3: get the optical flow, then training the foreground and background Atlas
        train_atlas(frames_folder)

    # prepare the frontend image and its caption for the interaction with the user
    print("\n======>Processing images for interaction Chat...")
    img = Image.open(ori_frame_paths[0])
    width, height = img.size
    ratio = min(512 / width, 512 / height)
    width_new, height_new = (round(width * ratio), round(height * ratio))
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64
    img = img.resize((width_new, height_new))
    print(f"Resize image from {width}x{height} to {width_new}x{height_new}")
    img = img.convert("RGB")

    # We use separated frontend and backend space to arange the conversation interface, foreground atlas, and background atlas
    os.makedirs("./frontend_images", exist_ok=True)
    os.makedirs("./backend_images", exist_ok=True)
    os.system("rm ./frontend_images/*.png")
    os.system("rm ./backend_images/*.png")
    uuid_current = get_current_time()  # str(uuid.uuid4())[:8]

    frontend_img_path = os.path.join("frontend_images", f"{uuid_current}.png")
    image_filename = os.path.basename(frontend_img_path)
    img.save(frontend_img_path, "PNG")
    foreground_atlas = Image.open(
        os.path.join(final_atlas_path, "atlas_ori_foreground.png")
    ).resize((512, 512))
    foreground_atlas.save(
        os.path.join("backend_images", f"{uuid_current}_atlas_foreground.png"), "PNG"
    )
    background_atlas = Image.open(
        os.path.join(final_atlas_path, "atlas_ori_background.png")
    ).resize((512, 512))
    background_atlas.save(
        os.path.join("backend_images", f"{uuid_current}_atlas_background.png"), "PNG"
    )


parser = argparse.ArgumentParser()
parser = config_parser_ProPainter(exist_parser=parser)
args = parser.parse_args()

data_folder = f"../datasets/flower"


run_workspace_init(data_folder)
