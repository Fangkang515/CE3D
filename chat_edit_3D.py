# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
import numpy as np
import argparse
import inspect
import tempfile
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from copy import deepcopy

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines import BlipDiffusionControlNetPipeline
from controlnet_aux import CannyDetector

from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

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
from TRF.train import reconstruction
from TRF.opt import config_parser as config_parser_TRF

from datetime import datetime
from IPython import embed
import openai

from prompts_engineering import (
    VISUAL_CHATGPT_PREFIX,
    VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
    VISUAL_CHATGPT_SUFFIX,
    VISUAL_CHATGPT_PREFIX_CN,
    VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN,
    VISUAL_CHATGPT_SUFFIX_CN,
    INTRO,
    CSS,
)

import sys

sys.path.insert(0, "./model_zoos/ProPainter")
from model_zoos.ProPainter.inference_propainter_for_3D_editing import (
    inference_propainter,
    config_parser_ProPainter,
)

sys.path.insert(0, "./hash_atlas")
from hash_atlas.only_edit import Atlas2frames


import urllib3, socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000),
]


os.makedirs("image", exist_ok=True)


func_name_mapping = {
    "line": "Line",
    "line2image": "Line2Img",
    "scribble": "Scr",
    "scribble2image": "Scr2Img",
    "depth": "Dep",
    "depth2image": "Dep2Img",
    "normal": "Norm",
    "normal2image": "Norm2Img",
    "edge": "Edge",
    "edge2image": "Edge2Img",
    "hed": "Hed",
    "hed2image": "Hed2Img",
    "pose": "Pose",
    "pose2image": "Pose2Img",
    "seg2image": "Seg2Img",
    "instruct-pix2pix": "p2p",
    "image-driven": "ImgDri",
    "Remove": "Rm",
    "Replace": "Rp",
    "RemoveBackground": "RmB",
    "update": "Upd",
    "segmentation": "seg",
    "SR": "SR",
    "Extract": "Ext",
    "Doubao": "DB",
    "GPT4oImage": "4o",
    "GhibliStyle": "Ghibli",
    "Dereflection": "Defle",
    "LowLight": "LL",
    "ArtisticTypography": "ArTypo",
    "Defocus": "Defo",
    "Recoloring": "Reco",
    "Sturation": "Stu",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def get_current_time():
    current_datetime = datetime.now()
    current_datetime = current_datetime.strftime("%m%d%H%M")
    return current_datetime


def get_new_image_name(org_img_name, func_name="update", area="", obj=""):
    # ori_img_name like: 09181123.png, the func_name like: Rm (means remove), area like:'b','f' or 'bf'. b means background, f means foreground
    new_id = datetime.now().strftime("%M%S")
    func_name = func_name_mapping[func_name]
    new_file_name = f"{org_img_name}_{new_id}_{area}_{func_name}".strip("_")
    return new_file_name


def set_zero_atlas(in_path, out_path):
    img = cv2.imread(in_path) * 0.0
    img = img.astype(np.uint8)
    cv2.imwrite(out_path, img)


def save_same_atlas_diff_name(in_path, out_path):
    img = cv2.imread(in_path)
    cv2.imwrite(out_path, img)


def PIL_rgba2rgb(img):
    img = np.array(img).astype(np.float32)
    img = img[..., :3] * (img[..., 3:] / 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img


def frontend_2_backend(image_name, objects="", text="", func_name=""):
    """
    text is optional. it can be empty if this event is not editing by a instruct prompt
    """
    image_paths_to_edit, area = (
        [],
        "",
    )  # area could be: f, b, fb, where f means foreground, b means background
    image_path_fore = os.path.join(
        "./backend_images", f"{image_name}_atlas_foreground.png"
    )
    image_path_back = os.path.join(
        "./backend_images", f"{image_name}_atlas_background.png"
    )
    instruct_text = text
    if objects == "foreground":
        image_paths_to_edit.append(image_path_fore)
        area = "f"  # foreground
        instruct_text = text.replace("foreground", "image")
    elif objects == "background":
        image_paths_to_edit.append(image_path_back)
        area = "b"
        instruct_text = text.replace("background", "image")
    elif "entire" in objects or objects == "":
        image_paths_to_edit.append(image_path_fore)
        image_paths_to_edit.append(image_path_back)
        area = "fb"  # foreground and background
    else:  # using vqa to determine the editing area. the objects could be in foreground,background or both.
        answer_fore = global_vqa.inference(
            f"{image_path_fore},Is there {objects} in the image?"
        )
        # answer_back = global_vqa.inference(f"{image_path_back},Is there {objects} in the image?")
        if answer_fore == "yes":
            image_paths_to_edit.append(image_path_fore)
            area = "f"
        else:
            image_paths_to_edit.append(image_path_back)
            area = "b"
        """
        if answer_fore == 'yes' and answer_back == 'no':
            image_paths_to_edit.append(image_path_fore)
            area = 'f'
        elif answer_fore == 'no' and answer_back == 'yes':
            image_paths_to_edit.append(image_path_back)
            area = 'b'
        else:  # answer_fore and answer_back are both 'no' or both 'yes'
            image_paths_to_edit.append(image_path_fore)
            image_paths_to_edit.append(image_path_back)
            area = 'fb'
        """
    new_image_name = get_new_image_name(
        image_name, func_name=func_name, area=area, obj=objects.replace(" ", "_")
    )
    new_image_path_fore = os.path.join(
        "./backend_images", f"{new_image_name}_atlas_foreground.png"
    )
    new_image_path_back = os.path.join(
        "./backend_images", f"{new_image_name}_atlas_background.png"
    )
    return (
        image_path_fore,
        image_path_back,
        image_paths_to_edit,
        area,
        new_image_name,
        new_image_path_fore,
        new_image_path_back,
        instruct_text,
    )


def merge_fore_and_back_as_fake_fore(
    fore_path, back_path, new_image_path_fore, save_merge=True
):
    img_f = cv2.imread(fore_path)
    img_b = cv2.imread(back_path)
    mask = (cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY) > 2).astype(np.float32)
    img_merge = mask[:, :, None] * img_f + (1 - mask[:, :, None]) * img_b
    img_merge_path = new_image_path_fore.replace("foreground", "merge")
    assert img_merge_path != new_image_path_fore
    if save_merge:
        cv2.imwrite(img_merge_path, img_merge)
    return img_merge_path


def pad_numpy_edge(mask, padding):
    true_indices = np.argwhere(mask)
    mask_array = np.zeros_like(mask, dtype=bool)
    for idx in true_indices:
        padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
        mask_array[padded_slice] = True
    new_mask = (mask_array * 255).astype(np.uint8)
    return new_mask


class MergeEditSplitPipe:
    @staticmethod
    def merge_atlas(image_name, func_name, objects="", save_merge=True):
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
        ) = frontend_2_backend(image_name, func_name=func_name, objects=objects)
        merge_atlas_path = merge_fore_and_back_as_fake_fore(
            image_path_fore, image_path_back, new_image_path_fore, save_merge=save_merge
        )
        merge_edited_atlas_path = merge_atlas_path.replace(".png", "_edited.png")
        return (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        )

    @staticmethod
    def split_atlas(
        image_path_fore,
        merge_edited_atlas_path,
        new_image_path_fore,
        new_image_path_back,
        using_subtraction=False,
    ):
        mask_fore = global_foreground_mask
        # mask_fore = pad_numpy_edge(mask_fore, 2)
        updated_fore_atlas = cv2.imread(merge_edited_atlas_path).astype(np.float32) * (
            mask_fore.astype(np.float32)[:, :, None] / 255.0
        )
        cv2.imwrite(new_image_path_fore, updated_fore_atlas.astype(np.uint8))

        mask_fore = Image.fromarray(mask_fore)
        mask_fore.save(new_image_path_fore.replace(".png", "_mask.png"))

        img_merge = Image.open(merge_edited_atlas_path)
        if using_subtraction:
            img_merge = np.array(img_merge)
            new_back_atlas = (img_merge[:, :, ::-1] - updated_fore_atlas).astype(
                np.uint8
            )
            cv2.imwrite(new_image_path_back, new_back_atlas)
        else:
            # mask edited_merge_atlas ---> inpaiting the mask area ---> inpainted result as the background_atlas
            new_back_atlas = global_inpaint("background", img_merge, mask_fore)
            new_back_atlas.save(new_image_path_back)


class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    @prompts(
        name="Answer Question About The Scene",
        description="useful when you need an answer for a question based on a scene. "
        "like: what is the foreground color of the scene? Is there a building in the background?"
        "The input to this tool should be a comma separated string of two, representing the scene_path and the question",
    )
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        image_path.replace(".scn", ".png")
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


class Inpainting:
    def __init__(self, device):
        self.device = device
        self.revision = "fp16" if "cuda" in self.device else None
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
        ).to(device)

    def __call__(
        self, prompt, image, mask_image, height=512, width=512, num_inference_steps=50
    ):
        update_image = self.inpaint(
            prompt=prompt,
            image=image.resize((width, height)),
            mask_image=mask_image.resize((width, height)),
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        ).images[0]
        return update_image


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    @prompts(
        name="Get Scene Description",
        description="useful when you want to know the caption of the scene. like: describe this scene. "
        "The input to this tool should be a string, representing the scene_path. ",
    )
    def inference(self, image_path):
        image_path.replace(".scn", ".png")
        try:
            inputs = self.processor(
                Image.open(image_path).convert("RGB"), return_tensors="pt"
            ).to(self.device, self.torch_dtype)
        except FileNotFoundError:
            inputs = self.processor(
                Image.open(os.path.join("./frontend_images", image_path)).convert(
                    "RGB"
                ),
                return_tensors="pt",
            ).to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}"
        )
        return captions


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            resume_download=True,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image From User Input Text",
        description="useful when you want to generate an image based only on a user input text. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, text):
        uuid_current = get_current_time()  # str(uuid.uuid4())[:8]
        new_id = datetime.now().strftime("%S")
        image_filename = f"{uuid_current}_{new_id}.png"
        prompt = text + ", " + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(os.path.join("frontend_images", image_filename))
        image.save(os.path.join("backend_images", image_filename))
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}"
        )
        return image_filename


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(
        name="Edge Detection On Scene",
        description="useful when you want to detect the edge or canny of the scene. "
        "like: detect the edges of this scene, or canny detection on scene, "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="edge")
        updated_image_path = f"{new_image_name}.png"
        image = Image.open(merge_atlas_path).convert("RGB")

        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        canny.save(merge_edited_atlas_path)
        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
            using_subtraction=False,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )
        print(
            f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}"
        )
        return updated_image_path


class CannyText2Image:
    def __init__(self, device):
        print(f"Initializing CannyText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-canny",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Canny Scene",
        description="useful when you want to generate a new real Scene from both the user description and a canny Scene."
        " like: generate a real Scene of a object or something from this canny/edge Scene,"
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description. ",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="edge2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed CannyText2Image, Input Canny: {image_path}, Input Text: {instruct_text}, "
            f"Output Text: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Image2Line:
    def __init__(self, device):
        print("Initializing Image2Line")
        self.detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Line Detection On Scene",
        description="useful when you want to detect the straight line of the Scene. "
        "like: detect the straight lines of this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="line")
        updated_image_path = f"{new_image_name}.png"
        image = Image.open(merge_atlas_path).convert("RGB")

        mlsd = self.detector(image)
        mlsd.save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
            False,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}"
        )
        return updated_image_path


class LineText2Image:
    def __init__(self, device):
        print(f"Initializing LineText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-mlsd", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Line Scene",
        description="useful when you want to generate a new real Scene from both the user description and a straight line Scene. "
        "like: generate a real Scene of a object or something from this straight line Scene, "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description. ",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="line2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
            f"Output Text: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

    @prompts(
        name="Hed Detection On Scene",
        description="useful when you want to detect the soft hed boundary of the scene. "
        "like: detect the soft hed boundary of this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="hed")
        updated_image_path = f"{new_image_name}.png"
        image = Image.open(merge_atlas_path).convert("RGB")

        hed = self.detector(image)
        hed.save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
            True,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {updated_image_path}"
        )
        return updated_image_path


class HedText2Image:
    def __init__(self, device):
        print(f"Initializing HedText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Soft Hed Boundary Scene",
        description="useful when you want to generate a new real Scene from both the user description and a soft hed boundary Scene. "
        "like: generate a real Scene of a object or something from this soft hed boundary Scene, "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="hed2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        # self.detector = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

    @prompts(
        name="Scribble Detection On Scene",
        description="useful when you want to generate a scribble or sketch of the Scene. "
        "like: generate a scribble or sketch of this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="scribble")
        updated_image_path = f"{new_image_name}.png"
        image = Image.open(merge_atlas_path).convert("RGB")

        scribble = self.detector(image, scribble=True)
        scribble.save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
            True,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed Image2Scribble, Input Image: {inputs}, Output Scribble: {updated_image_path}"
        )
        return updated_image_path


class ScribbleText2Image:
    def __init__(self, device):
        print(f"Initializing ScribbleText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-scribble",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Scribble Scene",
        description="useful when you want to generate a new real Scene from both the user description and a scribble Scene. "
        "like: generate a real Scene of a object or something from this scribble or sketch. "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="scribble2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Image2Pose:
    def __init__(self, device):
        print("Initializing Image2Pose")
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Pose Detection On Scene",
        description="useful when you want to detect the human pose of the Scene. "
        "like: generate human poses of this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="pose")
        updated_image_path = f"{new_image_name}.png"
        image = Image.open(merge_atlas_path).convert("RGB")

        pose = self.detector(image)
        pose.save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}"
        )
        return updated_image_path


class PoseText2Image:
    def __init__(self, device):
        print(f"Initializing PoseText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Pose Scene",
        description="useful when you want to generate a new real Scene from both the user description and a human pose Scene. "
        "like: generate a new real Scene of a human from this pose. "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="pose2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed PoseText2Image, Input Pose: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class SegText2Image:
    def __init__(self, device):
        print(f"Initializing SegText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-seg", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Segmentations",
        description="useful when you want to generate a new real Scene from both the user description and segmentations. "
        "like: generate a real Scene of a object or something from this segmentation Scene, "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="seg2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)
        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class SRImage:
    def __init__(self, device):
        print("Initializing Super Resolution Image")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, variant="fp16", torch_dtype=self.torch_dtype
        ).to(device)

    @prompts(
        name="Make Scene Clearer (Super Resolution)",
        description="useful when you want to makethe Scene clearer. "
        "like: make this Scene clearer or deblur this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="SR")
        # image_path_fore, image_path_back, image_paths_to_edit, area, new_image_name, new_image_path_fore, new_image_path_back, instruct_text = frontend_2_backend(image_name, '', '', 'SR')
        updated_image_path = f"{new_image_name}.png"
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            "fewer digits, cropped, worst quality, low quality"
        )

        ori_prompt = global_img_caption.inference(merge_atlas_path)
        prompt = f"{ori_prompt}, {self.a_prompt}"
        print(ori_prompt)

        low_res_image_fore = Image.open(image_path_fore).convert("RGB")
        low_res_image_fore.save(new_image_path_fore)  # for next other type editing
        high_res_image_fore = self.pipeline(
            prompt, image=low_res_image_fore, negative_prompt=self.n_prompt
        ).images[0]
        high_res_image_fore.resize((1000, 1000)).save(
            new_image_path_fore.replace(".png", "_sr.png")
        )

        low_res_image_back = Image.open(image_path_back).convert("RGB")
        low_res_image_fore.save(new_image_path_back)
        high_res_image_back = self.pipeline(prompt, image=low_res_image_back).images[0]
        high_res_image_back.resize((1000, 1000)).save(
            new_image_path_back.replace(".png", "_sr.png")
        )

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore.replace(".png", "_sr.png"),
            new_image_path_back.replace(".png", "_sr.png"),
            editing_space,
            f"./frontend_images/{new_image_name}.png",
            resolution=1000,
        )

        print(
            f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}"
        )
        return updated_image_path


class InstructPix2Pix:
    def __init__(self, device):
        print(f"Initializing InstructPix2Pix to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    @prompts(
        name="Instruct Scene Using Text",
        description="useful when you want to the style, weather, color of a Scene to be like the text. "
        "like: make it look like a painting. or make it like a robot. or turn into autumn. or color the flower yellow. "
        "The input to this tool should be a comma separated string of three, "
        "representing the scene_path, the Human's text, and the object involved in this conversation.",
    )
    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting InstructPix2Pix Inference")
        image_name, text, objects = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
            inputs.split(",")[2].strip(),
        )
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
        ) = frontend_2_backend(image_name, objects, text, "instruct-pix2pix")

        # using tool to process the atlas and save the new atlas
        for image_path in image_paths_to_edit:
            print(image_path)
            original_image = Image.open(image_path).convert("RGB")
            image = self.pipe(
                instruct_text,
                image=original_image,
                num_inference_steps=40,
                image_guidance_scale=1.95,
                guidance_scale=5.5,
            ).images[0]
            save_image_path = (
                new_image_path_back
                if "background" in image_path
                else new_image_path_fore
            )
            image.save(save_image_path)
        if area == "f":
            Image.open(image_path_back).save(new_image_path_back)
        if area == "b":
            Image.open(image_path_fore).save(new_image_path_fore)

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Original Text: {text}, Using Text: {instruct_text}, Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Image2Depth:
    def __init__(self, device):
        print("Initializing Image2Depth")
        self.depth_estimator = pipeline("depth-estimation")

    @prompts(
        name="Predict Depth On Scene",
        description="useful when you want to detect depth of the Scene. like: generate the depth from this Scene, "
        "or predict the depth for this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="depth")

        image = Image.open(merge_atlas_path).convert("RGB")
        depth = self.depth_estimator(image)["depth"]
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        depth.save(merge_edited_atlas_path)
        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
            False,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )
        print(
            f"\nProcessed Image2Depth, Input Image: {inputs}, Output Depth:{new_image_name}.png"
        )
        return f"{new_image_name}.png"


class DepthText2Image:
    def __init__(self, device):
        print(f"Initializing DepthText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Depth",
        description="useful when you want to generate a new real Scene from both the user description and depth Scene. "
        "like: generate a real Scene of a object or something from this depth map."
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="depth2image")

        image = Image.open(merge_atlas_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed DepthText2Image, Input Depth: {image_name}.png, Input Text: {instruct_text}, Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Image2Normal:
    def __init__(self, device):
        print("Initializing Image2Normal")
        self.depth_estimator = pipeline(
            "depth-estimation", model="Intel/dpt-hybrid-midas"
        )
        self.bg_threhold = 0.4

    @prompts(
        name="Predict Normal Map On Scene",
        description="useful when you want to detect norm map of the Scene. "
        "like: generate normal map from this Scene, or predict normal map of this Scene. "
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference(self, inputs):
        image_name = inputs.split(".")[0].strip()
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            instruct_text,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="normal")
        updated_image_path = f"{new_image_name}.png"
        image = Image.open(merge_atlas_path).convert("RGB")

        original_size = image.size
        image = self.depth_estimator(image)["predicted_depth"][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        image.save(merge_edited_atlas_path)
        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
            False,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed Image2Normal, Input Image: {inputs}, Output Depth: {updated_image_path}"
        )
        return updated_image_path


class NormalText2Image:
    def __init__(self, device):
        print(f"Initializing NormalText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker", torch_dtype=self.torch_dtype
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Scene Condition On Normal Map",
        description="useful when you want to generate a new real Scene from both the user description and normal map. "
        "like: generate a real Scene of a object or something from this normal map, "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the user description",
    )
    def inference(self, inputs):
        image_name, instruct_text = inputs.split(",")[0].split(".")[
            0
        ].strip(), ",".join(inputs.split(",")[1:])
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="normal2image")
        image_path = merge_atlas_path
        updated_image_path = merge_edited_atlas_path

        image = Image.open(image_path).convert("RGB")
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        image.save(updated_image_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        print(
            f"\nProcessed NormalText2Image, Input Normal: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class ReferenceImageTextEditing:
    def __init__(self, device):
        print(f"Initializing ReferenceImageTextEditing to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
            "Salesforce/blipdiffusion-controlnet", torch_dtype=self.torch_dtype
        ).to(device)
        self.negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    @prompts(
        name="Generate New Scene Condition Base on a Reference image and Text prompt",
        description="useful when you want to generate a new Scene from both the user description and a reference image. "
        "like: generate a new real Scene of an object or something for the Scene X based on the reference image Y, "
        "or refer to image Y to generate a new Scene at night for the Scene X. "
        "The input to this tool should be a comma separated string of three, "
        "representing the scene_path, reference image path, and the user description",
    )
    def inference_ref_txt(self, inputs):
        image_name, refer_img_path, instruct_text = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
            ",".join(inputs.split(",")[2:]),
        )
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="image-driven")

        refer_img_path = os.path.join("./backend_images", refer_img_path)
        style_subject = global_vqa.inference(
            f"{refer_img_path}, what is the main object in the image?"
        )
        tgt_subject = global_vqa.inference(
            f"{merge_atlas_path}, what is the main object in the image?"
        )
        text_prompt = instruct_text  # "at the night"
        cldm_cond_image = Image.open(merge_atlas_path).resize((512, 512))
        style_image = Image.open(refer_img_path)
        canny = CannyDetector()
        cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
        output = self.blip_diffusion_pipe(
            text_prompt,
            style_image,
            cldm_cond_image,
            style_subject,
            tgt_subject,
            guidance_scale=7.5,
            num_inference_steps=50,
            neg_prompt=self.negative_prompt,
            height=512,
            width=512,
        ).images
        output[0].save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )
        print(
            f"\nProcessed RefImageTextEditing, input image:{merge_atlas_path}, reference image: {refer_img_path}, Input Text: {instruct_text}, "
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"

    @prompts(
        name="Generate New Image Condition Only Based on a Reference image",
        description="useful when you want to generate a new image based only on a reference image. "
        "like: generate a new real image for the image X based on the reference image Y, "
        "or refer to image Y to generate a new image for the image X. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image path and the reference image path",
    )
    def inference_ref(self, inputs):
        image_name, refer_img_path = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
        )
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(image_name, func_name="image-driven")

        refer_img_path = os.path.join("./ref_img", refer_img_path)
        style_subject = global_vqa.inference(
            f"{refer_img_path}, what is the main object in the image?"
        )
        tgt_subject = global_vqa.inference(
            f"{merge_atlas_path}, what is the main object in the image?"
        )
        text_prompt = tgt_subject  # "at the night"
        cldm_cond_image = Image.open(merge_atlas_path).resize((512, 512))
        style_image = Image.open(refer_img_path)
        canny = CannyDetector()
        cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
        output = self.blip_diffusion_pipe(
            text_prompt,
            style_image,
            cldm_cond_image,
            style_subject,
            tgt_subject,
            guidance_scale=7.5,
            num_inference_steps=50,
            neg_prompt=self.negative_prompt,
            height=512,
            width=512,
        ).images
        output[0].save(merge_edited_atlas_path)

        MergeEditSplitPipe.split_atlas(
            image_path_fore,
            merge_edited_atlas_path,
            new_image_path_fore,
            new_image_path_back,
        )
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )
        print(
            f"\nProcessed RefImageTextEditing, input image:{merge_atlas_path}, reference image: {refer_img_path},"
            f"Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class Segmenting:
    def __init__(self, device):
        print(f"Inintializing Segmentation to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints", "sam")

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

    def segment_image_with_click(self, img, is_positive: bool, evt: gr.SelectData):
        self.sam_predictor.set_image(img)
        self.saved_points.append([evt.index[0], evt.index[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)
        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        return img

    def segment_image_with_coordinate(self, img, is_positive: bool, coordinate: tuple):
        """
        Args:
            img (numpy.ndarray): the given image, shape: H x W x 3.
            is_positive: whether the click is positive, if want to add mask use True else False.
            coordinate: the position of the click
                      If the position is (x,y), means click at the x-th column and y-th row of the pixel matrix.
                      So x correspond to W, and y correspond to H.
        Output:
            img (PLI.Image.Image): the result image
            result_mask (numpy.ndarray): the result mask, shape: H x W

        Other parameters:
            transparency (float): the transparenccy of the mask
                                  to control he degree of transparency after the mask is superimposed.
                                  if transparency=1, then the masked part will be completely replaced with other colors.
        """
        self.sam_predictor.set_image(img)
        self.saved_points.append([coordinate[0], coordinate[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        img = Image.fromarray(img)

        result_mask = masks[0]

        return img, result_mask

    @prompts(
        name="Segment the Scene",
        description="useful when you want to segment all the part of the Scene, but not segment a certain object."
        "like: segment all the object in this Scene, or generate segmentations on this Scene, "
        "or segment the Scene,"
        "or perform segmentation on this Scene, "
        "or segment all the object in this Scene."
        "The input to this tool should be a string, representing the scene_path",
    )
    def inference_all(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m)))

        updated_image_path = get_new_image_name(image_path, func_name="segment-image")
        plt.axis("off")
        plt.savefig(updated_image_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        return updated_image_path


class Text2Box:
    def __init__(self, device):
        print(f"Initializing ObjectDetection to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints", "groundingdino")
        self.model_config_path = os.path.join("checkpoints", "grounding_config.py")
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
        # load image
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

    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=2)

        return image_pil, mask

    # @prompts(name="Detect the Give Object",
    #         description="useful when you only want to detect or find out given objects in the picture"
    #                     "The input to this tool should be a comma separated string of two, "
    #                     "representing the image_path, the text description of the object to be found")
    @prompts(
        name="Frame the position of the object",
        description="useful when you want to know the position of an object in the picture"
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path, the text description of the object",
    )
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.load_image(image_path)

        boxes_filt, pred_phrases = self.get_grounding_boxes(image, det_prompt)

        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        image_with_box = self.plot_boxes_to_image(image_pil, pred_dict)[0]

        updated_image_path = get_new_image_name(
            image_path, func_name="detect-something"
        )
        updated_image = image_with_box.resize(size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ObejectDetecting, Input Image: {image_path}, Object to be Detect {det_prompt}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class ObjectSegmenting:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting):
        self.grounding = Text2Box
        self.sam = Segmenting

    @prompts(
        name="Segment the given object",
        description="useful when you only want to segment the certain objects in the Scene according to the given text. "
        "like: segment the cat, or can you segment an obeject for me. "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path, the text description of the object to be found",
    )
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.grounding.load_image(image_path)

        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, det_prompt)
        updated_image_path = self.sam.segment_image_with_boxes(
            image_pil, image_path, boxes_filt, pred_phrases
        )
        print(
            f"\nProcessed ObejectSegmenting, Input Image: {image_path}, Object to be Segment {det_prompt}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path

    def merge_masks(self, masks):
        """
        Args:
            mask (numpy.ndarray): shape N x 1 x H x W
        Outputs:
            new_mask (numpy.ndarray): shape H x W
        """
        if type(masks) == torch.Tensor:
            x = masks
        elif type(masks) == np.ndarray:
            x = torch.tensor(masks, dtype=int)
        else:
            raise TypeError(
                "the type of the input masks must be numpy.ndarray or torch.tensor"
            )
        x = x.squeeze(dim=1)
        value, _ = x.max(dim=0)
        new_mask = value.cpu().numpy()
        new_mask.astype(np.uint8)
        return new_mask

    def get_mask(self, image_path, text_prompt):
        print(f"image_path={image_path}, text_prompt={text_prompt}")
        # image_pil (PIL.Image.Image) -> size: W x H
        # image (numpy.ndarray) -> H x W x 3
        image_pil, image = self.grounding.load_image(image_path)

        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(
            image, text_prompt
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        # masks (torch.tensor) -> N x 1 x H x W
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        # merged_mask -> H x W
        merged_mask = self.merge_masks(masks)
        # draw output image
        for mask in masks:
            image = self.sam.show_mask(
                mask[0].cpu().numpy(), image, random_color=True, transparency=0.3
            )

        # merged_mask_image = Image.fromarray(merged_mask)
        return merged_mask


class ObjectRemoveOrReplace:  # remove something(replace with background) or replace something with anther thing
    template_model = True

    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting):
        print("Initializing Object Remove or Replace Editing")
        self.sam = Segmenting
        self.grounding = Text2Box
        self.inpaint = global_inpaint

    def pad_edge(self, mask, padding):
        mask = mask.numpy()  # mask: Tensor [H, W]
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        # new_mask
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

    def merge_fore_and_back_as_fake_fore_1(self, fore_path, back_path, mask):
        img_f = cv2.imread(fore_path)
        img_b = cv2.imread(back_path)
        img_merge = (
            mask[:, :, None] / 255.0 * img_f + (1 - mask[:, :, None] / 255.0) * img_b
        )
        img_f_remain_after_mask = (1 - mask[:, :, None] / 255.0) * img_f
        img_merge_path = fore_path.replace("foreground", "merge")
        assert img_merge_path != fore_path
        cv2.imwrite(img_merge_path, img_merge)
        return img_merge_path, img_f_remain_after_mask

    def get_fore_remain_after_mask(self, fore_path, mask):
        img_f = cv2.imread(fore_path)
        img_f_remain_after_mask = (
            1 - (mask[:, :, None]).astype(np.float32) / 255.0
        ) * img_f
        return img_f_remain_after_mask

    def replace_sth_pipe(
        self, image_path, new_image_name, to_be_replaced_txt, replace_with_txt
    ):
        image_pil, image = self.grounding.load_image(image_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(
            image, to_be_replaced_txt
        )

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        # if the minsize of image is not equal image_pil, the results will not correct
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu()  # tensor
        mask = self.pad_edge(mask, padding=20)  # numpy
        mask_image = Image.fromarray(mask)
        mask_image.resize(image_pil.size).save(
            f"./backend_images/{new_image_name}_FocusOBJ_mask.png"
        )
        updated_image = self.inpaint(
            prompt=replace_with_txt, image=image_pil, mask_image=mask_image
        )
        updated_image = updated_image.resize(image_pil.size)
        return updated_image, np.array(mask_image)

    @prompts(
        name="Remove Something From The Scene",
        description="useful when you want to remove an object or something from the Scene, like: remove the building. "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path and the object need to be removed. ",
    )
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = (
            inputs.split(",")[0].strip(),
            inputs.split(",")[1].strip(),
        )
        inputs = f"{image_path},{to_be_removed_txt},background"
        image_name = image_path.split(".")[0].strip()
        print(image_name)
        _, _, _, area, _, _, _, _ = frontend_2_backend(
            image_name, func_name="Remove", objects=to_be_removed_txt
        )
        if area == "f":
            return self.run_case_remove_from_foreground(inputs)
        else:
            return self.run_case_remove_from_background(inputs)

    def run_case_remove_from_foreground(self, inputs):
        # merge  --> remove foreground from merge --> merge_edited as background, black as foreground
        print("\n-------- run_case_remove_from_foreground ----------\n")
        image_name, to_be_replaced_txt, replace_with_txt = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
            inputs.split(",")[2].strip(),
        )
        print(image_name)
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(
            image_name, func_name="Remove", objects=to_be_replaced_txt
        )
        edited_image, _ = self.replace_sth_pipe(
            merge_atlas_path, new_image_name, to_be_replaced_txt, replace_with_txt
        )
        edited_image.save(new_image_path_back)
        set_zero_atlas(image_path_fore, new_image_path_fore)
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
            True,
        )
        print(
            f"\nProcessed ObjectRemove, Input Image: {image_name}.png, Remove {to_be_replaced_txt}, Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"

    def run_case_remove_from_background(self, inputs):
        # remove something from background  --> background_edited as background, foreground as foreground
        print("\n-------- run_case_remove_from_background ----------\n")
        image_name, to_be_replaced_txt, replace_with_txt = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
            inputs.split(",")[2].strip(),
        )
        print(image_name)
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(
            image_name, func_name="Remove", objects=to_be_replaced_txt, save_merge=False
        )
        edited_image, _ = self.replace_sth_pipe(
            image_path_back, new_image_name, to_be_replaced_txt, replace_with_txt
        )
        edited_image.save(new_image_path_back)
        save_same_atlas_diff_name(image_path_fore, new_image_path_fore)
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )
        print(
            f"\nProcessed ObjectRemove, Input Image: {image_name}.png, Remove {to_be_replaced_txt}, Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"

    @prompts(
        name="Replace Something From The Scene",
        description="useful when you want to replace an object from the object description with another object from its description. "
        "like: replace the flower to a sunflower. "
        "The input to this tool should be a comma separated string of three, "
        "representing the scene_path, the object to be replaced, the object to be replaced with.",
    )
    def inference_replace_sam(self, inputs):  # remove + text-driven inpainting
        image_name, to_be_replaced_txt = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
        )
        print(image_name)
        _, _, _, area, _, _, _, _ = frontend_2_backend(
            image_name, func_name="Replace", objects=to_be_replaced_txt
        )
        if area == "f":
            return self.run_case_replace_sth_in_foreground(inputs)
        else:
            return self.run_case_replace_sth_in_background(inputs)

    def run_case_replace_sth_in_foreground(self, inputs):
        # Also can using pix2pix for workaround
        # merge -- > relace in merge --> split merge, then splited_foreground as foreground. remain ori background --> mask_of_new_obj=True
        print("\n-------- run_case_replace_sth_in_foreground ----------\n")
        image_name, to_be_replaced_txt, replace_with_txt = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
            inputs.split(",")[2].strip(),
        )
        print(image_name)
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(
            image_name, func_name="Replace", objects=to_be_replaced_txt
        )
        edited_image, mask = self.replace_sth_pipe(
            merge_atlas_path, new_image_name, to_be_replaced_txt, replace_with_txt
        )
        edited_image.save(merge_edited_atlas_path)
        save_same_atlas_diff_name(image_path_back, new_image_path_back)

        img_f_remain_after_mask = self.get_fore_remain_after_mask(image_path_fore, mask)
        image_pil, image = self.grounding.load_image(merge_edited_atlas_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(
            image, replace_with_txt
        )
        image = cv2.cvtColor(cv2.imread(merge_edited_atlas_path), cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu()  # tensor
        mask_of_new_obj = self.pad_edge(mask, padding=1)  # numpy
        mask_image = Image.fromarray(mask_of_new_obj)
        mask_image.resize(image_pil.size).save(
            f"./backend_images/{new_image_name}_mask_OfNewObj.png"
        )
        # updated_image = (cv2.imread(save_image_path) * (mask_of_new_obj[:, :, None] / 255.)).astype(np.uint8)
        new_image_fore = (
            (
                cv2.imread(merge_edited_atlas_path)
                * (mask_of_new_obj[:, :, None] / 255.0)
                + img_f_remain_after_mask
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        cv2.imwrite(new_image_path_fore, new_image_fore)

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
            True,
        )
        print(
            f"\nProcessed ObjectReplace, Input Image: {image_name}.png, Replace {to_be_replaced_txt} to {replace_with_txt}, Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"

    def run_case_replace_sth_in_background(self, inputs):
        # replace in background -- > replaced as background, foreground as foreground
        print("\n-------- run_case_replace_sth_in_background ----------\n")
        image_name, to_be_replaced_txt, replace_with_txt = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
            inputs.split(",")[2].strip(),
        )
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
            merge_atlas_path,
            merge_edited_atlas_path,
        ) = MergeEditSplitPipe.merge_atlas(
            image_name,
            func_name="Replace",
            objects=to_be_replaced_txt,
            save_merge=False,
        )
        edited_image, _ = self.replace_sth_pipe(
            image_path_back, new_image_name, to_be_replaced_txt, replace_with_txt
        )
        edited_image.save(new_image_path_back)
        save_same_atlas_diff_name(image_path_fore, new_image_path_fore)
        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )
        print(
            f"\nProcessed ObjectReplace, Input Image: {image_name}.png, Replace {to_be_replaced_txt} to {replace_with_txt}, Output Image: {new_image_name}.png"
        )
        return f"{new_image_name}.png"


class BackgroundRemoveOrExtractObject:
    template_model = True

    def __init__(
        self,
        VisualQuestionAnswering: VisualQuestionAnswering,
        Text2Box: Text2Box,
        Segmenting: Segmenting,
    ):
        self.obj_segmenting = ObjectSegmenting(Text2Box, Segmenting)

    @prompts(
        name="Remove the background",
        description="useful when you want to only remove the background, like: 'remove the background'. "
        "the input should be a string scene_path",
    )
    def inference(self, image_path):
        image_name = image_path.split(".")[0]
        print(image_name)
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
        ) = frontend_2_backend(image_name, "foreground", "", "RemoveBackground")
        for image_path in image_paths_to_edit:
            mask = self.get_mask(image_path)
            image = Image.open(image_path).convert("RGB")
            mask = Image.fromarray(mask)
            image.putalpha(mask)
            image = PIL_rgba2rgb(image)
            save_image_path = (
                new_image_path_back
                if "background" in image_path
                else new_image_path_fore
            )
            image.save(save_image_path)

        if area == "f":
            set_zero_atlas(image_path_back, new_image_path_back)
        if area == "b":
            set_zero_atlas(image_path_fore, new_image_path_fore)

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        return f"{new_image_name}.png"

    @prompts(
        name="Extract an object",
        description="useful when you want to extract an object from the Scene, like 'extract the statue'. "
        "The input to this tool should be a comma separated string of two, "
        "representing the scene_path, and the object to be extracted.",
    )
    def inference_extract(self, inputs):
        image_name, extract_obj = (
            inputs.split(",")[0].split(".")[0].strip(),
            inputs.split(",")[1].strip(),
        )
        print(image_name)
        (
            image_path_fore,
            image_path_back,
            image_paths_to_edit,
            area,
            new_image_name,
            new_image_path_fore,
            new_image_path_back,
            _,
        ) = frontend_2_backend(image_name, extract_obj, "", "Extract")
        for image_path in image_paths_to_edit:
            mask = self.get_mask(image_path, extract_obj)
            image = Image.open(image_path).convert("RGB")
            mask = Image.fromarray(mask)
            image.putalpha(mask)
            save_image_path = (
                new_image_path_back
                if "background" in image_path
                else new_image_path_fore
            )
            image = PIL_rgba2rgb(image)
            image.save(save_image_path)

        if area == "f":
            set_zero_atlas(image_path_back, new_image_path_back)
        if area == "b":
            set_zero_atlas(image_path_fore, new_image_path_fore)

        Atlas2frames(
            training_atlas_results,
            frames_folder,
            mask_folder,
            new_image_path_fore,
            new_image_path_back,
            editing_space,
            f"./frontend_images/{new_image_name}.png",
        )

        return f"{new_image_name}.png"

    def get_mask(self, image_path, extract_obj="main_obj"):
        """
        given an image path, return the mask of the extract object.
        """
        if extract_obj == "main_obj":
            vqa_input = f"{image_path}, what is the main object in the image?"
            text_prompt = global_vqa.inference(vqa_input)
        else:
            text_prompt = extract_obj
        mask = self.obj_segmenting.get_mask(image_path, text_prompt)

        return mask


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if (
            "ImageCaptioning" not in load_dict
            and "VisualQuestionAnswering" not in load_dict
        ):
            raise ValueError(
                "You have to load ImageCaptioning and VisualQuestionAnswering as basic functions for VisualChatGPT"
            )

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        # Load Template Foundation Models.
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {
                    k
                    for k in inspect.signature(module.__init__).parameters.keys()
                    if k != "self"
                }
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names}
                    )
        print(f"All the Available Functions: {self.models}")
        if args.proxy:
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )
        self.llm = OpenAI(
            temperature=0,
            model_name=args.LLM_model_name,
            request_timeout=5000,
            max_tokens=4096,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )

    def init_agent(self, lang):
        self.memory.clear()  # clear previous history
        if lang == "English":
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = (
                VISUAL_CHATGPT_PREFIX,
                VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                VISUAL_CHATGPT_SUFFIX,
            )
            place = "Enter the chat text"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = (
                VISUAL_CHATGPT_PREFIX_CN,
                VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN,
                VISUAL_CHATGPT_SUFFIX_CN,
            )
            place = "输入聊天文字"
            label_clear = "清除"

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": PREFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
                "suffix": SUFFIX,
            },
        )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(placeholder=place),
            gr.update(value=label_clear),
        )

    def run_text(self, text, state, index):
        res = self.agent(
            {"input": text.strip()}
        )  # Core Core Core process: using LLM agent to edit scene
        res["output"] = res["output"].replace("\\", "/")
        response = re.sub(
            "([-\w]*.png)",
            lambda m: f"*{m.group(0).replace('.png', '.scn')}:* ![](file=frontend_images/{m.group(0)}#w50)",
            res["output"],
        )
        response = response.replace(").", ")")
        response = response.replace(
            "\n```", ""
        )  # sometimes, chatGPT will return this in sentence, dont konw why...
        state = state + [(text, response)]
        print(
            f"\nProcessed run_text, Input text: {text}\n Current state: {state}\n Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, self.update_display_frames(index)[-1]

    def update_display_frames(self, index):
        if index > args.slider_show_max:
            index = args.slider_show_max
        return (
            global_display_original_frames[index],
            global_display_edited_frames[index],
        )

    def train_TRF(self):
        # work_space like: '/data/llff/flower/work_space'
        args.basedir = f"{work_space}/TensoRF_results"
        args.datadir = os.path.dirname(work_space)
        args.edited_img_folder = f"{work_space}/editing_space/frames"
        args.expname = os.path.basename(args.basedir)
        args.no_dir = True  # disable direction
        args.n_iters = 10000
        reconstruction(args)
        return f"{args.basedir}/imgs_path_all/video.mp4"

    def run_workspace_init(self, text, state, lang):
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
        |  |final_recoverd_frames_from_atlas
        |  |editing_space
        |  |  |turn_into_autumn
        |  |  |remove_flower
        |-frontend_images  # for interface displayed images
        |  |abc.png
        |-backend_images  # for real edited images of atlas
        |  |abc_foreground_atlas.png
        |  |abc_background_atlas.png
        """
        global work_space, atlas_space, mask_folder, flow_folder, editing_space, training_atlas_results, frames_folder, global_foreground_mask
        global global_display_original_frames, global_display_edited_frames

        folder_path_3D = text
        folder_path_3D_list = os.listdir(folder_path_3D)
        ori_img_path_name = (
            "images_8" if "images_8" in folder_path_3D_list else "images_2"
        )
        frames_folder = os.path.join(folder_path_3D, ori_img_path_name)
        ori_img_names = sorted(os.listdir(frames_folder))
        ori_frame_paths = [os.path.join(frames_folder, p) for p in ori_img_names]
        global_display_original_frames = deepcopy(ori_frame_paths)

        atlas_space = os.path.join(folder_path_3D, "work_space", "atlas_space")
        work_space = os.path.join(folder_path_3D, "work_space")
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

        final_atlas_path = os.path.join(folder_path_3D, "work_space", "final_atlas")
        os.makedirs(final_atlas_path, exist_ok=True)

        editing_space = os.path.join(folder_path_3D, "work_space", "editing_space")
        os.makedirs(os.path.join(editing_space, "frames"), exist_ok=True)
        #  using VQA to get the foreground and background
        foreground = self.models["VisualQuestionAnswering"].inference(
            f"{ori_frame_paths[0]},what are the main objects of this image?"
        )  # .split(' ')[-1]
        background = self.models["VisualQuestionAnswering"].inference(
            f"{ori_frame_paths[0]},what are the background of this image?"
        )  # .split(' ')[-1]
        print(
            f"\n======>the scene's foreground is {foreground}, and background is {background}"
        )

        if not os.path.isfile(
            os.path.join(final_atlas_path, "atlas_ori_foreground.png")
        ):
            # step1: get the foreground mask and save to mask_folder
            for i, image_path in enumerate(ori_frame_paths):
                mask_save_path = os.path.join(mask_folder, ori_img_names[i])
                self.models["ObjectRemoveOrReplace"].get_mask_by_sam(
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
            sys.path.insert(0, "./model_zoos/hash_atlas")
            from hash_atlas.train_atlas import train_atlas

            train_atlas(frames_folder)

        global_display_edited_frames = [
            p.replace(ori_img_path_name, "work_space/editing_space/frames")
            for p in global_display_original_frames
        ]
        global_display_original_frames = [
            p.replace(ori_img_path_name, "work_space/final_recoverd_frames_from_atlas")
            for p in global_display_original_frames
        ]
        for i in range(len(global_display_original_frames)):  # At the beginning, the original and edited are the same
            cmd = f"cp {global_display_original_frames[i]} {global_display_edited_frames[i]}"
            os.system(cmd)

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
        if args.clean_FBends:
            os.system("rm ./frontend_images/*.png")
            os.system("rm ./backend_images/*.png")
        uuid_current = get_current_time()  # str(uuid.uuid4())[:8]

        frontend_img_path = os.path.join("frontend_images", f"{uuid_current}.png")
        image_filename = os.path.basename(frontend_img_path).replace(".png", ".scn")
        img.save(frontend_img_path, "PNG")
        foreground_atlas = Image.open(
            os.path.join(final_atlas_path, "atlas_ori_foreground.png")
        ).resize((512, 512))
        foreground_atlas.save(
            os.path.join("backend_images", f"{uuid_current}_atlas_foreground.png"),
            "PNG",
        )
        background_atlas = Image.open(
            os.path.join(final_atlas_path, "atlas_ori_background.png")
        ).resize((512, 512))
        background_atlas.save(
            os.path.join("backend_images", f"{uuid_current}_atlas_background.png"),
            "PNG",
        )
        global_foreground_mask = (
            cv2.cvtColor(
                cv2.resize(
                    cv2.imread(
                        os.path.join(
                            "backend_images", f"{uuid_current}_atlas_foreground.png"
                        )
                    ),
                    (512, 512),
                ),
                cv2.COLOR_BGR2GRAY,
            )
            > 2
        )
        global_foreground_mask = pad_numpy_edge(global_foreground_mask, 2)

        description = self.models["ImageCaptioning"].inference(frontend_img_path)

        if lang == "Chinese":
            Human_prompt = f'\nHuman: 提供名为 {image_filename}的场景。它的描述是: {description}。这些信息帮助你理解这个场景，但是你应该使用工具来完成后续任务，而不是直接从该描述中想象。如果你明白了, 说 "收到". \n'
            AI_prompt = "🤗 收到。 "
        else:
            Human_prompt = f'\nHuman: provide a scene named {image_filename}. The description is: {description}. This information helps you to understand this scene but you should use tools to finish following tasks, rather than directly imagine from the description. If you understand, say "Received". \n'
            AI_prompt = "🤗 Received. "
        # self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        # state = state + [(f"![](file={frontend_img_path}#w50)*{image_filename}*", AI_prompt)]
        state = state + [
            (f"![](file={frontend_img_path}#w50)*{image_filename}*", AI_prompt)
        ]
        # state = state + [(f"![](./{frontend_img_path})*{image_filename}*", AI_prompt)]
        print(
            f"\nInput scene: {frontend_img_path.replace('.png', '.scn')}\nCurrent state: {state}\n",
            f"Current Memory: {self.agent.memory.buffer}",
        )
        return (
            state,
            state,
            gr.update(maximum=len(global_display_original_frames)),
            self.update_display_frames(0)[0],
        )


if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        type=str,
        default="ImageCaptioning_cuda:0,Text2Image_cuda:0",
        help="the visual models that you want to use",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="the WebUI will be in: http://localhost:7861",
    )
    parser.add_argument(
        "--slider_show_max",
        default=10,
        help="select number of views of the frames and edited frames in slider for showing",
    )
    parser.add_argument(
        "--clean_FBends",
        action="store_true",
        help="enable this para, the frontend and backend paths will be cleared",
    )
    parser.add_argument(
        "--LLM_model_name",
        type=str,
        default="gpt-4-1106-preview",
        help="the Large language model. also like: [gpt-4o-mini-2024-07-18, gpt-4-turbo]",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default="",
        help="When you need to access the internet scientifically, please set up a proxy",
    )
    parser.add_argument(
        "--api_base", type=str, default="", help="You cane specify an opanai.api_base"
    )

    parser = config_parser_ProPainter(exist_parser=parser)
    parser = config_parser_TRF(exist_parser=parser)

    global args
    args = parser.parse_args()
    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
    if args.api_base:
        openai.api_base = args.api_base

    global global_vqa, global_inpaint, global_img_caption
    global_vqa = VisualQuestionAnswering(device="cuda:0")
    global_inpaint = Inpainting(device="cuda:0")
    global_img_caption = ImageCaptioning(device="cuda:0")

    load_dict = {
        e.split("_")[0].strip(): e.split("_")[1].strip() for e in args.load.split(",")
    }
    bot = ConversationBot(load_dict=load_dict)

    # tutorials：https://www.gradio.app/guides/blocks-and-event-listeners
    # with gr.Blocks(css="#chatbot .overflow-y-auto{max-height: 150vh}") as demo:
    with gr.Blocks(css=CSS) as demo:
        gr.HTML(INTRO)
        lang = gr.Radio(
            choices=["Chinese", "English"], value=None, label="Language"
        )  # select english or chinese
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="Chat Window", layout="panel"
                )  # , value=[["🤗 Hi"]])
                with gr.Row():
                    with gr.Column(scale=0.9, min_width=0):
                        txt = gr.Textbox(
                            show_label=False,
                            placeholder="Enter your chat text",
                            container=False,
                        )  # .style(container=False)
                    with gr.Column(scale=0.1, min_width=0):
                        clear = gr.Button("Clear")
            with gr.Column(scale=0.02, min_width=0):
                pass
            with gr.Column(scale=0.28, min_width=0):
                txt_dir = gr.Textbox(
                    show_label=False,
                    placeholder="Enter the 3D scene path first",
                    container=False,
                )  # .style(container=False)
                display_original_frames = gr.Image(
                    label="Original Frames",
                    interactive=False,
                    elem_id="original_frames",
                )
                display_edited_frames = gr.Image(
                    label="Edited Frames", interactive=False, elem_id="edited_frames"
                )
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=args.slider_show_max,
                    step=1,
                    label="Frame Index",
                    elem_id="frame_slider",
                )
                display_trained_video = gr.Video(
                    label="Training Result", elem_id="training_result"
                )
                train_TRF = gr.Button(
                    value="Train NeRF model for this edited scene",
                    elem_id="train_nerf_model",
                )

        lang.change(
            fn=bot.init_agent, inputs=[lang], outputs=[input_raws, lang, txt, clear]
        )
        txt_dir.submit(
            bot.run_workspace_init,
            [txt_dir, state, lang],
            [chatbot, state, frame_slider, display_original_frames],
        )
        # the first [] is the input params of the func run_text，the second [] is the return/output param of the func run_text
        txt.submit(
            bot.run_text,
            [txt, state, frame_slider],
            [chatbot, state, display_edited_frames],
        )
        txt.submit(lambda: "", None, txt)
        frame_slider.change(
            bot.update_display_frames,
            inputs=[frame_slider],
            outputs=[display_original_frames, display_edited_frames],
        )
        train_TRF.click(bot.train_TRF, outputs=[display_trained_video])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="0.0.0.0", server_port=args.port, allowed_paths=["/"])
