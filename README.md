## Interactive 3D/4D Scene Editing via Large Language Model
### :star2: :star2: **Update:**  We have released a new powerful :heart: CE3D++ :heart: framework, which leverages an LLM to integrate dozens of different models to enable various 3D and 4D scene editing during chatting :star2: :star2:

### [Project Page](http://sk-fun.fun/CE3D/) | [Paper](https://arxiv.org/abs/2407.06842) | [Video](https://www.youtube.com/watch?v=btO1Ky9I21s) | [Datasets and Ckpts](https://drive.google.com/drive/folders/1KUIFMgvHeZtKIML-hzzBIjZ-3Kxo4JGt?usp=drive_link) | 

<img src="./assets/chat_example.png" width="800">


# Updates
  - [x] Hash-Atlas codes have been released in the folder of *hash_atlas*
  - [x] Pretrained weights and a dataset example have been released
  - [x] Release LLM-driven dialogue codes
  - [x] Support 4D scene, as well as add more visual tools


# Demo (usage tutorial)

<be>

**You can edit 3D scenes in the same way you interact with ChatGPT**.

<img src="./assets/demo1.gif" width="800">


# Quick Start

**After you have [set up the environment](https://github.com/Fangkang515/CE3D/tree/main?tab=readme-ov-file#step-1-create-a-new-environment--activate-the-new-environment), you only need to run the following command:**
```
# prepare your private OpenAI key.
export OPENAI_API_KEY={Your_Private_Openai_Key}

# running the demo (Advice for two Tesla A100/A800 80GB)
make run-all

# You can also run manually to select the required features and configure the GPUs.
python3 chat_edit_3D.py --port 7862 --clean_FBends --load "Segmenting_cuda:0,\
  ImageCaptioning_cuda:0,Text2Image_cuda:0,VisualQuestionAnswering_cuda:0,\
  Text2Box_cuda:0,Inpainting_cuda:0,InstructPix2Pix_cuda:0,Image2Depth_cuda:0,\
  DepthText2Image_cuda:0,SRImage_cuda:0,Image2Scribble_cuda:1,\
  ScribbleText2Image_cuda:1,Image2Canny_cuda:1,CannyText2Image_cuda:1,\
  Image2Line_cuda:1,LineText2Image_cuda:1,Image2Hed_cuda:1,HedText2Image_cuda:1,\
  Image2Pose_cuda:1,PoseText2Image_cuda:1,SegText2Image_cuda:1,\
  Image2Normal_cuda:1,NormalText2Image_cuda:1,ReferenceImageTextEditing_cuda:1"

```
Then visit http://localhost:7862 on the web to freely edit the 3D scene.

The above operation integrates more than 20 visual models and requires about 100G of GPU memory. If you don't have it, you can run the **small version of CE3D**, which only requires 20G of GPU memory (but the editing ability is limited). The commands are as follows:
```
make run-small-instruct

# You can also run manually:
python3 chat_edit_3D.py --port 7862 --clean_FBends --load "Segmenting_cuda:0,\
ImageCaptioning_cuda:0,VisualQuestionAnswering_cuda:0,Text2Box_cuda:0,\
Inpainting_cuda:0,InstructPix2Pix_cuda:0"
```

#### Step 1: Create a new environment & Activate the new environment
```
https://github.com/Fangkang515/CE3D.git
cd CE3D

conda create -n CE3D python=3.10
conda activate CE3D

pip install -r requirements.txt
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install  git+https://github.com/facebookresearch/segment-anything.git

install tiny-cuda-nn according to https://github.com/NVlabs/tiny-cuda-nn/tree/master

```

#### Step 2: Download the [checkpoints](https://drive.google.com/file/d/1euRnJpn75MP0V_nKKlc6oSoSwwhYwBFP/view?usp=drive_link) / [pretrained-weights](https://drive.google.com/file/d/14FY23C8u9-igNCm1Wo1Mm8JVs35sszw6/view?usp=sharing) / [dataset examples](https://drive.google.com/file/d/1nsWj1La8sTAj88Kbc9f0VSbLOwIyoNtZ/view?usp=drive_link)
```
# optional: Download the checkpoints of Sam, Dino, et al. (Directly running the code will also automatically download this file)
unzip checkpoints.zip

# Download the pretrained-weights (atlas)
cd hash_atlas
unzip weights.zip
cd ..

# Download the dataset example
cd datasets
unzip flower.zip
cd ..
```

#### Tips: If you want to train the Atlas manually 
```
# note: (the final code will be automatically executed to obtain Atlas of a 3D scene)
# 
cd hash_atlas
python atlas_playground.py

# Then You will find an atlas_space folder in the ../datasets/flower directory, like below:
 |-flower   
    |-XXX
    |-work_space
    |  |atlas_space
    |  |  |flow
    |  |  |mask
    |  |  |propainter
    |  |  |training_results
    |  |final_atlas
    |  |  | atlas_ori_foreground.png
    |  |  | atlas_ori_background.png
    |  |editing_space
    |  |  |XXX
```


## Citation

If you find our code or paper useful, please consider citing.
```
@article{fang2024chat,
  title={Chat-Edit-3D: Interactive 3D Scene Editing via Text Prompts},
  author={Fang, Shuangkang and Wang, Yufeng and Tsai, Yi-Hsuan and Yang, Yi and Ding, Wenrui and Zhou, Shuchang and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2407.06842},
  year={2024}
}
```



