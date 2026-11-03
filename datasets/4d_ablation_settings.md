# 4D Hash-Atlas: Experimental Settings and Ablation Results

This document provides the datasets, scene selections, implementation settings, and quantitative results for the six-factor study of 4D Hash-Atlas reconstruction in **Table 5** of *Chat-Edit-3D++: Interactive 3D and 4D Scene Editing via Large Language Models*.

## 1. Datasets and scenes

We evaluate the 4D Hash-Atlas on six dynamic scenes drawn from two datasets.

| Dataset | Scenes | Experiments |
| --- | --- | --- |
| [LOVEU-TGVE 2023](https://arxiv.org/abs/2310.16003) | `cat-dog-play`, `dirt-road-driving`, `squirrel-climb` | Atlas resolution, tracking confidence, pivot-view selection |
| LOVEU-TGVE 2023 | `dirt-road-driving`, `squirrel-climb` | Video length |
| [DynamicNeRF](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Dynamic_View_Synthesis_From_Dynamic_Monocular_Video_ICCV_2021_paper.pdf) | `umbrella`, `Balloon1`, `Jumping` | Number of moving objects, dynamic-scene complexity |

The atlas-resolution, tracking-confidence, and pivot-selection studies use **three scenes** each; the video-length study uses **two scenes**. The moving-object-count and scene-complexity analyses use **three DynamicNeRF scenes**.

## 2. Frame sampling and dynamic-object tracking

**Video frames.** We extract consecutive frames from each input video without temporal subsampling. For the video-length experiment, a setting of $N$ frames uses the first $N$ frames of the sequence, with $N \in \{10,30,60,120\}$. Video length is reported in frames.

**Pivot frame.** The proposed pivot frame is chosen as the frame in which the foreground object has the largest visible area, measured by the number of foreground-mask pixels. The pivot-selection experiment compares this strategy with choosing the first, middle, or last frame of the sequence.

**Point tracking.** We sample keypoints on the moving foreground object in the pivot frame and track them across surrounding frames with CoTracker3. The resulting point correspondences are used by the pivot-motion loss to encourage the same moving point to map to a consistent location in atlas space.

**Tracking confidence.** The score used to filter a tracked point at a given frame is

$$
\text{joint score} = \text{visibility} \times \text{confidence}.
$$

We evaluate three thresholds: **0.40 (Low)**, **0.60 (Medium)**, and **0.80 (High)**.

## 3. Hash-Atlas implementation

The general Hash-Atlas configuration follows Section 4.1 of the paper.

| Component | Setting |
| --- | --- |
| Atlas resolution | $1000 \times 1000$ pixels (except in the resolution study) |
| Mapping network $F_m$ | 8-layer MLP, hidden width 256, ReLU activations |
| Atlas appearance network $F_h$ | Multi-resolution hash encoding followed by a 2-hidden-layer MLP of width 64 |
| Hash encoding | Base resolution 16; 16 levels; per-level scale 1.5; hash table size $2^{15}$; 2 feature channels per level |
| $F_h$ activation | ReLU in hidden layers, tanh at output |
| Optimizer | AdamW |
| Initial learning rate for $F_m$ | $10^{-3}$ |
| Initial learning rate for $F_h$ | $10^{-2}$ |
| Optimization | 100,000 iterations; 10,000 sampled pixels per iteration |
| Position pretraining $L_{\mathrm{pos}}$ | First 1,000 iterations |
| Foreground-alpha pretraining $L_{\alpha}$ | First 30,000 iterations |
| Dynamic correspondences | CoTracker3, with a pivot-motion consistency loss |

## 4. Ablation settings and quantitative results

Reconstruction quality is measured using **peak signal-to-noise ratio (PSNR, dB)**. The following values reproduce the results reported in Table 5.

### 4.1 Atlas resolution

**Scenes:** `cat-dog-play`, `dirt-road-driving`, `squirrel-climb` (LOVEU-TGVE).

| Atlas resolution | PSNR (dB) |
| --- | ---: |
| $250 \times 250$ | 25.17 |
| $500 \times 500$ | 27.59 |
| $1000 \times 1000$ | 28.21 |
| $1500 \times 1500$ | 28.35 |

PSNR improves as the atlas resolution increases, with diminishing gains at higher resolutions. We adopt **$1000 \times 1000$** as the default resolution to balance reconstruction quality and computational cost.

### 4.2 Video length

**Scenes:** `dirt-road-driving`, `squirrel-climb` (LOVEU-TGVE).

Each experiment reconstructs the atlas from the first $N$ consecutive video frames.

| Video length (frames) | PSNR (dB) |
| --- | ---: |
| 10 | 33.64 |
| 30 | 32.35 |
| 60 | 29.82 |
| 120 | 27.46 |

Reconstruction quality decreases over longer sequences, which contain more extensive motion and pose changes and place greater demands on temporal point correspondences.


### 4.3 Number of moving objects

**Scenes:** `umbrella`, `Balloon1`, `Jumping` (DynamicNeRF).

| Moving objects | PSNR (dB) |
| --- | ---: |
| 1 | 28.85 |
| 2 | 28.02 |
| 3 | 26.46 |

The results characterize reconstruction performance under different dynamic-scene conditions. Scenes containing more moving objects present additional challenges for object separation, tracking, and atlas construction.

### 4.4 Tracking-confidence threshold

**Scenes:** `cat-dog-play`, `dirt-road-driving`, `squirrel-climb` (LOVEU-TGVE).

| Setting | Joint-score threshold | PSNR (dB) |
| --- | ---: | ---: |
| Low | 0.40 | 25.32 |
| Medium | 0.60 | 28.21 |
| High | 0.80 | 27.94 |

The **0.60** threshold yields the best PSNR among the tested settings. Confidence filtering changes both the reliability and the coverage of retained point correspondences; overly strict filtering can discard useful tracks. The results indicate that the tracking confidence threshold affects atlas reconstruction quality, reflecting a trade-off between correspondence reliability and coverage.


### 4.5 Pivot-view selection

**Scenes:** `cat-dog-play`, `dirt-road-driving`, `squirrel-climb` (LOVEU-TGVE).

| Pivot-view strategy | PSNR (dB) |
| --- | ---: |
| First frame | 28.02 |
| Middle frame | 27.93 |
| Last frame | 27.86 |
| Proposed pivot (largest visible foreground area) | **28.21** |

Selecting the frame with the largest visible foreground area improves the reported PSNR by **0.19 dB**, **0.28 dB**, and **0.35 dB** relative to selecting the first, middle, and last frame, respectively.

### 4.6 Dynamic-scene complexity

**Scenes:** `umbrella`, `Balloon1`, `Jumping` (DynamicNeRF).

| Complexity level | Scene characteristics | PSNR (dB) |
| --- | --- | ---: |
| Level 1 | Relatively simple dynamic motion | 29.32 |
| Level 2 | More substantial occlusions | 27.75 |
| Level 3 | Large motion and multiple independently moving objects | 24.77 |

This cross-scene comparison illustrates the increasing difficulty of constructing stable atlas representations under complex motion and occlusion.

## 5. Summary

The experiments highlight three practical aspects of 4D Hash-Atlas construction: **atlas resolution** determines the amount of texture detail that can be preserved; **tracking and pivot selection** influence the quality of the foreground mapping; and **longer or more complex dynamic scenes** make temporally consistent atlas reconstruction more challenging. We use a $1000 \times 1000$ atlas and the largest-visible-area pivot strategy in the standard configuration.

---

**Reference:** *Chat-Edit-3D++: Interactive 3D and 4D Scene Editing via Large Language Models*, Section 4.5, Table 5.  
**Repository:** https://github.com/Fangkang515/CE3D
