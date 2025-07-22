# DepthCLIP: Open-Vocabulary Segmentation on Depth Maps via Contrastive Learning

DepthCLIP is a research framework that explores **open-vocabulary semantic segmentation using depth maps only**, leveraging contrastive learning to align depth-based features with **frozen CLIP text and image embeddings**. This work was developed as part of Ryan Jin’s senior thesis at Yale University (advised by Prof. Alex Wong, Yale Vision Lab).

While CLIP has shown strong zero-shot generalization on RGB imagery, its extension to depth data — crucial for low-light, texture-poor, or RGB-scarce environments — remains underexplored. DepthCLIP bridges this gap with a **ResNet-UNet architecture** and a **hybrid contrastive loss** that combines pixel-text and area-image alignment.

Refer the following pdf for a more comprehensive report.
[DepthCLIP.pdf](https://github.com/user-attachments/files/21361883/Jin_Ryan_490_Report.pdf)

---

## Features

- **Depth-only open-vocabulary segmentation** – No RGB data required.
- **ResNet-based UNet with ASPP** for multi-scale context.
- **Contrastive learning** with:
  - Pixel-text alignment (InfoNCE loss against CLIP text embeddings).
  - Area-image alignment (align object crops with CLIP image embeddings).
  - Smoothness regularization for spatial consistency.
- **Curriculum-based distractor sampling** – progressively harder negative samples.
- **Equivalence-aware evaluation metrics** – handles near-synonymous labels with top-k pixel accuracy and mIoU.
- **Distributed, mixed-precision training** with PyTorch DDP.

---

## Results

Evaluated on the **SUN RGB-D** dataset (depth maps only):

- **85% Top-5 Pixel Accuracy**
- **67% Top-5 mIoU**
- **27% Standard mIoU**

DepthCLIP demonstrates **robust segmentation** in **low-light and texture-poor environments**, narrowing the gap with RGB-based methods under challenging conditions.

Here are some samples of RGB (not fed into the model at inference, here solely for us to understand what the scene is), depth map (fed into model), ground truth segmentation, and predicted segmentation.

<img width="1604" height="806" alt="Sample 4" src="https://github.com/user-attachments/assets/6678ec6f-44fe-4132-a843-0ca78ec4a48b" />

<img width="1604" height="806" alt="Sample 11" src="https://github.com/user-attachments/assets/e6ee3076-aee7-4933-a011-4561334f9727" />


## Motivation for This Research: Impact of Low Light on the Performance of RGB Segmentation Models

<img width="5985" height="1462" alt="sample_004619_variation" src="https://github.com/user-attachments/assets/6d83f1f1-44f0-4214-9ef9-1f711aca7310" />

---

## Model Architecture

The architecture consists of:

- **Input**: 256×256 depth maps  
- **Encoder-Decoder**: ResNet-18 backbone with ASPP  
- **Two contrastive pathways**:
  - Pixel embeddings aligned to candidate text embeddings (CLIP ViT-B/32).
  - Object-level area embeddings aligned to CLIP image embeddings of cropped regions.  

Both pathways are optimized jointly via:

```
L_total = W_t * L_text + W_i * L_image + W_s * L_smooth
```

![Architecture](https://github.com/user-attachments/assets/c31a6e38-6662-494f-bb60-6953be40c68e)


---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/jinryan/DepthCLIP.git
cd DepthCLIP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Key dependencies:  
- `torch`, `torchvision`, `transformers`
- `matplotlib`, `tensorboard`
- `nltk` (for label cleaning)

3. (Optional) Setup SUN RGB-D:
   - Download from [SUN RGB-D](https://rgbd.cs.princeton.edu/).
   - Preprocess using provided scripts to unify labels and generate equivalence sets.

---

## Usage

```bash train_segmentation_model.sh```
Supports **DistributedDataParallel (DDP)** and **mixed-precision** training.

Also performs evaluation and outputs standard, top-k, and equivalence-aware metrics (pixel accuracy, mIoU).

---

## Repository Structure

```
DepthCLIP/src/depth_segmentation_model
├── dataloader.py
├── datasets.py
├── evaluation.py
├── log.py
├── model.py
├── train.py
├── train_util.py
├── validate.py
```

---

## Future Work

- Experiment with **Vision Transformers (ViTs)** (e.g., SegFormer, Mask2Former) for better long-range context.
- Incorporate **ontological hierarchies** (WordNet, ConceptNet) for label reasoning.
- Explore **automatic prompt optimization** (reinforcement learning, multi-prompt ensembling).
- Optimize for **real-time inference** (distillation, quantization, pruning).
- Extend beyond indoor SUN RGB-D to **robotics, outdoor, and AR/VR domains**.

---

## Citation

If you use DepthCLIP in your research, please cite:

```
@misc{jin2025depthclip,
  title={DepthCLIP: Open-Vocabulary Segmentation on Depth Maps via Contrastive Learning},
  author={Ryan Jin},
  year={2025},
  institution={Yale University},
  note={Senior Thesis, advised by Alex Wong}
}
```

---

## License
MIT License.
