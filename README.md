# Anomaly score: Evaluating generative models and individual generated images based on complexity and vulnerability

Official repository of SAT | [paper]([https://openaccess.thecvf.com/content/CVPR2024/papers/Hwang_Anomaly_Score_Evaluating_Generative_Models_and_Individual_Generated_Images_based_CVPR_2024_paper.pdf]) [arxiv]([https://arxiv.org/abs/2210.11407](https://arxiv.org/abs/2312.10634))

[Jaehui Hwang](https://j-h-hwang.github.io/), Junghyuk Lee, [Jong-Seok Lee](https://mcml.yonsei.ac.kr/)

Special thanks to Junha Park for his effort in building this repository.

## Introduction

With the advancement of generative models, the assessment of generated images becomes more and more important. Previous methods measure distances between features of reference and generated images from trained vision models. In this paper, we conduct an extensive investigation into the relationship between the representation space and input space around generated images. We first propose two measures related to the presence of unnatural elements within images: complexity, which indicates how non-linear the representation space is, and vulnerability, which is related to how easily the extracted feature changes by adversarial input changes. Based on these, we introduce a new metric to evaluating image-generative models called anomaly score (AS). Moreover, we propose AS-i (anomaly score for individual images) that can effectively evaluate generated images individually. Experimental results demonstrate the validity of the proposed approach.

## Quick start
- See [main.py](./main.py) for how to evaluate generative models and individual images through anomaly score (AS) and AS-i.
```python
import torch
from anomaly_score import AnomalyScore
from anomaly_score_i import AnomalyScore_i

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
model.eval()
model.to(device)

images_real = torch.randn(128, 3, 224, 224).to(device) ### change to reference images (ImageNet, Cifar-10, FFHQ)
images_generated = torch.randn(128, 3, 224, 224).to(device) ### change to generate images by generative models

anomaly_score = AnomalyScore(model)
result = anomaly_score(images_real, images_generated)

print("Evaluate a generative model")
print(result)

anomaly_score_i = AnomalyScore_i(model)
result_i = anomaly_score_i(images_generated)
print("Evaluate an individual image")
print(result_i)
```

## How to cite
```
@inproceedings{hwang2024anomaly,
  title={Anomaly Score: Evaluating Generative Models and Individual Generated Images based on Complexity and Vulnerability},
  author={Hwang, Jaehui and Lee, Junghyuk and Lee, Jong-Seok},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8754--8763},
  year={2024}
}
```
