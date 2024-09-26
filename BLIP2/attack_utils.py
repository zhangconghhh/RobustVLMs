import torch, os, pdb
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from lavis.models import load_model_and_preprocess


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize(images, device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)  
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device) 
    # images = images/255
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images, device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images
    # return images*255
