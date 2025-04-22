import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTokenizer
import matplotlib.pyplot as plt
from . import net_utils
from .log_utils import apply_colormap

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.branches = nn.ModuleList()

        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3 if rate > 1 else 1,
                              padding=rate if rate > 1 else 0, dilation=rate, bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(len(dilation_rates) * out_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        out = [branch(x) for branch in self.branches]

        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)

        out.append(global_feat)

        out = torch.cat(out, dim=1)
        out = self.project(out)
        out = F.normalize(out, p=2, dim=1)
        return out


class TextEncoder(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        
        self.clip_model = clip_model
        self.clip_model.to(device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model.config._name_or_path)
        self.device = device
        
        # Freeze parameters
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()
    
    def forward(self, text):
        self.clip_model.eval()
        inputs = self.clip_tokenizer(text, padding=True, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        text_embedding = self.clip_model.get_text_features(**inputs)
        
        return F.normalize(text_embedding, p=2, dim=-1)
    
    def to(self, device):
        self.clip_model.to(device)
        return self

class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        self.clip_model = clip_model
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model.config._name_or_path)
        
        # Freeze parameters
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
            
        self.clip_model.eval()
        
    def forward(self, image):
        self.clip_model.eval()
        inputs = self.clip_processor(images=image, return_tensors="pt", do_rescale=False)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
        
        image_embedding = self.clip_model.get_image_features(**inputs)
        return F.normalize(image_embedding, p=2, dim=-1)
    
    def to(self, device):
        self.clip_model.to(device)
        return self

