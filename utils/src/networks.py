import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTokenizer
import matplotlib.pyplot as plt
from . import net_utils
from .log_utils import apply_colormap



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

