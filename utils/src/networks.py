import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import net_utils

class DepthEncoder(torch.nn.Module):
    """
    Resnet-like encoder for depth maps
    
    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """
    def __init__(self,
                 n_layer=18,
                 input_channels=1,  # Changed to 1 for depth maps
                 n_filters=[32, 64, 128, 256, 256],
                 embedding_dim=256,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=True):
        super(DepthEncoder, self).__init__()
        
        # Reuse the original ResNetEncoder initialization logic
        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        # Rest of the initialization remains similar to the original code
        activation_func = net_utils.activation_func(activation_func)

        # Encoder blocks
        self.conv1 = net_utils.Conv2d(
            input_channels,
            n_filters[0],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create encoder blocks dynamically
        self.blocks = nn.ModuleList()
        in_channels = n_filters[0]
        for i, (n_filter, n_block) in enumerate(zip(n_filters[1:], n_blocks), 1):
            block_group = []
            for j in range(n_block):
                stride = 2 if j == 0 and i > 1 else 1
                block = resnet_block(
                    in_channels,
                    n_filter,
                    stride=stride,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm
                )
                block_group.append(block)
                in_channels = n_filter * 4 if use_bottleneck else n_filter
            
            self.blocks.append(nn.Sequential(*block_group))

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, embedding_dim)
        )

    def forward(self, x):
        # Ensure input is single-channel
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.max_pool(x)

        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x).view(x.size(0), -1)

        # Project to embedding space
        embedding = self.projection_head(x)
        
        # L2 normalize the embedding
        return F.normalize(embedding, p=2, dim=1)
    
    def compute_loss(self, output_embedding, ground_truth_embedding, temperature=0.1):
        cosine_similarity = output_embedding @ ground_truth_embedding.t() / temperature
        
        labels = torch.arange(output_embedding.size(0), device=output_embedding.device)
        loss_a = F.cross_entropy(cosine_similarity, labels)
        loss_b = F.cross_entropy(cosine_similarity.t(), labels)
        
        total_loss = (loss_a + loss_b) / 2
        
        return total_loss
        

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        self.clip_model = clip_model
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model.config._name_or_path)
        
        # Freeze parameters
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
    
    def forward(self, text):
        inputs = self.clip_tokenizer(text, padding=True, return_tensors="pt")
        text_embedding = self.clip_model.get_text_features(**inputs)
        return F.normalize(text_embedding, dim=-1)
    
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
    
    def forward(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_embedding = self.clip_model.get_image_features(**inputs)
        return F.normalize(image_embedding, dim=-1)
    
    def to(self, device):
        self.clip_model.to(device)
        return self
