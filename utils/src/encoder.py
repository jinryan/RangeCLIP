import torch
import torch.nn as nn
import torch.nn.functional as F
from . import net_utils

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
                 use_batch_norm=False,
                 use_instance_norm=False):
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
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,)

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
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm
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
        
        features = []
        
        x1 = self.conv1(x)
        features.append(x1)
        
        x2 = self.max_pool(x1)
        
        for block in self.blocks:
            x2 = block(x2)
            features.append(x2)
        
        final_feature_map = x2
        pooled = self.global_pool(x2).view(x2.size(0), -1)
        embedding = self.projection_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding, features, final_feature_map
    
    
    def save_encoder(self, checkpoint_path, step, optimizer):
        checkpoint = {
            'train_step': step
        }
        
        if isinstance(self, torch.nn.DataParallel):
            checkpoint['encoder'] = self.module.state_dict()
        else:
            checkpoint['encoder'] = self.state_dict()
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
    
    def restore_encoder(self, restore_path, device, optimizer=None):
        checkpoint = torch.load(restore_path, map_location=device)
        
        if isinstance(self, torch.nn.DataParallel):
            self.module.load_state_dict(checkpoint['encoder'])  # Corrected line
        else:
            self.load_state_dict(checkpoint['encoder'])
        
        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        train_step = checkpoint.get('train_step', 0)
        
        return train_step, optimizer if optimizer else None