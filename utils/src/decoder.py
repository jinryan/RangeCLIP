import torch
import torch.nn as nn
import torch.nn.functional as F
from . import net_utils

class DepthDecoder(nn.Module):
    """
    UNet decoder for depth map segmentation
    
    Arg(s):
        n_filters : list
            number of filters for each decoder block (in reverse order of encoder)
        embedding_dim : int
            dimension of the encoder's output embedding
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolutions
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """
    def __init__(self,
                 n_filters=[256, 128, 64],
                 encoder_filters=[32, 64, 128, 256, 512],
                 embedding_dim=256,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(DepthDecoder, self).__init__()
        
        activation_func = net_utils.activation_func(activation_func)
        
        # Create upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        # Add a decoder block to transform the embedding to spatial features
        in_channels = embedding_dim
        
        # Create decoder blocks with skip connections
        for i, n_filter in enumerate(n_filters):
            # 
            # For the first layer, we don't have skip connections yet
            if i == 0:
                self.up_blocks.append(
                    DecoderBlock(
                        in_channels,
                        n_filter,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm,
                        use_skip=False
                    )
                )
            else:
                # Use skip connections for other layers
                self.up_blocks.append(
                    DecoderBlock(
                        in_channels,
                        n_filter,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm,
                        use_skip=True,
                        skip_channels=encoder_filters[-i-1]
                    )
                )
            in_channels = n_filter
        
        # Final output layer for segmentation
        self.output_conv = net_utils.Conv2d(
            in_channels,
            embedding_dim, # Output is embedding dimension
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=None
        )
        
    def forward(self, spatial_feature_map, encoder_features, target_shape):
        """
        Forward pass through the decoder
        
        Arg(s):
            spatial_feature_map : torch.Tensor
                feature map from the encoder, shape [B, embedding_dim, H, W]
            encoder_features : list
                list of feature maps from encoder layers
                
        Returns:
            torch.Tensor : segmentation with pixel-wise
        """
        # Skip connections exclude the last encoder feature used for spatial_feature_map
        expected_skips = len(encoder_features) - 1  # last one goes to spatial_feature_map
        assert len(self.up_blocks) <= expected_skips, \
            f"Decoder expects ≤{expected_skips} up blocks, got {len(self.up_blocks)}"

                
        # Start with feature map from the last encoder block
        # We need to reshape the embedding to match the spatial dimensions
        
        x = spatial_feature_map
        encoder_features = encoder_features[::-1]

        for i, up_block in enumerate(self.up_blocks):
            if i == 0:
                x = up_block(x)  # no skip
            else:
                skip_feature = encoder_features[i]
                x = up_block(x, skip_feature)

        
        output = self.output_conv(x)
        output = F.interpolate(output, size=target_shape, mode='nearest')
        
        
        
        output = F.normalize(output, p=2, dim=1)

        return output

class DecoderBlock(nn.Module):
    """
    Decoder block for UNet architecture
    
    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_skip : bool
            if set, then use skip connection from encoder
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=F.leaky_relu,
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_skip=True,
                 skip_channels=0):
        super(DecoderBlock, self).__init__()
        
        self.use_skip = use_skip
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        conv_in_channels = in_channels
        if use_skip:
            conv_in_channels = in_channels + skip_channels
        
        # Convolutional layers
        self.conv1 = net_utils.Conv2d(
            conv_in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm
        )
        
        self.conv2 = net_utils.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm
        )
    
    def forward(self, x, skip=None):
        """
        Forward pass
        
        Arg(s):
            x : torch.Tensor
                input tensor
            skip : torch.Tensor
                skip connection from encoder
                
        Returns:
            torch.Tensor : output feature map
        """
        x = self.upsample(x)
        
        # Concatenate with skip connection if available
        if self.use_skip and skip is not None:
            # Adjust the size if there's a mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x