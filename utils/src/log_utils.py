import os
import torch
import numpy as np
from matplotlib import pyplot as plt


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console

    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def colorize(T, colormap='magma'):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps

    Arg(s):
        T : torch.Tensor[float32]
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)
    shape = T.shape

    # Convert to numpy array and transpose
    if shape[0] > 1:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)))
    else:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap and transpose back
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)
    color = np.transpose(color, (0, 3, 1, 2))

    # Convert back to tensor
    return torch.from_numpy(color.astype(np.float32))

def apply_colormap(tensor, cmap='magma'):
        """
        Converts a single-channel depth map to a 3-channel color image using a colormap.
        """
        tensor = tensor.squeeze(1)  # Remove channel dim: N x H x W
        tensor = tensor - tensor.min()  # Normalize to 0-1
        tensor = tensor / (tensor.max() + 1e-8)  # Avoid division by zero

        tensor_np = tensor.cpu().numpy()  # Convert to NumPy for colormap
        colored_images = []

        for i in range(tensor_np.shape[0]):
            colored = plt.get_cmap(cmap)(tensor_np[i])[:, :, :3]  # Apply colormap and remove alpha
            colored = torch.tensor(colored).permute(2, 0, 1)  # Convert back to Tensor (3 x H x W)
            colored_images.append(colored)

        return torch.stack(colored_images)  # Stack into (N x 3 x H x W)
    

def validate_tensor(tensor, name, threshold_large=1e10, threshold_small=1e-10, log_warnings=True):
    """
    Validates a tensor for numerical issues like NaN, Inf, or extreme values.

    Args:
        tensor: Input tensor to validate
        name: Name of tensor for logging
        threshold_large: Threshold for detecting extremely large values
        threshold_small: Threshold for detecting extremely small non-zero values
        log_warnings: Whether to log warnings

    Returns:
        bool: True if tensor has no numerical issues
    """
    with torch.no_grad():
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()

        large_values = (torch.abs(tensor) > threshold_large).sum()
        small_values = ((torch.abs(tensor) > 0) & (torch.abs(tensor) < threshold_small)).sum()

        if log_warnings and (has_nan or has_inf or large_values > 0 or small_values > 0):
            min_val = tensor.min()
            max_val = tensor.max()
            mean_val = tensor.mean()
            std_val = tensor.std()

            warning_msgs = []
            if has_nan:
                warning_msgs.append("Contains NaN values")
            if has_inf:
                warning_msgs.append("Contains Inf values")
            if large_values > 0:
                warning_msgs.append(f"{large_values.item()} elements have abs value > {threshold_large}")
            if small_values > 0:
                warning_msgs.append(f"{small_values.item()} non-zero elements have abs value < {threshold_small}")

            print(f"WARNING - {name}: {', '.join(warning_msgs)}")
            print(f"Stats: min={min_val.item():.6e}, max={max_val.item():.6e}, "
                  f"mean={mean_val.item():.6e}, std={std_val.item():.6e}")

        return not (has_nan or has_inf or large_values > 0)