from .loss import ClipLoss2D, ClipLoss3D


def create_loss(num_modalities: int):
    if num_modalities == 2:
        return ClipLoss2D
    elif num_modalities == 3:
        return ClipLoss3D
    else:
        raise ValueError(f'Unsupported number of modalities: {num_modalities}')