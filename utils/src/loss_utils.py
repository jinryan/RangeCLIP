import torch


'''
Loss functions for depth completion
'''
EPSILON = 1e-8

def l1_loss(src, tgt, w=None, normalize=False):
    '''
    Computes l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image (output depth)
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image (gt)
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        normalize : [bool]
            if normalize : normalized l1 loss
            else : plain l1 loss
    Returns:
        float : mean l1 loss across batch
    '''

    if w is None:
        w = torch.ones_like(src)

    loss_func = torch.nn.L1Loss(reduction='none')
    loss = loss_func(src, tgt)

    if normalize:
        loss = loss / (torch.abs(tgt) + EPSILON)

    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])

    return torch.mean(loss)

def l2_loss(src, tgt, w=None, normalize=False):
    '''
    Computes l2 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image (output depth)
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image (gt)
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        normalize : [bool]
            if normalize : normalized l2 loss
            else : plain l2 loss
    Returns:
        float : mean l2 loss across batch
    '''

    if w is None:
        w = torch.ones_like(src)

    loss_func = torch.nn.MSELoss(reduction='none')
    loss = loss_func(src, tgt)

    if normalize:
        loss = loss / (torch.pow(tgt, 2) + EPSILON)

    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])

    return torch.mean(loss)
