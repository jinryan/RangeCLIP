import numpy as np

def info_nce(src, tgt, temperature=0.1):
    """
    Compute InfoNCE loss between source and target embeddings
    
    Args:
        src : numpy[float32]
            source embeddings
        tgt : numpy[float32]
            target embeddings
        temperature : float, optional
            Temperature parameter to scale similarities (default: 0.1)
    
    Returns:
        float : InfoNCE loss
    """
    cosine_similarities = np.dot(src, tgt.T)
    scaled_similarities = cosine_similarities / temperature
    
    exp_similarities = np.exp(scaled_similarities)
    softmax_probs = exp_similarities / np.sum(exp_similarities, axis=1, keepdims=True)
    
    batch_size = src.shape[0]
    diagonal_mask = np.eye(batch_size, dtype=bool)
    
    positive_log_probs = -np.log(softmax_probs[diagonal_mask])
    
    loss = np.mean(positive_log_probs)
    
    return loss
    

def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))
