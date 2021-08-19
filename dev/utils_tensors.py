import numpy as np
import torch


def tensor2array(tensor):
    """
    Convert a torch tensor to numpy ndarray.

    Parameters
    ----------
    var : torch.Tensor
        Input, to be converted to numpy.

    Returns
    -------
    array : np.ndarray
        Numpy array.

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.is_cuda:
        tensor = tensor.to('cpu')
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.numpy()
    if tensor.dtype == np.uint8:
        # convert byte tensor to actual boolean tensor.
        tensor = tensor.astype(bool)
    return tensor


def array2tensor(array, device=None, cuda=False):
    """
    Convert a numpy ndarray or torch tensor to torch variable.

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Numpy array.

    Returns
    -------
    tensor : torch.Tensor
        Tensor.

    """
    if isinstance(array, torch.Tensor):  # note I checked this also works with CUDA tensors
        tensor = array
    else:
        tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    if cuda:
        tensor = tensor.cuda()
    return tensor