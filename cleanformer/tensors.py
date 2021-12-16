"""
any constant tensors are defined here.
they will be registered as buffers.
"""
import torch


def subsequent_mask(max_length: int) -> torch.LongTensor:
    """
    :param max_length (L)
    Subsequently allow positions
    :return (L, L)
    """
    ones = torch.ones(size=(max_length, max_length))
    mask = torch.tril(ones, diagonal=0)
    return mask.long()


def pos_encodings(max_length: int, hidden_size: int) -> torch.Tensor:
    """
    :return: (L, H)
    """
    positions = torch.arange(max_length).view(-1, 1)  # (,) -> (L)
    freqs = 0.0001**(torch.arange(hidden_size)[::2] / hidden_size).view(1, -1)  # (,) -> (H)
    encodings = torch.zeros(size=(max_length, hidden_size))  # (L, H)
    # fill in the pairs by broadcast-multiplying freqs to positions
    encodings[:, ::2] = torch.sin(freqs * positions)   # evens = sin
    # odds = cos, but with the same frequency as above
    # why the same frequency?
    # A: so that dist(PE(pos + k) - PE(pos)) stays constant
    encodings[:, 1::2] = torch.cos(freqs * positions)
    return encodings
