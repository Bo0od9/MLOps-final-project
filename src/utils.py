from typing import Tuple

import torch


def get_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Creates a boolean mask for variable-length sequences.

    Parameters
    ----------
    lengths : torch.Tensor
        1D tensor of shape (batch_size,) containing the actual length of each sequence
        in the batch.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (batch_size, max_seq_len) where True indicates a valid
        element and False indicates padding.
    """
    lengths = lengths.to(dtype=torch.long)
    max_len = lengths.max().item()
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)

    return mask


def get_last(data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Extracts the last valid element from each sequence in a flattened batch.

    Parameters
    ----------
    data : torch.Tensor
        Tensor of shape (total_elements, ...) containing flattened sequences concatenated
        sequentially.
    lengths : torch.Tensor
        1D tensor of shape (batch_size,) containing the length of each sequence.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, ...) containing the last elements of each sequence.
    """
    lengths = lengths.to(device=data.device, dtype=torch.long)
    idx = torch.cumsum(lengths, dim=0) - 1
    return data.index_select(0, idx)


def create_masked_tensor(data: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a batch of variable-length sequences into a padded tensor and corresponding mask.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor containing flattened sequences.
    lengths : torch.Tensor
        1D tensor of sequence lengths.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - padded_tensor: Padded tensor.
        - mask: Boolean mask where True indicates valid elements.
    """
    lengths = lengths.to(device=data.device, dtype=torch.long)
    batch_size = lengths.numel()
    max_len = lengths.max().item()
    total = lengths.sum().item()

    pos = torch.arange(max_len, device=data.device).unsqueeze(0)
    mask = pos < lengths.unsqueeze(1)

    if data.dim() == 1:
        padded = data.new_zeros((batch_size, max_len))
    else:
        padded = data.new_zeros((batch_size, max_len, *data.shape[1:]))

    starts = torch.cumsum(lengths, dim=0) - lengths
    batch_ids = torch.repeat_interleave(torch.arange(batch_size, device=data.device), lengths)
    pos_ids = torch.arange(total, device=data.device) - torch.repeat_interleave(starts, lengths)

    padded[batch_ids, pos_ids] = data

    return padded, mask
