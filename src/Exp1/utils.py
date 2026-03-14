import copy
import random
from typing import Dict

import numpy as np
import torch


StateDict = Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_mask(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    total = int(mask.sum())
    if total == 0:
        return 0.0, 0, 0
    pred = logits.argmax(dim=1)
    correct = int((pred[mask] == y[mask]).sum())
    return correct / total, correct, total


def clone_state_dict(state_dict: StateDict) -> StateDict:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def zeros_like_state_dict(state_dict: StateDict) -> StateDict:
    return {k: torch.zeros_like(v) for k, v in state_dict.items()}


def add_state_dict(base_state: StateDict, residual_state: StateDict, alpha: float = 1.0) -> StateDict:
    out = clone_state_dict(base_state)
    for key in out:
        out[key] = out[key] + alpha * residual_state[key]
    return out


def weighted_average_state_dicts(state_dicts, weights):
    total_weight = float(sum(weights))
    new_state = copy.deepcopy(state_dicts[0])

    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key])
        for state, weight in zip(state_dicts, weights):
            new_state[key] += state[key] * (weight / total_weight)

    return new_state
