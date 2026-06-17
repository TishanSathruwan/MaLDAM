"""
Utilities for VICRegL 
"""

import torch
import torch.distributed as dist
import torch.nn as nn

class MLP(nn.Module):
    """
        Multi-layer Perceptron (MLP) module
    """
    def __init__(self, mlp, embedding, norm_layer):
        super().__init__()
        mlp_spec = f"{embedding}-{mlp}"
        f = list(map(int, mlp_spec.split("-")))
        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            if norm_layer == "batch_norm":
                layers.append(nn.BatchNorm1d(f[i + 1]))
            elif norm_layer == "layer_norm":
                layers.append(nn.LayerNorm(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """ forward pass """
        return self.mlp(x)

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        """ forward pass """
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def batch_all_gather(x):
    """ perform full-size batch gather """

    if not dist.is_available() or not dist.is_initialized():
        return x

    x_list = FullGatherLayer.apply(x.contiguous())
    return torch.cat(x_list, dim=0)

def gather_center(x):
    """ gather and center features """
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batched_index_select(input, dim, index):
    """ select elements from input along dimension dim using index """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    """ select nearest neighbours  """
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    _, sorted_values_indices = torch.sort(topk_values, dim=1)
    _, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def neirest_neighbores_on_location(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)

def exclude_bias_and_norm(p):
    """ Exclude bias and norm parameters """
    return p.ndim == 1

def off_diagonal(x):
    """ Extract off-diagonal elements from a square matrix """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res