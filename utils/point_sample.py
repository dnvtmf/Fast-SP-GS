import torch
from torch import Tensor

from fast_2d_gs._C import try_use_C_extension


@try_use_C_extension
def FurthestSampling(points: Tensor, npoint: int):
    """ farthest point sample
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = points.device
    B, N, C = points.shape
    if N <= npoint:
        return torch.arange(N, dtype=torch.long, device=device).view(-1, N).expand(B, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    # farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    farthest = torch.zeros((B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
