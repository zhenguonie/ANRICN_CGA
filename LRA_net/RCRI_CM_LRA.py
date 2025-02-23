import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

def downsampling(num_sample, contour, LOA=None):
    B, N, C = contour.shape
    contour = contour.contiguous()
    num_index = min(N,num_sample)
    new_index = torch.linspace(0,N-1,num_index).long()
    sampled_points = contour[:, new_index, :]
    if LOA is not None:
        new_LOA = LOA[:, new_index, :]
    else:
        new_LOA = None
    new_index = new_index.unsqueeze(0).expand(B, -1)
    return sampled_points, new_LOA, new_index

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points

def index_in_LGA(num_neighbors, nxyz, new_index):
    B, S = new_index.shape
    new_index_expanded = new_index.unsqueeze(2)  # shape [B, S, 1]
    start_id = new_index_expanded - num_neighbors // 2  # shape [B, S, 1]
    end_id = new_index_expanded + (num_neighbors - num_neighbors // 2)  # shape [B, S, 1]
    range_idx = torch.arange(0, num_neighbors, device=new_index.device).expand(B, S, -1)  # shape [B, S, num_neighbors]
    group_idx = (start_id + range_idx) % nxyz
    return group_idx

def adjust_normals(P,N):
    K = 32
    num_batch, num_p, _ = P.shape
    adjusted_N = N.clone()
    # centroid
    indices = torch.arange(num_p).unsqueeze(0).expand(num_batch, -1)  # (num_batch, num_p)
    indices_K = (indices.unsqueeze(2) + torch.arange(K).unsqueeze(0).unsqueeze(0)) % num_p  # (num_batch, num_p, K)
    selected_points = P[torch.arange(P.shape[0]).unsqueeze(1).unsqueeze(2), indices_K]  # (num_batch, num_p, K, 3)
    centroids = selected_points.mean(dim=2)  # (num_batch, num_p, 3)
    # Calculate the adjustment of normal vectors
    line_vecs = centroids - P  # (num_batch, num_p, 3)
    cross_products = (N[:, :, 0] * line_vecs[:, :, 1]) - (N[:, :, 1] * line_vecs[:, :, 0])  # (num_batch, num_p)
    # Update Normal Vector
    adjusted_N[cross_products > 0] = -N[cross_products > 0]
    return adjusted_N

def compute_LOA_one(group_xyz, weighting=False):
    B, S, N, C = group_xyz.shape
    dists = torch.norm(group_xyz, dim=-1, keepdim=True) # nn lengths
    
    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)
    # eigen_values, vec = M.symeig(eigenvectors=True)
    eigen_values, vec = torch.linalg.eigh(M)
    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def compute_LOA(xyz, weighting=True, num_neighbors = 64, DIR = False):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, num_neighbors, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)
    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)
    #eigen_values, vec = M.linalg.eigh(eigenvectors=True)
    eigen_values, vec = torch.linalg.eigh(M)
    LRA = vec[:,:,:,0]
    if DIR:
        LRA = adjust_normals(xyz,LRA)
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # N 3

def LOAI_features(contour, LOA, sampled_points, new_LOA, idx, global_cm=False):
    B, S, C = sampled_points.shape
    new_LOA = new_LOA.unsqueeze(-1)
    idx_ordered = idx
    epsilon=1e-7
    # Dividing the contour into multiple LGAs by sampling points
    all_LGA_points = index_points(contour, idx_ordered)
    if not global_cm:
        local_LGA_points = all_LGA_points - sampled_points.view(B, S, 1, C)
    else:
        # Treat all data as one LGA
        local_LGA_points = all_LGA_points
    d1 = torch.norm(local_LGA_points, dim=-1, keepdim=True)
    d1_unit = local_LGA_points / d1
    d1_unit[d1_unit != d1_unit] = 0
    d1_norm = index_points(LOA, idx_ordered)
    a1 = torch.matmul(d1_unit, new_LOA)
    a2 =  (d1_unit * d1_norm).sum(-1, keepdim=True)
    a3 = torch.matmul(d1_norm, new_LOA)
    a3 = torch.acos(torch.clamp(a3, -1 + epsilon, 1 - epsilon))  
    D_0 = (a1 < a2)
    D_0[D_0 ==0] = -1
    a3 = D_0.float() * a3
    LGA_inner_vec = local_LGA_points - torch.roll(local_LGA_points, 1, 2)
    LGA_inner_length = torch.norm(LGA_inner_vec, dim=-1, keepdim=True)
    LGA_inner_unit = LGA_inner_vec / LGA_inner_length
    LGA_inner_unit[LGA_inner_unit != LGA_inner_unit] = 0 
    a4 = (LGA_inner_unit * d1_norm).sum(-1, keepdim=True)
    a5 = (LGA_inner_unit * torch.roll(d1_norm, 1, 2)).sum(-1, keepdim=True)
    a6 = (d1_norm * torch.roll(d1_norm, 1, 2)).sum(-1, keepdim=True)
    a6 = torch.acos(torch.clamp(a6, -1 + epsilon, 1 - epsilon))
    D_1 = (a4 < a5)
    D_1[D_1 ==0] = -1
    a6 = D_1.float() * a6
    inner_angle_feat = (d1_unit * torch.roll(d1_unit,1,2)).sum(-1, keepdim=True)
    d2 = torch.acos(torch.clamp(inner_angle_feat, -1 + epsilon, 1 - epsilon))
    RIF_LOA = torch.cat([d1,d2,a1,a2,a3,a4,a5,a6], dim=-1)
    return RIF_LOA, idx_ordered

def Feature_Encoding(num_sample, num_neighbors, contour, LOA, global_cm):
    # Determine if it is a global feature extraction module
    if global_cm:
        device = contour.device
        B, N, C = contour.shape
        S=1
        centroid = torch.mean(contour, dim=1, keepdim=True)
        global_LGA = contour.view(B, 1, N, C)
        global_LGA_local = global_LGA - centroid.view(B, S, 1, C)
        new_LOA = compute_LOA_one(global_LGA_local, weighting=True)
        idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        RIF_LOA, idx_ordered = LOAI_features(contour, LOA, centroid, new_LOA, idx, global_cm=True)
        return None, RIF_LOA, new_LOA, idx_ordered
    else:
        contour = contour.contiguous()
        if LOA is not None:
            LOA = LOA.contiguous()
        # Contour input points are sampled sequentially at equal intervals
        sampled_points, new_LOA, new_index = downsampling(num_sample, contour, LOA)
        # Indexing sample point serial numbers in an LGA
        idx = index_in_LGA(num_neighbors, contour.shape[1], new_index)
        RIF_LOA, idx_ordered = LOAI_features(contour, LOA, sampled_points, new_LOA, idx)
        return sampled_points, RIF_LOA, new_LOA, idx_ordered

class rcri_cm(nn.Module):
    def __init__(self, num_sample, num_neighbors, filter_in_channel, filter, global_cm):
        super(rcri_cm, self).__init__()
        self.num_sample = num_sample
        self.num_neighbors = num_neighbors
        self.global_cm = global_cm

        # Define the network parameters for the feature enhancement module
        self.feature_enhancement_convs = nn.ModuleList()
        self.feature_enhancement_bns = nn.ModuleList()
        RIF_channel = 8
        feature_enhancement_channels = [32, 64]
        for out_channel in feature_enhancement_channels:
            self.feature_enhancement_convs.append(nn.Conv2d(RIF_channel, out_channel, 1))
            self.feature_enhancement_bns.append(nn.BatchNorm2d(out_channel))
            RIF_channel = out_channel

        # Define the network parameters of the Filter
        self.filter_convs = nn.ModuleList()
        self.filter_bns = nn.ModuleList()
        temp_channel = filter_in_channel
        for out_channel in filter:
            self.filter_convs.append(nn.Conv2d(temp_channel, out_channel, 1))
            self.filter_bns.append(nn.BatchNorm2d(out_channel))
            temp_channel = out_channel
    
    def forward(self, contour, LOA, feature_maps):
        # Reshape: transform from [B, C, N] to [B, N, C]
        if feature_maps is not None:  
            feature_maps = feature_maps.permute(0, 2, 1)
        B, N, C = contour.shape
        # Feature Encoding based on LOAI
        sampled_points, RIF_feature_map, new_LOA, idx_ordered = Feature_Encoding(self.num_sample, self.num_neighbors, contour, LOA, self.global_cm)
        # Feature Enhancement
        RIF_feature_map = RIF_feature_map.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.feature_enhancement_convs):
            bn = self.feature_enhancement_bns[i]
            RIF_feature_map =  F.relu(bn(conv(RIF_feature_map)))
        # concatenate the previous layer features
        if feature_maps is not None:
            if idx_ordered is not None:
                grouped_points = index_points(feature_maps, idx_ordered)
            else:
                grouped_points = feature_maps.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_RIF_feature_map = torch.cat([RIF_feature_map, grouped_points], dim=1)
        else:
            new_RIF_feature_map = RIF_feature_map
        # fuse 
        for i, conv in enumerate(self.filter_convs):
            bn = self.filter_bns[i]
            new_RIF_feature_map =  F.relu(bn(conv(new_RIF_feature_map)))
        # Max Pool
        RIF_feature_map = torch.max(new_RIF_feature_map, 2)[0]
        return sampled_points, new_LOA, RIF_feature_map
