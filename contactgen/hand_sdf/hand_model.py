from typing import Optional, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.linalg import transform_points

from .implicit_decoder import ImplicitDecoder


class ArtiHand(nn.Module):

    def __init__(self,
                 model_params: Dict[str, Any],
                 pose_size: int = 4,
                 shape_size: int = 10,
                 pose_deform_projection_bias: bool = False,
                 num_joints: int = 16):
        super(ArtiHand, self).__init__()
        
        self.num_joints = num_joints
        self.pose_size = pose_size

        self.latent_size = shape_size
        self.shape_size_list = self.latent_size
        self.shape_size_list = [self.shape_size_list] * self.num_joints
      
      
        models = []
        for i in range(self.num_joints):
            model_params['latent_size'] = self.pose_size + self.shape_size_list[i]
            models += [ImplicitDecoder(**model_params)]

        self.model = nn.ModuleList(models)
        self.projection = nn.Linear(in_features=self.num_joints * 3,
                                    out_features=self.num_joints * self.pose_size,
                                    bias=pose_deform_projection_bias)

    def forward(self, queries, soft_blend=100.0):
        batch_size, num_joints, num_queries, dim = queries.shape

        res_list = [self.model[i](q.reshape(batch_size * num_queries, -1), 1.0)
                for i, q in enumerate(torch.split(queries, 1, dim=1))]
        res = torch.cat(res_list, dim=-1)
        del res_list
        res_parts = res.reshape(batch_size, num_queries, -1)

        if soft_blend is not None:
            weights = F.softmin(soft_blend * res_parts, dim=-1)
            res = (res_parts * weights).sum(-1, keepdim=True)
        else:
            res = res_parts.min(dim=-1)[0]
        return res, res_parts

    def add_shape_feature(self,
                          queries: torch.FloatTensor,
                          shape_indices: Optional[torch.LongTensor] = None,
                          latent_shape_code: Optional[torch.FloatTensor] = None):
        batch_size, num_joints, num_queries, dim = queries.shape
        assert (shape_indices is None) + (latent_shape_code is None) == 1
        if shape_indices is not None:
            latents = self.latents(shape_indices)
        else:
            latents = latent_shape_code

        latents = latents.unsqueeze(1).expand(-1, num_queries, -1)
        latents = latents.unsqueeze(1).expand(-1, num_joints, -1, -1)
        queries = torch.cat([latents, queries], dim=-1)
        return queries

    def add_pose_feature(self, queries, root, trans):
        batch_size = trans.shape[0]
        num_queries = queries.shape[2]
        num_joints = trans.shape[1]

        root = root.unsqueeze(1).unsqueeze(1).expand(batch_size, num_joints, 1, 3)
        root = root.reshape(-1, 1, 3)
        root = transform_points(trans.reshape(-1, 4, 4), root)
        root = root.reshape(batch_size, num_joints, 1, 3)
        root = root.reshape(batch_size, -1)
        reduced_root = self.projection(root)
        reduced_root = reduced_root.reshape(batch_size, num_joints, 1,
                                                    -1)
        trans_feature = reduced_root.expand(batch_size, num_joints, num_queries, -1)
        queries = torch.cat([trans_feature, queries], dim=-1)
        return queries

    def transform_queries(self, queries, trans):
        batch_size = trans.shape[0]
        num_queries = queries.shape[-2]
        num_joints = trans.shape[1]

        if queries.dim() == 3:
            queries = queries.unsqueeze(1).expand(batch_size, num_joints,
                                                  num_queries, 3)
        queries = queries.reshape(-1, num_queries, 3)
        trans = trans.reshape(-1, 4, 4)
        queries = transform_points(trans, queries)
        queries = queries.reshape(batch_size, num_joints, num_queries, 3)

        return queries