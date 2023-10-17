import torch
from kornia.geometry.linalg import inverse_transformation
import torch.nn.functional as F


def compute_uv(hand_frames, obj_verts, obj_parts):
    B, N, P = obj_verts.shape[0], obj_verts.shape[1], hand_frames.shape[1]
    inv_hand_frames = inverse_transformation(hand_frames.reshape(-1, 4, 4))
    part_label = F.one_hot(obj_parts.reshape(-1), num_classes=P).reshape(B, N, P).transpose(1, 2)
    obj_verts = obj_verts.unsqueeze(dim=1).expand(B, P, N, 3).reshape(-1, N, 3)
    local_verts = torch.bmm(obj_verts, inv_hand_frames[:, :3, :3].transpose(1, 2)) + inv_hand_frames[:, None, :3, 3]
    local_verts = local_verts.reshape(B, P, N, 3)
    local_verts = (part_label[:, :, :, None] * local_verts).sum(dim=1)
    uv_pred = local_verts / (torch.norm(local_verts, dim=2, keepdim=True) + 1e-10)
    return uv_pred


def compute_uv_loss(pred_uv, target_uv, weight=None):
    loss = 1 - torch.cosine_similarity(pred_uv, target_uv, dim=-1)
    if weight is not None:
        loss = loss * weight    
    return loss.sum(dim=1).mean(dim=0)




