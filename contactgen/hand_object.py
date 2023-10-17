import pickle
import torch
from torch.nn import functional as F
import pytorch3d.ops
from pytorch3d.structures import Meshes

from .contact.diffcontact import calculate_contact_capsule
from .contact.opt_utils import compute_uv


class HandObject:
    def __init__(self, device, face_path="assets/closed_mano_faces.pkl", hand_part_label_path="assets/hand_part_label.pkl"):
        with open(face_path, 'rb') as f:
            self.hand_faces = torch.Tensor(pickle.load(f)).unsqueeze(0).to(device)
        with open(hand_part_label_path, 'rb') as f:
            self.hand_part_label = torch.Tensor(pickle.load(f)).long().to(device)

    def forward(self, hand_verts, hand_frames, obj_verts, obj_vn):
        hand_mesh = Meshes(verts=hand_verts, faces=self.hand_faces)
        obj_contact_target, _ = calculate_contact_capsule(hand_mesh.verts_padded(),
                                                          hand_mesh.verts_normals_padded(),
                                                          obj_verts, obj_vn,
                                                          caps_top=0.0005, caps_bot=-0.0015,
                                                          caps_rad=0.003,
                                                          caps_on_hand=False)
        obj_cmap = obj_contact_target
        
        _, nearest_idx, _ = pytorch3d.ops.knn_points(obj_verts, hand_verts, K=1, return_nn=True)
        nearest_idx = nearest_idx.squeeze(dim=-1)
        obj_partition = self.hand_part_label[nearest_idx]

        obj_uv = compute_uv(hand_frames, obj_verts, obj_partition)
        obj_partition = F.one_hot(obj_partition, num_classes=16)

        data_out = {
            "verts_object": obj_verts,
            "feat_object": obj_vn, 
            "contacts_object": obj_cmap, 
            "partition_object": obj_partition,
            "uv_object": obj_uv,
        }
        return data_out
