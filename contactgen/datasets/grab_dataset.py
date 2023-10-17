import os
import pickle
import numpy as np
import torch
from torch.utils import data
import trimesh

from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues


class Grab(data.Dataset):
    def __init__(self, 
                 dataset_root='grab_data',
                 ds_name='train',
                 n_samples=2048):
        super().__init__()
        self.ds_name = ds_name
        self.ds_path = os.path.join(dataset_root, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))

        frame_names = np.load(os.path.join(dataset_root, ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = np.asarray([os.path.join(dataset_root, fname) for fname in frame_names])
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.n_samples = n_samples
        
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_root, 'sbj_info.npy'), allow_pickle=True).item()
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        
        with open(os.path.join("assets/closed_mano_faces.pkl"), 'rb') as f:
            self.hand_faces = pickle.load(f)

        self.obj_root = os.path.join(dataset_root, "obj_meshes")
        self.mano_layer = ManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("mano/models"), use_pca=False, joint_rot_mode="rotmat")

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        
    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch

    def __getitem__(self, item):
        obj_name = self.frame_objs[item]
        obj_mesh_path = os.path.join(self.obj_root, obj_name + '.ply')
        obj_mesh = trimesh.load(obj_mesh_path, file_type="ply")
        obj_faces = obj_mesh.faces

        rot_mat = self.ds["root_orient_obj_rotmat"][item].numpy().reshape(3, 3)
        transl = self.ds["trans_obj"][item].numpy()
        obj_verts = obj_mesh.vertices @ rot_mat + transl
        offset = obj_verts.mean(axis=0, keepdims=True)
        obj_verts = obj_verts - offset

        sbj_idx = self.frame_sbjs[item]
        v_template = self.sbj_vtemp[sbj_idx]

        global_orient = self.ds['global_orient_rhand_rotmat'][item]
        rhand_rotmat = self.ds['fpose_rhand_rotmat'][item]

        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)
        th_trans = self.ds['trans_rhand'][item].unsqueeze(dim=0) - torch.FloatTensor(offset)
        th_v_template = v_template.unsqueeze(dim=0)
        hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        obj_verts = obj_verts @ rot_mat.T
        global_orient = torch.from_numpy(rot_mat).float() @ global_orient
        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)
        
        root_center = hand_frames[:, 0, :3, 3]
        th_trans = (root_center[:, None, :] @ rot_mat.T).squeeze(dim=1) - root_center + th_trans
        hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)
        
        if self.ds_name == "train":
            orient = torch.FloatTensor(1, 3).uniform_(-np.pi/6, np.pi/6)
            aug_rot_mats = batch_rodrigues(orient.view(-1, 3)).view([1, 3, 3])
            aug_rot_mat = aug_rot_mats[0]
            obj_verts = obj_verts @ aug_rot_mat.numpy().T
            global_orient = aug_rot_mat @ global_orient
            handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)

            root_center = hand_frames[:, 0, :3, 3]
            th_trans = (root_center[:, None, :] @ aug_rot_mat.T).squeeze(dim=1) - root_center + th_trans
            hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        hand_verts = hand_verts.squeeze(dim=0).float()
        hand_frames = hand_frames.squeeze(dim=0).float()

        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = sample[0].astype(np.float32)
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)

        return {
            "hand_verts": hand_verts,
            "hand_frames": hand_frames,
            "obj_verts": obj_verts,
            "obj_vn": obj_vn
        }