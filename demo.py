import os
import random
import argparse
import pickle
import numpy as np
import trimesh
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from manopth.manolayer import ManoLayer
from contactgen.utils.cfg_parser import Config
from contactgen.model import ContactGenModel
from contactgen.hand_sdf.hand_model import ArtiHand
from contactgen.contact.contact_optimizer import optimize_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Demo')
    parser.add_argument('--checkpoint', default='checkpoint/checkpoint.pt', type=str, help='exp dir')
    parser.add_argument('--obj_path', default="", type=str, help='object mesh path')
    parser.add_argument('--n_samples', default=5, type=int, help='number of samples per object')
    parser.add_argument('--save_root', default='exp/demo_results', type=str, help='result save root')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    # contact solver options
    parser.add_argument('--w_contact', default=1e-1, type=float, help='contact weight')
    parser.add_argument('--w_pene', default=3.0, type=float, help='penetration weight')
    parser.add_argument('--w_uv', default=1e-2, type=float, help='uv weight')
    
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    cfg_path = "contactgen/configs/default.yaml"
    model_path = "checkpoint/checkpoint.pt"

    cfg = Config(default_cfg_path=cfg_path)
    device = torch.device('cuda')
    model = ContactGenModel(cfg).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    
    config_file = "contactgen/hand_sdf/config.yaml"
    config = OmegaConf.load(config_file)
    hand_model = ArtiHand(config['model_params'], pose_size=config['pose_size'])
    checkpoint = torch.load("contactgen/hand_sdf/hand_model.pt")
    hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
    hand_model.eval()
    hand_model.to(device)

    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=26, side='right', flat_hand_mean=False)
    mano_layer.to(device)
    with open("assets/closed_mano_faces.pkl", 'rb') as f:
        hand_face = pickle.load(f)
    
    obj_mesh = trimesh.load(args.obj_path)
    offset = obj_mesh.vertices.mean(axis=0, keepdims=True)
    obj_verts = obj_mesh.vertices - offset
    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh.faces)
    sample = trimesh.sample.sample_surface(obj_mesh, 2048)
    obj_verts = sample[0].astype(np.float32)
    obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)
    
    obj_verts = torch.from_numpy(obj_verts).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
    obj_vn = torch.from_numpy(obj_vn).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
    with torch.no_grad():
        sample_results = model.sample(obj_verts, obj_vn)
    contacts_object, partition_object, uv_object = sample_results
    contacts_object = contacts_object.squeeze()
    partition_object = partition_object.argmax(dim=-1)
        
    global_pose, mano_pose, mano_shape, mano_trans = optimize_pose(hand_model, mano_layer, obj_verts, contacts_object, partition_object, uv_object,
                                                                   w_contact=args.w_contact, w_pene=args.w_pene, w_uv=args.w_uv) 
    hand_verts, hand_frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
    hand_verts = hand_verts.detach()
    hand_verts = hand_verts.cpu().numpy()
    
    for i in range(len(hand_verts)):
        obj_mesh.export(os.path.join(args.save_root, args.obj_path.split('/')[-1]))
        hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_face)
        hand_mesh.export(os.path.join(args.save_root, 'grasp_{}.obj'.format(i)))
 
    print("all done")