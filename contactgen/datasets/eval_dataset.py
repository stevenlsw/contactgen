import os
import numpy as np
import trimesh


class TestSet():
    def __init__(self, 
                 obj_root='grab_data/obj_meshes',
                 test_list='grab_data/test_list.txt',
                 n_samples=2048):
        self.obj_root = obj_root
        self.n_samples = n_samples
        object_list = []
        with open(test_list, 'r') as f:
            for line in f:
                object_list.append(line.strip())
        self.object_list = object_list

    def __len__(self):
        return len(self.object_list)

    def __getitem__(self, item):
        obj_name = self.object_list[item].split(".")[0]
        obj_mesh_path = os.path.join(self.obj_root, self.object_list[item])
        obj_mesh = trimesh.load(obj_mesh_path)
        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = sample[0].astype(np.float32)
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)

        return {
            "obj_name": obj_name,
            "obj_verts": obj_verts,
            "obj_vn": obj_vn
        }