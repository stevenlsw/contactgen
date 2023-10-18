import argparse
import os
import open3d
import trimesh

colors = {'light_red': [0.85882353, 0.74117647, 0.65098039],
          'light_blue': [145/255, 191/255, 219/255]}


def ho_plot(hand_mesh, obj_mesh, save_path, viewpoint_path="assets/viewpoint.json"):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis_hand = open3d.geometry.TriangleMesh()
    vis_hand.vertices = open3d.utility.Vector3dVector(hand_mesh.vertices)
    vis_hand.triangles = open3d.utility.Vector3iVector(hand_mesh.faces)
    vis_hand.paint_uniform_color(colors['light_red'])
    vis_hand.compute_vertex_normals()
    vis.add_geometry(vis_hand)
    vis_obj = open3d.geometry.TriangleMesh()
    vis_obj.vertices = open3d.utility.Vector3dVector(obj_mesh.vertices)
    vis_obj.triangles = open3d.utility.Vector3iVector(obj_mesh.faces)
    vis_obj.paint_uniform_color(colors['light_blue'])        
    vis_obj.compute_vertex_normals()
    vis.add_geometry(vis_obj)
    ctr = vis.get_view_control()
    param = open3d.io.read_pinhole_camera_parameters(viewpoint_path)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()
        

if __name__ == '__main__':
    parse = argparse.ArgumentParser("Visualize grasp")
    parse.add_argument("--hand_path", type=str, default="exp/demo_results/grasp_0.obj", help="hand mesh path")
    parse.add_argument("--obj_path", type=str, default="assets/toothpaste.ply", help="object mesh path")
    parse.add_argument("--save_path", type=str, default="exp/demo_results/grasp_0.png", help="save path")
    args = parse.parse_args()
    hand_mesh = trimesh.load(args.hand_path)
    obj_mesh = trimesh.load(args.obj_path)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    ho_plot(hand_mesh, obj_mesh, args.save_path)
