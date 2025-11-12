import os
import json
import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import _pickle as cPickle
from tqdm import tqdm
import sys
import glob
import multiprocessing

cls_id_to_name = {1: "box",
                  2: "bottle",
                  3: "can",
                  4: "cup",
                  5: "remote",
                  6: "teapot",
                  7: "cutlery",
                  8: "glass",
                  9: "shoe",
                  10: "tube"}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))
# from utils.data_utils import CameraInfo

class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

dataset_root = '/mnt/DATA/robotarm/dataset/housecat6d'
obj_model_root = os.path.join(dataset_root, 'obj_models_small_size_final')


def load_mesh_model(obj_class, obj_name):
    model_path = os.path.join(obj_model_root, obj_class, f'{obj_name}.obj')
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()
    return mesh

def render_depth(mesh_list, cam_info, extrinsic, output_path, width=640, height=480, depth_scale=1000):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0, 0, 0, 0])  # black background
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.NO_SHADOWS, np.array([0, 0, -1])
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy)
    renderer.setup_camera(intrinsic, extrinsic)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    for idx, mesh in enumerate(mesh_list):
        renderer.scene.add_geometry(f"obj_{idx}", mesh, mat)

    depth_image = renderer.render_to_depth_image(z_in_view_space=True)
    depth_np = np.asarray(depth_image) * depth_scale
    depth_np = depth_np.astype(np.uint16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, depth_np)

def render_housecat6d_depth(scene_path, cam_info, instance_labels, frame_idx, width, height):
    # scene_path = os.path.join(dataset_root, scene_name)

    with open(os.path.join(scene_path, 'labels', f'{frame_idx:06d}_label.pkl'), 'rb') as f:
        gts = cPickle.load(f)
    
    mesh_list = []
    for inst_id, class_id, inst_name in instance_labels:
        # class_id = inst['class_id']
        # inst_id = inst['inst_id']
        # obj_name = taxonomy[str(class_id)]['objs'][str(inst_id)]
        # obj_class = taxonomy[str(class_id)]['class_name']
        obj_class = cls_id_to_name[int(class_id)]
        mesh = load_mesh_model(obj_class, inst_name)

        RT = np.eye(4)
        RT[:3, :3] = np.array(gts['rotations'][int(inst_id)-1]).reshape(3, 3)
        RT[:3, 3] = np.array(gts['translations'][int(inst_id)-1])
        mesh.transform(RT)
        mesh_list.append(mesh)

    output_path = os.path.join(scene_path, 'depth_gt', f'{frame_idx:06d}.png')
    render_depth(mesh_list, cam_info, extrinsic=np.eye(4), output_path=output_path, width=width, height=height)

def process_scene(scene_name):
    width, height = 1096, 852
    rgb_dir = os.path.join(dataset_root, scene_name, 'rgb')
    frame_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    frame_num = len(frame_paths)
    
    scene_path = os.path.join(dataset_root, scene_name)
    intrinsic = np.loadtxt(os.path.join(scene_path, 'intrinsics.txt')).reshape(3,3)
    cam_info = CameraInfo(width, height, intrinsic[0, 0], intrinsic[1, 1],
                         intrinsic[0, 2], intrinsic[1, 2],
                         1000.0)

    with open(os.path.join(scene_path, "meta.txt"), 'r') as f:
        scene_taxonomy = [each_line.strip().split(" ") for each_line in f.readlines()]
    
    for frame_idx in range(frame_num):
        try:
            render_housecat6d_depth(scene_path, cam_info, scene_taxonomy, frame_idx, width, height)
        except Exception as e:
            print(f"[Error] {scene_name} frame {frame_idx}: {e}")

if __name__ == '__main__':

    split = 'val'
    if split == 'train':
        scenes = [f'scene{i:02d}' for i in range(2, 35)]
    elif split == 'val':
        scenes = [f'val_scene{i:01d}' for i in range(1, 3)]
    elif split == 'test':
        scenes = [f'test_scene{i:01d}' for i in range(1, 6)]

    # 设置进程池
    num_processes = min(16, len(scenes))  # 根据机器资源调节
    ctx = multiprocessing.get_context("forkserver")
    pool = ctx.Pool(processes=num_processes)

    for scene in scenes:
        pool.apply_async(process_scene, args=(scene,))

    pool.close()
    pool.join()

    print("All rendering finished.")
