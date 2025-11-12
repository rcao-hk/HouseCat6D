import os
from PIL import Image
import open3d as o3d
import numpy as np
import re
import sys
import glob
import copy
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image


dataset_root = '/mnt/DATA//robotarm/dataset/housecat6d'
obj_model_root = '/mnt/DATA//robotarm/dataset/housecat6d/obj_models_small_size'
restored_depth_root = '/mnt/DATA/robotarm/rcao/result/depth/housecat6d/hammer_dav2_complete_obs_iter_unc_cali_convgru_l1_only_0.2_l1+grad_sigma_conf_272x208'

def get_filtered_scene(root_dir):
    pattern = re.compile(r'^(train|test|val)_scene\d+$')
    scenes = [d for d in os.listdir(root_dir) if re.match(pattern, d)]
    return scenes

# scenes = get_filtered_scene(dataset_root)
scenes = ['scene01']


for scene_name in scenes:

    data_root = os.path.join(dataset_root, scene_name)

    intrinsics = np.loadtxt(os.path.join(data_root, 'intrinsics.txt'))
    width, height = 1096, 852
    factor_depth = 1000.0

    n_images = len(glob.glob("{0}/{1}/*.png".format(data_root, "rgb")))
    print("number of images :", n_images)
    
    # extr_pose = np.loadtxt(os.path.join(data_root, 'extrinsics', 'tof_to_pol.txt'))
    with open(os.path.join(data_root, "meta.txt"), 'r') as f:
        instance_labels = [each_line.strip().split(" ") for each_line in f.readlines()]
                
    camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    for anno_idx in range(0, 2):
        rgb_path = os.path.join(data_root, 'rgb', '{:06d}.png'.format(anno_idx))
        gt_depth_path = os.path.join(data_root, 'depth_gt' ,'{:06d}.png'.format(anno_idx))
        depth_path = os.path.join(data_root, 'depth', '{:06d}.png'.format(anno_idx))
        restored_depth_path = os.path.join(restored_depth_root, scene_name,  '{:06d}_depth.png'.format(anno_idx))
        mask_path = os.path.join(data_root, 'instance', '{:06d}.png'.format(anno_idx))
        cam_pose_path = os.path.join(data_root, 'camera_pose', '{:06d}.txt'.format(anno_idx))
        # depth_path = os.path.join(dataset_root, 'restored_depth/scene_{:04d}/{}/{:04d}.png'.format(scene_idx, camera, anno_idx))
        # meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        # normal_path = os.path.join(dataset_root, 'normals/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        gt_depth = np.array(Image.open(gt_depth_path))
        # restored_depth = np.array(Image.open(restored_depth_path))
        label = np.array(Image.open(mask_path))
        cam_pose = np.loadtxt(cam_pose_path)
        
        # statistic zero depth
        print('scene:', scene_name, 'index:', anno_idx)
        print(np.sum(gt_depth==0), np.sum(gt_depth==0)/(gt_depth.shape[0]*gt_depth.shape[1]))
        # print('gt_depth range:', gt_depth.min(), gt_depth.max())
        # obj_trans_model_list = []
        # for obj_model, obj_pose in zip(obj_model_list, obj_pose_list):
        #     trans_pose = np.dot(np.linalg.inv(cam_pose), obj_pose)
        #     obj_trans_model = copy.deepcopy(obj_model)
        #     obj_trans_model.transform(trans_pose)
        #     obj_trans_model_list.append(obj_trans_model)
            
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
        gt_cloud = create_point_cloud_from_depth_image(gt_depth, camera_info, organized=True)
        # restored_cloud = create_point_cloud_from_depth_image(restored_depth, camera_info, organized=True)
        
        scene_mask = gt_depth > 0
        cloud = cloud[scene_mask]
        color = color[scene_mask]
        gt_cloud = gt_cloud[scene_mask]
        
        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
        scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        scene = scene.voxel_down_sample(voxel_size=0.002)
        
        gt_scene = o3d.geometry.PointCloud()
        gt_scene.points = o3d.utility.Vector3dVector(gt_cloud.reshape(-1, 3))
        # gt_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        gt_scene.paint_uniform_color([1.0, 0.0, 0.0])
        gt_scene = gt_scene.voxel_down_sample(voxel_size=0.002)
        
        # restored_scene = o3d.geometry.PointCloud()
        # restored_scene.points = o3d.utility.Vector3dVector(restored_cloud.reshape(-1, 3))
        # # restored_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        # restored_scene.paint_uniform_color([0.5, 0.5, 0.5])
        
        # vis_scene = gt_scene + restored_scene
        vis_scene = gt_scene + scene
        os.makedirs('vis', exist_ok=True)
        o3d.io.write_point_cloud(os.path.join('vis', '{}_{:06d}.ply'.format(scene_name, anno_idx)), vis_scene)
        # o3d.visualization.draw_geometries([origin_frame, gt_scene])

        # inst_ids = np.unique(label)
        # print('scene:', scene_name)
        # print('instance:', inst_ids)
        # for inst_id in inst_ids:
        #     inst_mask = label == inst_id
        #     choose = list(inst_mask.flatten().nonzero()[0])
        #     inst_pcd = scene.select_by_index(choose)
        #     print('instance index:', inst_id)
        #     o3d.visualization.draw_geometries([origin_frame, inst_pcd])

