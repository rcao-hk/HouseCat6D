import numpy as np
import os
import glob
import pickle
from utils.evaluation_utils import compute_independent_mAP, plot_mAP


def load_results(path):
    result_pkl_list = []
    for scene in ['test_scene1', 'test_scene2', 'test_scene3', 'test_scene4', 'test_scene5']:
        result_pkl_list.extend(glob.glob(os.path.join(path, scene, '*.pkl')))
    result_pkl_list = sorted(result_pkl_list)
    
    final_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
            result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
        if isinstance(result, list):
            final_results += result
        elif isinstance(result, dict):
            final_results.append(result)
        else:
            raise ValueError("Unknown result type")
    
    return final_results

def compute_iou_3d_at_75(results, synset_names, logger=None):
    # Compute 3D IoU@75 using existing code logic
    # compute_independent_mAP(final_results, synset_names,
    #                         degree_thresholds=[5, 10],
    #                         shift_thresholds=[2, 5, 10],
    #                         iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75], logger=logger)
    iou_3d_aps, pose_aps = compute_independent_mAP(results, synset_names, 
                                            degree_thresholds=[5, 10],
                                            shift_thresholds=[2, 5, 10],
                                             iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75], 
                                             logger=logger)
    return iou_3d_aps, pose_aps


# def compare_iou_diff(iou_3d_aps_method1, iou_3d_aps_method2):
    # """
    # 计算两个方法在 3D IoU@75 的差异
    # :param iou_3d_aps_method1: 第一种方法的 3D IoU@75 结果
    # :param iou_3d_aps_method2: 第二种方法的 3D IoU@75 结果
    # :return: 返回差异大的场景
    # """
    # iou_diff = iou_3d_aps_method1 - iou_3d_aps_method2
    # # 假设差异阈值为 0.05，可根据需要调整
    # threshold = 0.05
    # significant_diff_scenes = np.where(np.abs(iou_diff) > threshold)[0]
    # return significant_diff_scenes, iou_diff


def compare_iou_diff(iou_3d_aps_method1, iou_3d_aps_method2, top_k=5):
    """
    计算两个方法在 3D IoU@75 的差异，并返回差异最大的前 K 个物体。
    :param iou_3d_aps_method1: 第一种方法的 3D IoU@75 结果 (按物体排列)
    :param iou_3d_aps_method2: 第二种方法的 3D IoU@75 结果 (按物体排列)
    :param top_k: 返回差异最大的前 K 个物体
    :return: 差异最大的前 K 个物体的索引、差异值
    """
    # 计算两种方法的 3D IoU@75 差异
    iou_diff = iou_3d_aps_method1 - iou_3d_aps_method2
    
    # 对差异值排序，返回索引和差异值
    sorted_indices = np.argsort(iou_diff)  # 从大到小排序
    top_indices = sorted_indices[:top_k]  # 取前 K 个
    
    # 获取差异最大的物体和对应的差异值
    significant_diff = iou_diff[top_indices]
    
    return top_indices, significant_diff


def main(path_method1, path_method2, logger=None):
    # 加载两种方法的结果
    results_method1 = load_results(path_method1)
    results_method2 = load_results(path_method2)

    # 获取场景名
    synset_names = ['BG', 'box', 'bottle', 'can', 'cup', 'remote', 'teapot', 'cutlery', 
                    'glass', 'shoe', 'tube']
        
    # 计算每种方法的 3D IoU@75
    iou_3d_aps_method1, pose_aps_method1 = compute_iou_3d_at_75(results_method1, synset_names, logger)
    iou_3d_aps_method2, pose_aps_method2 = compute_iou_3d_at_75(results_method2, synset_names, logger)

    # plot_mAP(
    #     iou_3d_aps_method1,
    #     pose_aps_method1,
    #     iou_thres_list=[0.10, 0.25, 0.50, 0.75],
    #     degree_thres_list=[5, 10],
    #     shift_thres_list=[2, 5, 10],
    #     out_dir='',
    #     file_name='method1_mAP.png'
    # )

    # plot_mAP(
    #     iou_3d_aps_method2,
    #     pose_aps_method2,
    #     iou_thres_list=[0.10, 0.25, 0.50, 0.75],
    #     degree_thres_list=[5, 10],
    #     shift_thres_list=[2, 5, 10],
    #     out_dir='',
    #     file_name='method2_mAP.png'
    # )

    # 比较两种方法的差异
    # significant_diff, iou_diff = compare_iou_diff(iou_3d_aps_method1[1:-1, 3], iou_3d_aps_method2[1:-1, 3])

    # # 输出差异大的场景
    # print(f"Scenes with significant IoU@75 difference:")
    # for obj_idx, obj_iou in zip(significant_diff, iou_diff):
    #     print(f"Object {synset_names[1:][obj_idx]}: IoU diff = {obj_iou:.4f}")
    
    # print all differences
    
    print(f"IoU@25 differences for all objects:")
    for obj_idx, obj_iou in enumerate(iou_3d_aps_method1[1:-1, 1] - iou_3d_aps_method2[1:-1, 1]):
        print(f"Object {synset_names[1:][obj_idx]}: IoU diff = {obj_iou * 100.0:.4f}")

    print(f"IoU@50 differences for all objects:")
    for obj_idx, obj_iou in enumerate(iou_3d_aps_method1[1:-1, 2] - iou_3d_aps_method2[1:-1, 2]):
        print(f"Object {synset_names[1:][obj_idx]}: IoU diff = {obj_iou * 100.0:.4f}")
        
    print(f"IoU@75 differences for all objects:")
    for obj_idx, obj_iou in enumerate(iou_3d_aps_method1[1:-1, 3] - iou_3d_aps_method2[1:-1, 3]):
        print(f"Object {synset_names[1:][obj_idx]}: IoU diff = {obj_iou * 100.0:.4f}")
        
    print("Mean IoU differences:")
    for thres_idx, thres in enumerate([0.10, 0.25, 0.50, 0.75]):
        mean_diff = iou_3d_aps_method1[-1, thres_idx] - iou_3d_aps_method2[-1, thres_idx]
        print(f"Mean IoU@{int(thres*100)} diff = {mean_diff * 100.0:.4f}")
    

# path_root = 
# 示例路径
# path_method1 = os.path.join('VI-Net/log/housecat/results_raw')
# path_method2 = os.path.join('VI-Net/log/housecat/results_ours_vitl_restored_conf_0.1')
# path_method1 = os.path.join('VI-Net/log/housecat_restored/results_ours_vitl_restored')
# path_method2 = os.path.join('VI-Net/log/housecat_restored_conf_0.1/results_ours_vitl_restored_conf_0.1')

path_method1 = os.path.join('VI-Net/log/housecat/results_raw')
path_method2 = os.path.join('VI-Net/log/housecat_restored_conf_0.1/results_ours_vitl_restored_conf_0.1')

# 运行并找到差异大的场景
main(path_method1, path_method2)
