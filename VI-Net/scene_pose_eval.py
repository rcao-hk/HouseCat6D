#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import _pickle as cPickle
import numpy as np
from tqdm import tqdm
import cv2

# 你工程里的工具/评估函数（路径按你项目结构）
from utils.draw_utils import get_3d_bbox, transform_coordinates_3d, calculate_2d_projections, compute_matches
from utils.evaluation_utils import compute_independent_mAP

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------

SCENES = ['test_scene1', 'test_scene2', 'test_scene3', 'test_scene4', 'test_scene5']

# 用 HouseCat6D 的全量类别，保持与 pkl 内 gt_class_ids 对齐
HOUSECAT_SYNS = ['BG', 'box', 'bottle', 'can', 'cup', 'remote',
                 'teapot', 'cutlery', 'glass', 'shoe', 'tube']

# 仅评估 cutlery / glass
ALLOWED_CLASS_NAMES = {'cutlery', 'glass'}
ALLOWED_CLASS_IDS = [HOUSECAT_SYNS.index(n) for n in ALLOWED_CLASS_NAMES]  # -> [7, 8]

# 重要路径（按需修改）
dataset_root = '/mnt/DATA/robotarm/rcao/dataset/HouseCat6D'
method_roots = {
    'raw':            'VI-Net/log/housecat/results_raw',
    'restored':       'VI-Net/log/housecat_restored/results_ours_vitl_restored',
    'restored_conf':  'VI-Net/log/housecat_restored_conf_0.1/results_ours_vitl_restored_conf_0.1',
}
ref_method  = 'restored_conf'
base_method = 'restored'
top_k = 50
vis_root = 'vis_class_cutlery_glass'

os.makedirs(vis_root, exist_ok=True)

# ------------------------------------------------------------
# 数据加载 & 过滤
# ------------------------------------------------------------

def load_final_results_with_paths(root_path, scenes=SCENES):
    """
    返回:
      final_results: List[dict]  (每个 sample 的结果字典)
      pkl_paths:     List[str]   (与 final_results 一一对应的 .pkl 路径)
    """
    result_pkl_list = []
    for scene in scenes:
        result_pkl_list.extend(glob.glob(os.path.join(root_path, scene, '*.pkl')))
    result_pkl_list = sorted(result_pkl_list)

    final_results = []
    pkl_paths     = []

    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)

        if isinstance(result, dict):
            result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            final_results.append(result)
            pkl_paths.append(pkl_path)
        elif isinstance(result, list):
            for r in result:
                r['gt_handle_visibility'] = np.ones_like(r['gt_class_ids'])
                final_results.append(r)
                pkl_paths.append(pkl_path)
        else:
            raise ValueError(f'Unknown result type: {type(result)}')

    return final_results, pkl_paths


def filter_results_by_classes(final_results, pkl_paths, allowed_class_ids):
    keep_results, keep_paths = [], []
    for r, p in zip(final_results, pkl_paths):
        gt = np.asarray(r['gt_class_ids']).astype(np.int32)
        if np.intersect1d(gt, allowed_class_ids).size > 0:
            keep_results.append(r)
            keep_paths.append(p)
    return keep_results, keep_paths


# ------------------------------------------------------------
# 逐样本四个 pose 指标（5°2cm / 5°5cm / 10°2cm / 10°5cm）
# ------------------------------------------------------------

def compute_sample_pose_scores(
    final_results,
    synset_names,
    degree_thresholds=[5, 10],
    shift_thresholds=[2, 5],
    iou_3d_thresholds=[0.1, 0.25, 0.5, 0.75],
    iou_pose_thres=0.1,
):
    """
    对 final_results 中的每个 sample 单独调用一次 compute_independent_mAP([result]),
    返回:
      scores: (N_samples, num_classes+1, 4)
      metrics_names: ['5d_2cm', '5d_5cm', '10d_2cm', '10d_5cm']
    """
    num_samples = len(final_results)
    num_classes = len(synset_names)

    if num_samples == 0:
        return np.zeros((0, num_classes + 1, 4), dtype=np.float32), ['5d_2cm', '5d_5cm', '10d_2cm', '10d_5cm']

    degree_thres_list = list(degree_thresholds) + [360]
    shift_thres_list  = list(shift_thresholds) + [100]
    idx_d5  = degree_thres_list.index(5)
    idx_d10 = degree_thres_list.index(10)
    idx_s2  = shift_thres_list.index(2)
    idx_s5  = shift_thres_list.index(5)

    metrics_names = ['5d_2cm', '5d_5cm', '10d_2cm', '10d_5cm']
    scores = np.full((num_samples, num_classes + 1, 4), np.nan, dtype=np.float32)

    for s_idx, result in enumerate(tqdm(final_results, desc="compute per-sample pose scores")):
        _, pose_aps = compute_independent_mAP(
            [result],
            synset_names,
            degree_thresholds=degree_thresholds,
            shift_thresholds=shift_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
            iou_pose_thres=iou_pose_thres,
            use_matches_for_pose=True,
            logger=None
        )
        # pose_aps: (num_classes+1, D, S)
        m0 = pose_aps[:, idx_d5,  idx_s2]   # 5° 2cm
        m1 = pose_aps[:, idx_d5,  idx_s5]   # 5° 5cm
        m2 = pose_aps[:, idx_d10, idx_s2]   # 10° 2cm
        m3 = pose_aps[:, idx_d10, idx_s5]   # 10° 5cm

        scores[s_idx, :, 0] = m0
        scores[s_idx, :, 1] = m1
        scores[s_idx, :, 2] = m2
        scores[s_idx, :, 3] = m3

    return scores, metrics_names


def collect_scores_for_methods(method_roots, synset_names, allowed_class_ids=None):
    """
    method_roots: dict{name: root_path}
    返回:
      results_dict[name] = final_results (list of dict, 已过滤 & 对齐)
      paths_dict[name]   = pkl_paths   (list of str, 已过滤 & 对齐)
      scores_dict[name]  = scores (N, C+1, 4)
      metrics_names      = ['5d_2cm', '5d_5cm', '10d_2cm', '10d_5cm']
    """
    results_dict = {}
    paths_dict   = {}
    scores_dict  = {}
    metrics_names = None

    # 1) 先加载 & 类别过滤
    loaded = {}
    for name, root in method_roots.items():
        final_results, pkl_paths = load_final_results_with_paths(root)
        if allowed_class_ids is not None:
            final_results, pkl_paths = filter_results_by_classes(final_results, pkl_paths, allowed_class_ids)
        if len(final_results) == 0:
            raise RuntimeError(f"[{name}] 在 {root} 下没有符合筛选条件的样本（可能没有 pkl，或没有 cutlery/glass GT）")
        loaded[name] = (final_results, pkl_paths)

    # 2) 对齐样本（取公共相对路径键）
    key_map = {}
    for name, (fr, paths) in loaded.items():
        keys = [os.path.relpath(p, method_roots[name]) for p in paths]
        key_map[name] = keys

    common_keys = set.intersection(*[set(v) for v in key_map.values()])
    if len(common_keys) == 0:
        raise RuntimeError("不同方法在过滤后没有共同样本，请检查路径或筛选条件。")
    common_keys = sorted(common_keys)

    for name, (fr, paths) in loaded.items():
        keys = key_map[name]
        key2idx = {k:i for i,k in enumerate(keys)}
        sel_idx = [key2idx[k] for k in common_keys]

        final_results = [fr[i]    for i in sel_idx]
        pkl_paths     = [paths[i] for i in sel_idx]

        results_dict[name] = final_results
        paths_dict[name]   = pkl_paths

        scores, metrics_names = compute_sample_pose_scores(
            final_results, synset_names,
            degree_thresholds=[5, 10],
            shift_thresholds=[2, 5],
            iou_3d_thresholds=[0.1, 0.25, 0.5, 0.75],
            iou_pose_thres=0.1
        )
        scores_dict[name] = scores

    return results_dict, paths_dict, scores_dict, metrics_names


# ------------------------------------------------------------
# Top-K 样本筛选（仅聚合 cutlery/glass 两类）
# ------------------------------------------------------------

def find_topk_samples_advantage_multi(
    scores_dict,
    paths_dict,
    ref_method,
    base_method,
    allowed_class_ids,
    top_k=10
):
    # 任一方法的形状
    scores_ref  = scores_dict[ref_method]
    scores_base = scores_dict[base_method]
    N, Cplus1, M = scores_ref.shape

    def per_sample_score(scores):
        sub = scores[:, allowed_class_ids, :]  # (N, K, 4)
        return np.nanmean(sub, axis=(1, 2))    # (N,)

    score_ref  = per_sample_score(scores_ref)
    score_base = per_sample_score(scores_base)
    deltas = score_ref - score_base

    valid_idx = np.where(~np.isnan(deltas))[0]
    sorted_valid = valid_idx[np.argsort(deltas[valid_idx])[::-1]]
    topk_idx = sorted_valid[:top_k]

    pkl_paths_example = paths_dict[ref_method]
    topk_info = []
    for rank, s in enumerate(topk_idx):
        p = pkl_paths_example[s]
        scene_name  = os.path.basename(os.path.dirname(p))
        sample_name = os.path.splitext(os.path.basename(p))[0]
        method_scores = {m: float(np.nanmean(sc[s, allowed_class_ids, :])) for m, sc in scores_dict.items()}
        topk_info.append({
            'rank':       int(rank + 1),
            'index':      int(s),
            'scene':      scene_name,
            'sample':     sample_name,
            'delta_ref_base': float(deltas[s]),
            'ref_score':  float(score_ref[s]),
            'base_score': float(score_base[s]),
            'method_scores': method_scores,
            'pkl_path':   p,
        })
    return topk_info


def print_topk_summary(topk_info, ref_method, base_method):
    print(f"\nTop-{len(topk_info)} samples where {ref_method} beats {base_method} the most (mean of 4 pose metrics on cutlery/glass):")
    for info in topk_info:
        print(f"[{info['rank']:02d}] idx={info['index']:05d}, "
              f"{info['scene']}/{info['sample']}, "
              f"Δ={info['delta_ref_base']*100:.2f} "
              f"({ref_method}={info['ref_score']*100:.2f}, {base_method}={info['base_score']*100:.2f})")
        for m, v in info['method_scores'].items():
            print(f"      {m}: {v*100:.2f}")


# ------------------------------------------------------------
# 可视化（单图叠加 GT / Pred，不画轴，仅画 3D bbox）
# ------------------------------------------------------------

def to_points_8x2(projected_bbox):
    """
    将 projected_bbox 规范为 (8,2) 的 int32。
    支持输入形状:
      - (2,8)/(3,8)  -> 取前2行转置
      - (8,2)/(8,3)  -> 取前2列
    """
    pb = np.asarray(projected_bbox)
    if pb.ndim != 2:
        return None
    h, w = pb.shape
    if h in (2, 3):
        pts2d = pb[:2, :].T
    elif w in (2, 3):
        pts2d = pb[:, :2]
    else:
        return None
    pts2d = pts2d.astype(np.int32)
    if pts2d.shape[0] >= 8:
        return pts2d[:8, :]
    return None


def draw_bbox(img, imgpts, color):
    """
    使用你给的风格画 3D bbox（地面/立柱/顶部，三层颜色深浅）
    imgpts: (8,2) int32
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # ground（底面）浅色
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)

    # pillars（立柱）中等
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    # top（顶面）原色
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)

    return img


def _filter_result_to_allowed_classes(result, allowed_class_ids):
    out = dict(result)  # 浅拷贝
    # GT
    if 'gt_class_ids' in result and result['gt_class_ids'] is not None:
        gmask = np.isin(result['gt_class_ids'], allowed_class_ids)
        for k in ['gt_class_ids', 'gt_bboxes', 'gt_RTs', 'gt_scales', 'gt_handle_visibility']:
            if k in result and result[k] is not None:
                out[k] = np.asarray(result[k])[gmask]
    # PRED
    if 'pred_class_ids' in result and result['pred_class_ids'] is not None:
        pmask = np.isin(result['pred_class_ids'], allowed_class_ids)
        for k in ['pred_class_ids', 'pred_bboxes', 'pred_RTs', 'pred_scales', 'pred_scores', 'pred_mask']:
            if k in result and result[k] is not None:
                out[k] = np.asarray(result[k])[pmask]
    return out


def draw_detections(image, save_dir, image_name, intrinsics,
                    gt_bbox, gt_class_ids, gt_mask, gt_RTs, gt_scales,
                    pred_bbox, pred_class_ids, pred_mask, pred_RTs, pred_scores, pred_scales,
                    draw_gt=False, draw_pred=True):
    """
    - GT 和 Pred 叠加在一张图上
    - 不画 xyz 轴，只画 3D bbox
    - 输出文件：{image_name}_bbox.png
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{image_name}_bbox.png')
    draw_image_bbox = image.copy()

    # 画 GT（绿色）
    if draw_gt and gt_RTs is not None and gt_scales is not None:
        for ind, RT in enumerate(gt_RTs):
            bbox_3d = get_3d_bbox(gt_scales[ind], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            pts8 = to_points_8x2(projected_bbox)
            if pts8 is not None:
                draw_image_bbox = draw_bbox(draw_image_bbox, pts8, (0, 255, 0))

    # 画 Pred（红色）
    if draw_pred and pred_class_ids is not None and len(pred_class_ids) > 0:
        num_pred_instances = len(pred_class_ids)

        # 可选：按你原逻辑做匹配，重排 pred_RTs / pred_scales
        if (gt_class_ids is not None and gt_bbox is not None and
            pred_bbox is not None and pred_scores is not None):
            try:
                gt_match, pred_match, _, pred_indices = compute_matches(
                    gt_bbox, gt_class_ids, gt_mask,
                    pred_bbox, pred_class_ids, pred_scores, pred_mask,
                    0.5
                )
                if len(pred_indices):
                    pred_RTs    = pred_RTs[pred_indices]
                    pred_scales = pred_scales[pred_indices]
            except Exception:
                pass  # 有些实现里 mask 为 None 也能跑；不稳定就跳过重排

        if pred_RTs is not None and pred_scales is not None:
            for ind in range(num_pred_instances):
                RT = pred_RTs[ind]
                bbox_3d = get_3d_bbox(pred_scales[ind, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                pts8 = to_points_8x2(projected_bbox)
                if pts8 is not None:
                    draw_image_bbox = draw_bbox(draw_image_bbox, pts8, (255, 0, 0))

    # image 是 RGB，这里保存前转 BGR
    cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])


def visualize_topk_bboxes(
    topk_info,
    results_dict,
    intrinsics_dict,   # scene_name -> intrinsics.txt
    save_path_root,
    dataset_root,
    allowed_class_ids
):
    """
    对 top-k samples，遍历所有方法，各画一张 bbox 可视化图（同一张 RGB；GT+Pred 叠加）
    """
    os.makedirs(save_path_root, exist_ok=True)

    for info in topk_info:
        idx   = info['index']
        scene = info['scene']
        sample= info['sample']
        diff  = info['delta_ref_base']

        # RGB 路径（统一用 dataset_root/scene/rgb/sample.png）
        image_path = os.path.join(dataset_root, scene, 'rgb', f'{sample}.png')
        image = cv2.imread(image_path)[:, :, :3]  # BGR
        image = image[:, :, ::-1]                # 转 RGB

        # 内参
        if scene not in intrinsics_dict:
            raise ValueError(f"scene {scene} not in intrinsics_dict, please check naming.")
        intrinsics = np.loadtxt(intrinsics_dict[scene]).reshape(3, 3)

        # 每个方法单独输出一张
        for m_name, final_results in results_dict.items():
            result = final_results[idx]
            # 只保留允许类别（cutlery/glass）
            result_f = _filter_result_to_allowed_classes(result, allowed_class_ids)

            out_dir = os.path.join(save_path_root, f'draw_{m_name}')
            os.makedirs(out_dir, exist_ok=True)

            save_name = f"{scene}_{sample}_{m_name}_delta{diff:.3f}"

            draw_detections(
                image.copy(),
                out_dir,
                save_name,
                intrinsics,
                result_f.get('gt_bboxes', None),
                result_f.get('gt_class_ids', None),
                result_f.get('gt_mask', None),
                result_f.get('gt_RTs', None),
                result_f.get('gt_scales', None),
                result_f.get('pred_bboxes', None),
                result_f.get('pred_class_ids', None),
                result_f.get('pred_mask', None),
                result_f.get('pred_RTs', None),
                result_f.get('pred_scores', None),
                result_f.get('pred_scales', None),
                draw_gt=True,
                draw_pred=True
            )


# ------------------------------------------------------------
# 构造 intrinsics 映射并执行
# ------------------------------------------------------------

if __name__ == "__main__":
    # scene_name -> intrinsics.txt
    test_scenes_rgb = sorted(glob.glob(os.path.join(dataset_root, 'test_scene*', 'rgb')))
    intrinsics_dict = {}
    for rgb_dir in test_scenes_rgb:
        scene_dir = os.path.dirname(rgb_dir)          # /.../test_sceneX
        scene_name = os.path.basename(scene_dir)      # test_sceneX
        intr_path = os.path.join(scene_dir, 'intrinsics.txt')
        intrinsics_dict[scene_name] = intr_path

    # 计算多方法对齐后的 per-sample scores（仅 cutlery/glass）
    results_dict, paths_dict, scores_dict, metrics_names = collect_scores_for_methods(
        method_roots, HOUSECAT_SYNS, allowed_class_ids=ALLOWED_CLASS_IDS
    )

    # 选出 ref 相对 base 优势最大的 top-k 样本
    topk_info = find_topk_samples_advantage_multi(
        scores_dict,
        paths_dict,
        ref_method=ref_method,
        base_method=base_method,
        allowed_class_ids=ALLOWED_CLASS_IDS,
        top_k=top_k
    )

    print_topk_summary(topk_info, ref_method=ref_method, base_method=base_method)

    # 可视化叠加 GT/Pred bbox
    visualize_topk_bboxes(
        topk_info,
        results_dict,
        intrinsics_dict,
        vis_root,
        dataset_root,
        ALLOWED_CLASS_IDS
    )

    print(f"\nDone. Images saved to: {os.path.abspath(vis_root)}")
