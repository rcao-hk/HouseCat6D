#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import _pickle as cPickle
import numpy as np
from tqdm import tqdm
import cv2

# 你工程里的工具/评估函数（路径按你项目结构）
from utils.draw_utils import (
    get_3d_bbox, transform_coordinates_3d, calculate_2d_projections, compute_matches
)
from utils.evaluation_utils import (
    compute_independent_mAP, compute_3d_matches
)

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------

SCENES = ['test_scene1', 'test_scene2', 'test_scene3', 'test_scene4', 'test_scene5']

HOUSECAT_SYNS = ['BG', 'box', 'bottle', 'can', 'cup', 'remote',
                 'teapot', 'cutlery', 'glass', 'shoe', 'tube']

ALLOWED_CLASS_NAMES = {'glass', 'cutlery'}
ALLOWED_CLASS_IDS = [HOUSECAT_SYNS.index(n) for n in ALLOWED_CLASS_NAMES]  # -> [7, 8]

dataset_root = '/mnt/DATA/robotarm/rcao/dataset/HouseCat6D'
method_roots = {
    'raw':            'VI-Net/log/housecat/results_raw',
    'restored':       'VI-Net/log/housecat_restored/results_ours_vitl_restored',
    'restored_conf':  'VI-Net/log/housecat_restored_conf_0.1/results_ours_vitl_restored_conf_0.1',
}
method3 = 'restored_conf'
method2 = 'restored'
method1 = 'raw'

top_k = 50
vis_root = 'vis_glass_cutlery'
os.makedirs(vis_root, exist_ok=True)

# IoU “匹配比例”阈值（你要的：>0.1, >0.25, >0.5, >0.75 的比值）
IOU_RATIO_THRESHOLDS = [0.1, 0.25, 0.5, 0.75]


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------

def load_final_results_with_paths(root_path, scenes=SCENES):
    result_pkl_list = []
    for scene in scenes:
        result_pkl_list.extend(glob.glob(os.path.join(root_path, scene, '*.pkl')))
    result_pkl_list = sorted(result_pkl_list)

    if len(result_pkl_list) == 0:
        raise RuntimeError(f"No pkls found under: {root_path}")

    final_results = []
    pkl_paths = []

    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)

        if isinstance(result, dict):
            if 'gt_handle_visibility' not in result or result['gt_handle_visibility'] is None:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            final_results.append(result)
            pkl_paths.append(pkl_path)
        elif isinstance(result, list):
            for r in result:
                if 'gt_handle_visibility' not in r or r['gt_handle_visibility'] is None:
                    r['gt_handle_visibility'] = np.ones_like(r['gt_class_ids'])
                final_results.append(r)
                pkl_paths.append(pkl_path)
        else:
            raise ValueError(f'Unknown result type: {type(result)}')

    return final_results, pkl_paths


def filter_indices_by_allowed_gt(results_ref, allowed_class_ids):
    """基于 GT 是否包含 allowed 类别来筛 sample（保证所有方法一致）"""
    keep = []
    for r in results_ref:
        gt = np.asarray(r.get('gt_class_ids', [])).astype(np.int32)
        keep.append(np.intersect1d(gt, allowed_class_ids).size > 0)
    keep_idx = np.where(np.asarray(keep, dtype=bool))[0]
    return keep_idx


# ------------------------------------------------------------
# IoU “匹配比例”计算：#(GT matched with IoU>=thr) / #(GT)
# ------------------------------------------------------------
# def compute_iou_match_ratio_per_class(result, synset_names, class_ids, thresholds):
#     """
#     返回:
#       ratios: dict[cls_id] -> np.ndarray(shape=(len(thresholds),), float32)
#       若该 cls 在该 sample 没有 GT，则对应全 NaN
#     定义：
#       对每个 cls，ratio(thr) = mean(gt_matches[thr_idx, :] > -1)
#       即：该 cls 的 GT 有多少比例能被同类 pred 以 IoU>=thr 匹配到
#     """
#     gt_class_ids = np.asarray(result.get('gt_class_ids', [])).astype(np.int32)
#     gt_RTs = np.asarray(result.get('gt_RTs', []))
#     gt_scales = np.asarray(result.get('gt_scales', []))
#     gt_handle_visibility = np.asarray(result.get('gt_handle_visibility', np.ones_like(gt_class_ids)))

#     pred_class_ids = np.asarray(result.get('pred_class_ids', [])).astype(np.int32)
#     pred_bboxes = np.asarray(result.get('pred_bboxes', np.zeros((len(pred_class_ids), 4), dtype=np.float32)))
#     pred_scores = result.get('pred_scores', None)
#     if pred_scores is None:
#         pred_scores = np.ones((len(pred_class_ids),), dtype=np.float32)
#     else:
#         pred_scores = np.asarray(pred_scores).astype(np.float32)

#     pred_RTs = np.asarray(result.get('pred_RTs', []))
#     pred_scales = np.asarray(result.get('pred_scales', []))

#     ratios = {}
#     T = len(thresholds)

#     for cls_id in class_ids:
#         gmask = (gt_class_ids == cls_id)
#         pmask = (pred_class_ids == cls_id)

#         if np.sum(gmask) == 0:
#             ratios[cls_id] = np.full((T,), np.nan, dtype=np.float32)
#             continue

#         cls_gt_class_ids = gt_class_ids[gmask]
#         cls_gt_RTs = gt_RTs[gmask] if len(gt_RTs) else np.zeros((0, 4, 4))
#         cls_gt_scales = gt_scales[gmask] if len(gt_scales) else np.zeros((0, 3))
#         cls_gt_vis = gt_handle_visibility[gmask] if len(gt_handle_visibility) else np.ones((len(cls_gt_class_ids),))

#         if np.sum(pmask) == 0:
#             ratios[cls_id] = np.zeros((T,), dtype=np.float32)  # 有 GT 没 pred，则匹配比例为 0
#             continue

#         cls_pred_class_ids = pred_class_ids[pmask]
#         cls_pred_bboxes = pred_bboxes[pmask] if len(pred_bboxes) else np.zeros((len(cls_pred_class_ids), 4))
#         cls_pred_scores = pred_scores[pmask]
#         cls_pred_RTs = pred_RTs[pmask] if len(pred_RTs) else np.zeros((len(cls_pred_class_ids), 4, 4))
#         cls_pred_scales = pred_scales[pmask] if len(pred_scales) else np.zeros((len(cls_pred_class_ids), 3))

#         gt_matches, pred_matches, overlaps, _ = compute_3d_matches(
#             cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_vis, synset_names,
#             cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
#             thresholds
#         )

#         # gt_matches: (T, num_gt)，> -1 表示该 GT 在该阈值下被匹配到
#         num_gt = gt_matches.shape[1]
#         if num_gt == 0:
#             ratios[cls_id] = np.full((T,), np.nan, dtype=np.float32)
#         else:
#             r = (gt_matches > -1).mean(axis=1).astype(np.float32)  # (T,)
#             ratios[cls_id] = r

#     return ratios


def compute_mean_iou_per_class(
    result, synset_names, class_ids,
    match_iou_thres=0.0,     # 用于“是否允许匹配”的阈值；0 就表示只要 IoU>0 就可匹配
    penalize_miss=True       # True: 未匹配GT记0（惩罚漏检）；False: 只对匹配到的GT求均值
):
    """
    返回:
      mean_ious: dict[cls_id] -> float (该sample该类的平均IoU)
      若该 cls 在该 sample 没有 GT，则为 NaN
    """
    gt_class_ids = np.asarray(result.get('gt_class_ids', [])).astype(np.int32)
    gt_RTs = np.asarray(result.get('gt_RTs', []))
    gt_scales = np.asarray(result.get('gt_scales', []))
    gt_vis = np.asarray(result.get('gt_handle_visibility', np.ones_like(gt_class_ids)))

    pred_class_ids = np.asarray(result.get('pred_class_ids', [])).astype(np.int32)
    pred_bboxes = np.asarray(result.get('pred_bboxes', np.zeros((len(pred_class_ids), 4), dtype=np.float32)))
    pred_scores = result.get('pred_scores', None)
    pred_scores = np.ones((len(pred_class_ids),), dtype=np.float32) if pred_scores is None else np.asarray(pred_scores).astype(np.float32)
    pred_RTs = np.asarray(result.get('pred_RTs', []))
    pred_scales = np.asarray(result.get('pred_scales', []))

    mean_ious = {}

    for cls_id in class_ids:
        gmask = (gt_class_ids == cls_id)
        pmask = (pred_class_ids == cls_id)

        num_gt = int(np.sum(gmask))
        if num_gt == 0:
            mean_ious[cls_id] = np.nan
            continue

        # 取出该类的 GT / Pred
        cls_gt_ids = gt_class_ids[gmask]
        cls_gt_RTs = gt_RTs[gmask]
        cls_gt_scales = gt_scales[gmask]
        cls_gt_vis = gt_vis[gmask]

        if np.sum(pmask) == 0:
            mean_ious[cls_id] = 0.0 if penalize_miss else np.nan
            continue

        cls_pred_ids = pred_class_ids[pmask]
        cls_pred_bboxes = pred_bboxes[pmask]
        cls_pred_scores = pred_scores[pmask]
        cls_pred_RTs = pred_RTs[pmask]
        cls_pred_scales = pred_scales[pmask]

        # 用 compute_3d_matches 拿到 overlaps + 1-1 matching 结果
        gt_matches, pred_matches, overlaps, _ = compute_3d_matches(
            cls_gt_ids, cls_gt_RTs, cls_gt_scales, cls_gt_vis, synset_names,
            cls_pred_bboxes, cls_pred_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
            iou_3d_thresholds=[match_iou_thres]
        )

        # gt_matches: (1, num_gt). 其中 gt_matches[0,j] = matched_pred_index or -1
        m = gt_matches[0]
        ious = np.zeros((num_gt,), dtype=np.float32)
        matched_mask = (m > -1)
        if np.any(matched_mask):
            pred_idx = m[matched_mask].astype(np.int32)
            gt_idx = np.where(matched_mask)[0].astype(np.int32)
            ious[matched_mask] = overlaps[pred_idx, gt_idx].astype(np.float32)

        if penalize_miss:
            mean_ious[cls_id] = float(np.mean(ious))               # 未匹配的GT贡献0
        else:
            mean_ious[cls_id] = float(np.mean(ious[matched_mask])) if np.any(matched_mask) else 0.0

    return mean_ious

# ------------------------------------------------------------
# 逐样本 pose 指标 + iou_ratio 指标
# ------------------------------------------------------------

def compute_sample_pose_scores(
    final_results,
    synset_names,
    allowed_class_ids=None,
    degree_thresholds=[5, 10],
    shift_thresholds=[2, 5],
    iou_3d_thresholds=[0.1, 0.25, 0.5, 0.75],
    iou_pose_thres=0.1,
    iou_ratio_thresholds=IOU_RATIO_THRESHOLDS,
):
    """
    scores: (N, C+1, K)
      0..3:  pose AP: [5d2cm, 5d5cm, 10d2cm, 10d5cm]
      4..6:  3D IoU AP(mean): [@0.25, @0.5, @0.75]  (沿用你原逻辑 iou_3d_aps[-1,*])
      7..10: iou_match_ratio(按GT计): [>0.1, >0.25, >0.5, >0.75] (只对 allowed_class_ids 填；其余 NaN)
    """
    num_samples = len(final_results)
    num_classes = len(synset_names)

    if num_samples == 0:
        return np.zeros((0, num_classes + 1, 11), dtype=np.float32), ['5d_2cm', '5d_5cm', '10d_2cm', '10d_5cm']

    degree_thres_list = list(degree_thresholds) + [360]
    shift_thres_list  = list(shift_thresholds) + [100]
    idx_d5  = degree_thres_list.index(5)
    idx_d10 = degree_thres_list.index(10)
    idx_s2  = shift_thres_list.index(2)
    idx_s5  = shift_thres_list.index(5)

    idx_iou25 = iou_3d_thresholds.index(0.25)
    idx_iou50 = iou_3d_thresholds.index(0.5)
    idx_iou75 = iou_3d_thresholds.index(0.75)

    scores = np.full((num_samples, num_classes + 1, 11), np.nan, dtype=np.float32)

    for s_idx, result in enumerate(tqdm(final_results, desc="compute per-sample scores")):
        # ---- 1) pose AP / iou AP（沿用 compute_independent_mAP）----
        out = compute_independent_mAP(
            [result],
            synset_names,
            degree_thresholds=degree_thresholds,
            shift_thresholds=shift_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
            iou_pose_thres=iou_pose_thres,
            use_matches_for_pose=True,
            logger=None
        )
        if isinstance(out, tuple) and len(out) == 3:
            iou_3d_aps, _, pose_aps = out
        else:
            iou_3d_aps, pose_aps = out

        m0 = pose_aps[:, idx_d5,  idx_s2]   # 5° 2cm
        m1 = pose_aps[:, idx_d5,  idx_s5]   # 5° 5cm
        m2 = pose_aps[:, idx_d10, idx_s2]   # 10° 2cm
        m3 = pose_aps[:, idx_d10, idx_s5]   # 10° 5cm

        scores[s_idx, :, 0] = m0
        scores[s_idx, :, 1] = m1
        scores[s_idx, :, 2] = m2
        scores[s_idx, :, 3] = m3

        scores[s_idx, :, 4] = iou_3d_aps[:, idx_iou25]
        scores[s_idx, :, 5] = iou_3d_aps[:, idx_iou50]
        scores[s_idx, :, 6] = iou_3d_aps[:, idx_iou75]

        # ---- 2) iou “匹配比例” （只按 allowed_class_ids 计算并填入 7..10）----
        # if allowed_class_ids is not None and len(allowed_class_ids) > 0:
        #     ratios = compute_iou_match_ratio_per_class(
        #         result, synset_names, allowed_class_ids, iou_ratio_thresholds
        #     )
        #     for cls_id in allowed_class_ids:
        #         v = ratios.get(cls_id, None)
        #         if v is None:
        #             continue
        #         scores[s_idx, cls_id, 7:7+len(iou_ratio_thresholds)] = v

        #     # 可选：填 mean 行（只对 allowed 类做 mean），方便 debug
        #     scores[s_idx, -1, 7]  = np.nanmean(scores[s_idx, allowed_class_ids, 7])
        #     scores[s_idx, -1, 8]  = np.nanmean(scores[s_idx, allowed_class_ids, 8])
        #     scores[s_idx, -1, 9]  = np.nanmean(scores[s_idx, allowed_class_ids, 9])
        #     scores[s_idx, -1, 10] = np.nanmean(scores[s_idx, allowed_class_ids, 10])

        # ---- 用 IoU 值（mean IoU）填入 scores[..., 10] ----
        if allowed_class_ids is not None and len(allowed_class_ids) > 0:
            mean_ious = compute_mean_iou_per_class(
                result, synset_names, allowed_class_ids,
                match_iou_thres=0.0,
                penalize_miss=True
            )
            for cls_id in allowed_class_ids:
                scores[s_idx, cls_id, 10] = mean_ious.get(cls_id, np.nan)

            # mean 行（只对 allowed 类平均），方便后续直接用
            scores[s_idx, -1, 10] = np.nanmean(scores[s_idx, allowed_class_ids, 10])

    return scores, ['5d_2cm', '5d_5cm', '10d_2cm', '10d_5cm']


def collect_scores_for_methods(method_roots, synset_names, allowed_class_ids=None):
    """
    关键改动（为避免你之前 raw 过滤为空）：
      - 先加载所有方法，不做 per-method 过滤
      - 先对齐 common_keys
      - 再基于对齐后“参考方法”的 GT 做 allowed_class_ids 过滤，并同步到所有方法
    """
    loaded = {}
    key_map = {}

    # 1) load (unfiltered)
    for name, root in method_roots.items():
        final_results, pkl_paths = load_final_results_with_paths(root)
        keys = [os.path.relpath(p, root) for p in pkl_paths]
        loaded[name] = (final_results, pkl_paths)
        key_map[name] = keys

    # 2) align common keys
    common_keys = set.intersection(*[set(v) for v in key_map.values()])
    if len(common_keys) == 0:
        raise RuntimeError("不同方法没有共同样本，请检查各方法结果目录结构是否一致。")
    common_keys = sorted(common_keys)

    results_dict, paths_dict, scores_dict = {}, {}, {}

    for name, root in method_roots.items():
        fr, paths = loaded[name]
        keys = key_map[name]
        key2idx = {k: i for i, k in enumerate(keys)}
        sel_idx = [key2idx[k] for k in common_keys]

        results_dict[name] = [fr[i] for i in sel_idx]
        paths_dict[name] = [paths[i] for i in sel_idx]

    # 3) filter by GT classes (based on ref method)
    if allowed_class_ids is not None:
        ref_name = next(iter(method_roots.keys()))
        keep_idx = filter_indices_by_allowed_gt(results_dict[ref_name], allowed_class_ids)
        if len(keep_idx) == 0:
            raise RuntimeError(f"对齐后仍没有包含 {allowed_class_ids} 的样本，请检查数据/类别ID。")

        for name in method_roots.keys():
            results_dict[name] = [results_dict[name][i] for i in keep_idx]
            paths_dict[name] = [paths_dict[name][i] for i in keep_idx]

    # 4) compute scores
    metrics_names = None
    for name in method_roots.keys():
        scores, metrics_names = compute_sample_pose_scores(
            results_dict[name],
            synset_names,
            allowed_class_ids=allowed_class_ids,
            degree_thresholds=[5, 10],
            shift_thresholds=[2, 5],
            iou_3d_thresholds=[0.1, 0.25, 0.5, 0.75],
            iou_pose_thres=0.1,
            iou_ratio_thresholds=IOU_RATIO_THRESHOLDS
        )
        scores_dict[name] = scores

    return results_dict, paths_dict, scores_dict, metrics_names


# ------------------------------------------------------------
# Top-K 筛选：用 iou_match_ratio@0.75 作为 method_score 做优势排序
# ------------------------------------------------------------

def find_topk_samples_advantage_multi(
    scores_dict,
    paths_dict,
    method1,
    method2,
    method3,
    allowed_class_ids,
    top_k=10
):
    scores_m1 = scores_dict[method1]
    scores_m2 = scores_dict[method2]
    scores_m3 = scores_dict[method3]

    # 这里用 iou_match_ratio@0.75（按GT匹配比例）
    # OVERLAP75_IDX = 10  # scores[..., 10] 是 iou_ratio > 0.75

    # def per_sample_overlap75(scores):
    #     # (N, K) -> (N,)
    #     sub = scores[:, allowed_class_ids, OVERLAP75_IDX]
    #     return np.nanmean(sub, axis=1)

    # score1 = per_sample_overlap75(scores_m1)
    # score2 = per_sample_overlap75(scores_m2)
    # score3 = per_sample_overlap75(scores_m3)
    
    MEAN_IOU_IDX = 10

    def per_sample_iou(scores):
        sub = scores[:, allowed_class_ids, MEAN_IOU_IDX]  # (N,K)
        return np.nanmean(sub, axis=1)                    # (N,)

    score1 = per_sample_iou(scores_m1)
    score2 = per_sample_iou(scores_m2)
    score3 = per_sample_iou(scores_m3)

    # 你原来的 delta 公式（尽量不改）
    deltas = 0.3 * (score3 - score2) + 0.7 * (score2 - score1)

    valid_idx = np.where(~np.isnan(deltas))[0]
    sorted_valid = valid_idx[np.argsort(deltas[valid_idx])[::-1]]
    topk_idx = sorted_valid[:top_k]

    pkl_paths_example = paths_dict[method3]
    topk_info = []

    for rank, s in enumerate(topk_idx):
        p = pkl_paths_example[s]
        scene_name = os.path.basename(os.path.dirname(p))
        sample_name = os.path.splitext(os.path.basename(p))[0]

        # 记录每个方法在该 sample 的 iou_ratio@{0.1,0.25,0.5,0.75}（对 allowed 类取 mean）
        method_iou_ratios = {}
        for m, sc in scores_dict.items():
            r01 = float(np.nanmean(sc[s, allowed_class_ids, 7]))
            r25 = float(np.nanmean(sc[s, allowed_class_ids, 8]))
            r50 = float(np.nanmean(sc[s, allowed_class_ids, 9]))
            r75 = float(np.nanmean(sc[s, allowed_class_ids, 10]))
            method_iou_ratios[m] = (r01, r25, r50, r75)

        topk_info.append({
            'rank': int(rank + 1),
            'index': int(s),
            'scene': scene_name,
            'sample': sample_name,
            'delta': float(deltas[s]),
            'method1_score': float(score1[s]),   # overlap75
            'method2_score': float(score2[s]),   # overlap75
            'method3_score': float(score3[s]),   # overlap75
            'method_iou_ratios': method_iou_ratios,
            'pkl_path': p,
        })

    return topk_info


# ------------------------------------------------------------
# 可视化（保持你原来的 bbox 代码）
# ------------------------------------------------------------

def to_points_8x2(projected_bbox):
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
    imgpts = np.int32(imgpts).reshape(-1, 2)

    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)

    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)
    return img


def _filter_result_to_allowed_classes(result, allowed_class_ids):
    out = dict(result)
    if 'gt_class_ids' in result and result['gt_class_ids'] is not None:
        gmask = np.isin(result['gt_class_ids'], allowed_class_ids)
        for k in ['gt_class_ids', 'gt_bboxes', 'gt_RTs', 'gt_scales', 'gt_handle_visibility']:
            if k in result and result[k] is not None:
                out[k] = np.asarray(result[k])[gmask]
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
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{image_name}_bbox.png')
    draw_image_bbox = image.copy()

    if draw_gt and gt_RTs is not None and gt_scales is not None:
        for ind, RT in enumerate(gt_RTs):
            bbox_3d = get_3d_bbox(gt_scales[ind], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            pts8 = to_points_8x2(projected_bbox)
            if pts8 is not None:
                draw_image_bbox = draw_bbox(draw_image_bbox, pts8, (0, 255, 0))

    if draw_pred and pred_class_ids is not None and len(pred_class_ids) > 0:
        num_pred_instances = len(pred_class_ids)

        if (gt_class_ids is not None and gt_bbox is not None and
            pred_bbox is not None and pred_scores is not None):
            try:
                gt_match, pred_match, _, pred_indices = compute_matches(
                    gt_bbox, gt_class_ids, gt_mask,
                    pred_bbox, pred_class_ids, pred_scores, pred_mask,
                    0.5
                )
                if len(pred_indices):
                    pred_RTs = pred_RTs[pred_indices]
                    pred_scales = pred_scales[pred_indices]
            except Exception:
                pass

        if pred_RTs is not None and pred_scales is not None:
            for ind in range(num_pred_instances):
                RT = pred_RTs[ind]
                bbox_3d = get_3d_bbox(pred_scales[ind, :], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                pts8 = to_points_8x2(projected_bbox)
                if pts8 is not None:
                    draw_image_bbox = draw_bbox(draw_image_bbox, pts8, (255, 0, 0))

    cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])


def visualize_topk_bboxes(
    topk_info,
    results_dict,
    intrinsics_dict,
    save_path_root,
    dataset_root,
    allowed_class_ids
):
    os.makedirs(save_path_root, exist_ok=True)

    for info in topk_info:
        idx = info['index']
        scene = info['scene']
        sample = info['sample']

        image_path = os.path.join(dataset_root, scene, 'rgb', f'{sample}.png')
        image = cv2.imread(image_path)[:, :, :3]
        image = image[:, :, ::-1]

        if scene not in intrinsics_dict:
            raise ValueError(f"scene {scene} not in intrinsics_dict")
        intrinsics = np.loadtxt(intrinsics_dict[scene]).reshape(3, 3)

        for m_name, final_results in results_dict.items():
            result = final_results[idx]
            result_f = _filter_result_to_allowed_classes(result, allowed_class_ids)

            out_dir = os.path.join(save_path_root, f'draw_{m_name}')
            os.makedirs(out_dir, exist_ok=True)

            # 用 overlap75 做文件名（更直观）
            r01, r25, r50, r75 = info['method_iou_ratios'][m_name]
            save_name = f"{scene}_{sample}_{m_name}_r75_{r75:.3f}_delta_{info['delta']:.3f}"

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
# 入口
# ------------------------------------------------------------

if __name__ == "__main__":
    test_scenes_rgb = sorted(glob.glob(os.path.join(dataset_root, 'test_scene*', 'rgb')))
    intrinsics_dict = {}
    for rgb_dir in test_scenes_rgb:
        scene_dir = os.path.dirname(rgb_dir)
        scene_name = os.path.basename(scene_dir)
        intrinsics_dict[scene_name] = os.path.join(scene_dir, 'intrinsics.txt')

    results_dict, paths_dict, scores_dict, metrics_names = collect_scores_for_methods(
        method_roots, HOUSECAT_SYNS, allowed_class_ids=ALLOWED_CLASS_IDS
    )

    topk_info = find_topk_samples_advantage_multi(
        scores_dict,
        paths_dict,
        method1=method1,
        method2=method2,
        method3=method3,
        allowed_class_ids=ALLOWED_CLASS_IDS,
        top_k=top_k
    )

    # 简单打印一下 top-k（每个方法的 iou_ratio@0.1/0.25/0.5/0.75）
    print(f"\nTop-{len(topk_info)} by overlap@0.75 advantage (allowed classes={ALLOWED_CLASS_NAMES}):")
    for t in topk_info[:min(10, len(topk_info))]:
        print(f"[{t['rank']:02d}] {t['scene']}/{t['sample']} idx={t['index']:05d}  Δ={t['delta']:.3f}")
        for m in [method1, method2, method3]:
            r01, r25, r50, r75 = t['method_iou_ratios'][m]
            print(f"    {m:>12s}: r@0.1={r01:.3f}, r@0.25={r25:.3f}, r@0.5={r50:.3f}, r@0.75={r75:.3f}")

    visualize_topk_bboxes(
        topk_info,
        results_dict,
        intrinsics_dict,
        vis_root,
        dataset_root,
        ALLOWED_CLASS_IDS
    )

    print(f"\nDone. Images saved to: {os.path.abspath(vis_root)}")
