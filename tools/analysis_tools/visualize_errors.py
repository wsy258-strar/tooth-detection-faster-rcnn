# 可视化每张图片被分类错误的样本的矩形框
import argparse
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root
from mmdet.evaluation import bbox_overlaps


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize misclassified detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('prediction_path', help='path to test .pkl result')
    parser.add_argument('save_dir', help='directory to save visualized images')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold')
    parser.add_argument('--tp-iou-thr', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config')
    args = parser.parse_args()
    return args


def visualize_misclassifications(dataset, results, score_thr, tp_iou_thr, save_dir):
    for img_idx in range(len(dataset)):
        gt_info = dataset.get_data_info(img_idx)
        gt_bboxes = []
        gt_labels = []
        for inst in gt_info['instances']:
            gt_bboxes.append(inst['bbox'])
            gt_labels.append(inst['bbox_label'])
        gt_bboxes = np.array(gt_bboxes)
        gt_labels = np.array(gt_labels)

        pred_labels = results[img_idx]['pred_instances']['labels'].numpy()
        pred_bboxes = results[img_idx]['pred_instances']['bboxes'].numpy()
        pred_scores = results[img_idx]['pred_instances']['scores'].numpy()

        img_path = gt_info['img_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for gt_label, gt_bbox in zip(gt_labels, gt_bboxes):
            gt_bbox = gt_bbox.astype(int)
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f'GT: {dataset.metainfo["classes"][gt_label]}', (gt_bbox[0], gt_bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for pred_label, pred_bbox, pred_score in zip(pred_labels, pred_bboxes, pred_scores):
            if pred_score < score_thr:
                continue
            ious = bbox_overlaps(np.array([pred_bbox[:4]]), gt_bboxes)[0]
            max_iou = np.max(ious) if len(ious) > 0 else 0
            matched_gt = np.argmax(ious) if len(ious) > 0 else -1
            if max_iou >= tp_iou_thr and matched_gt != -1 and pred_label == gt_labels[matched_gt]:
                continue  # Correct classification
            pred_bbox = pred_bbox.astype(int)
            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
            cv2.putText(img, f'Pred: {dataset.metainfo["classes"][pred_label]} ({pred_score:.2f})', (pred_bbox[0], pred_bbox[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'misclassified_{img_idx}.jpg'), bbox_inches='tight', dpi=300)
        plt.close()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    results = load(args.prediction_path)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    visualize_misclassifications(dataset, results, args.score_thr, args.tp_iou_thr, args.save_dir)


if __name__ == '__main__':
    main()
# 执行命令： python tools/analysis_tools/visualize_errors.py /data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_final.py ./test.pkl ./error-results/bbox