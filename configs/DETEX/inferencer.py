from mmdet.apis import DetInferencer

#可视化矩形框的线条粗细和字体调整在/data/wangshunyi/mmdetection/mmdet/visualization/local_visualizer.py中
# draw_bbox draw_texts
inferencer = DetInferencer(model='/data/wangshunyi/mmdetection/configs/DETEX/faster-rcnn_final.py', weights='/data/wangshunyi/mmdetection/work_dirs/faster-rcnn_final/best_coco_0.517_epoch_12.pth')
inferencer('/data/wangshunyi/mmdetection/data/DENTEX/new_enumeration_data/test/train_242.png', out_dir='/data/wangshunyi/mmdetection/outputs/', no_save_pred=False)  