import preprocess.config as config
import mmcv
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
os.sys.path.append('./mmdetection')

import mmdet
from mmdet.apis import init_detector, inference_detector

cfg = config.cfg
checkpoint_file = './mmdetection/tutorial_exps/epoch_12.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(cfg, checkpoint_file, device=device)


def find_pick_points(segmentation_masks):
    pick_points = []

    for idx, mask_tensor in enumerate(segmentation_masks):
        mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255  # Convert to uint8 for OpenCV functions

        # Switch Segmentation
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the rotated bounding rectangle of the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Calculate the longest side of the bounding box
            side1 = np.linalg.norm(box[0] - box[1])
            side2 = np.linalg.norm(box[1] - box[2])

            if side1 > side2:
                long_side = box[0] - box[1]
                perpendicular_direction = np.array([-long_side[1], long_side[0]], dtype=float)  # Specify dtype=float
            else:
                long_side = box[1] - box[2]
                perpendicular_direction = np.array([-long_side[1], long_side[0]], dtype=float)  # Specify dtype=float

            perpendicular_direction /= np.linalg.norm(perpendicular_direction)
            

            theta = np.arctan2(perpendicular_direction[0], perpendicular_direction[1]) * 180 / np.pi

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                pick_points.append((cx, cy, theta))

    return pick_points
if __name__ == "__main__":
    with open('./data/test_ids.txt', 'r') as f:
        test_ids_str = f.read()
        test_ids_list = test_ids_str.split('\n')

    for test_id in tqdm(test_ids_list):
        test_file_sample = f'./data/images/{test_id}.png'
        img = mmcv.imread(test_file_sample,channel_order='rgb')
        result = inference_detector(model, img)
        pred_score_thr = 0.5
        pred_scores = result.pred_instances.scores
        selected_indices = torch.where(pred_scores >= pred_score_thr)[0]
        filtered_scores = pred_scores[selected_indices]
        filtered_masks = result.pred_instances.masks[selected_indices]
        filtered_labels = result.pred_instances.labels[selected_indices]
        filtered_bboxes = result.pred_instances.bboxes[selected_indices]
        filtered_masks_in_top = [mask for mask, label in zip(filtered_masks, filtered_labels) if label == 0]
        pick_points = find_pick_points(filtered_masks_in_top)
        
        pick_points_str = '\n'.join([' '.join(map(str, item)) for item in pick_points])
        with open(f'./result/part_1/{test_id}.txt', 'w') as f:
            f.write(pick_points_str)