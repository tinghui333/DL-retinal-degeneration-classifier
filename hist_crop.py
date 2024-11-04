import os
from glob import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random


def main():
    # Set up data path
    root_dir = './data/Histology_Images_CV'
    date_list = ['AI_P20', 'AI_P40', 'AI_P60', 'AI_P90', 'AI_P120', 'AI_P150']
    save_dir = os.path.join(root_dir, 'patch_images')

    # Set up parameters
    output_size = 1000 # desired output image size
    n_patch_per_image = 10 # desired number of output patches per image
    patch_overlap_rate = 0.5 # cropped patch overlap rate
    shrink_ratio = 10 # for faster operation

    shrink_patch_size = output_size // shrink_ratio
    shrink_patch_shift = int(np.floor(shrink_patch_size * patch_overlap_rate))

    fail_case = []
    rows = []

    for date in date_list:
        path_dict = defaultdict(int)
        print(date)
        if not os.path.isdir(os.path.join(save_dir, date)):
            os.makedirs(os.path.join(save_dir, date))

        img_list = glob(os.path.join(root_dir, date, '*', '*', '*.tif'))

        for img_path in img_list:
        # for img_path in tqdm(img_list):
            case_id = img_path.split('/')[-3]
            child_path = '/'.join(img_path.split('/')[-2:])
            path_dict[case_id] += 1

            image = cv2.imread(img_path)

            # Step 1: downsample image
            image_downsampled = downsample(image, shrink_ratio)
            ori_image_downsampled = cv2.cvtColor(image_downsampled, cv2.COLOR_BGR2RGB)
            image_downsampled = cv2.cvtColor(image_downsampled, cv2.COLOR_BGR2HSV)

            height, width, _ = np.shape(image_downsampled)

            # Step 2: generate blue area mask
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(image_downsampled, lower_blue, 
                               upper_blue)
            
            box_idx = []
            thr = 0.1
            if date in ('AI_P120', 'AI_P150'):
                thr = 0.05

            # Step 3: crop patches
            max_thr = 0
            for i in range(height // shrink_patch_shift):
                for j in range(width // shrink_patch_shift):
                    bottom = min(i*shrink_patch_shift + shrink_patch_size, height)
                    right = min(j*shrink_patch_shift + shrink_patch_size, width)
                    top = bottom - shrink_patch_size
                    left = right - shrink_patch_size

                    patch = mask[top:bottom, left:right]
                    this_thr = np.sum(patch > 0) / (shrink_patch_size*shrink_patch_size)
                    max_thr = max(max_thr, this_thr)
                    if this_thr > thr:
                        box_idx.append([top, bottom, left, right, this_thr])
            ori_n_box = len(box_idx)
            if len(box_idx) == 0:
                fail_case.append((f"{date}_{case_id}", max_thr))
                continue
            
            # Step 4: remove overlapping patches
            box_idx = remove_overlapping_boxes(box_idx, 0.1)
            rm_overlap_n_box = len(box_idx)

            if len(box_idx) > n_patch_per_image:
                box_idx = random.sample(box_idx, n_patch_per_image)

            fig, ax = plt.subplots()
            plt.margins(0, 0)
            ax.imshow(ori_image_downsampled, cmap='gray')
            for square in box_idx:
                top, bottom, left, right, ratio = square
                width = right - left
                height = bottom - top
                rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, date, f'map_{case_id}_{path_dict[case_id]}.png'))
            plt.close()

            for idx, square in enumerate(box_idx):
                top, bottom, left, right, _ = square
                crop_img = image[top*shrink_ratio:bottom*shrink_ratio, left*shrink_ratio:right*shrink_ratio]
                cv2.imwrite(os.path.join(save_dir, date, f"{case_id}_{path_dict[case_id]}_{idx}.png"), crop_img)

            print(f'Date: {date}, {case_id}-{path_dict[case_id]} got {ori_n_box}->{rm_overlap_n_box}->{len(box_idx)} images')

            new_row = {'date': date, 'case_id': case_id, 'ori_img_path': child_path, 'given_id': path_dict[case_id]}
            rows.append(new_row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f'save_path.csv'), index=False)

    print(fail_case)
    

def downsample(mask, factor):
    height, width, _ = mask.shape
    new_height = height // factor
    new_width = width // factor
    
    downsampled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    return downsampled_mask


def calculate_overlap_area(box1, box2):
    # box1 and box2 are tuples or lists containing (x1, y1, x2, y2) coordinates
    x1_overlap = max(box1[0], box2[0])
    x2_overlap = min(box1[1], box2[1])
    y1_overlap = max(box1[2], box2[2])
    y2_overlap = min(box1[3], box2[3])
    
    # Calculate width and height of overlap area
    overlap_width = max(0, x2_overlap - x1_overlap)
    overlap_height = max(0, y2_overlap - y1_overlap)
    
    # Calculate overlap area
    overlap_area = overlap_width * overlap_height
    
    return overlap_area


def remove_overlapping_boxes(boxes, overlap_threshold):
    boxes_to_keep = []
    removed_indices = set()
    box_area = calculate_overlap_area(boxes[0], boxes[0])
    
    for i, box1 in enumerate(boxes):
        if i in removed_indices or box1 in boxes_to_keep:
            continue
        
        curr_candidate = (i, box1.copy())
        for j in range(i + 1, len(boxes)):
            if j in removed_indices:
                continue
            
            box2 = boxes[j]
            if box2 in boxes_to_keep:
                continue

            # Calculate overlap area between box1 and box2
            overlap_area = calculate_overlap_area(curr_candidate[1], box2)
            
            # Calculate overlap ratio
            overlap_ratio = overlap_area / (box_area*2 - overlap_area)
            
            # Check if overlap ratio exceeds threshold
            if overlap_ratio > overlap_threshold:
                if curr_candidate[1][4] > box2[4]:
                    removed_indices.add(j)
                else:
                    removed_indices.add(curr_candidate[0])
                    curr_candidate = (j, box2.copy())
                    
        boxes_to_keep.append(curr_candidate[1])
    
    return boxes_to_keep


if __name__ == '__main__':
    main()