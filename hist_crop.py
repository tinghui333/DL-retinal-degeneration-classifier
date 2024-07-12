import os
from glob import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random
from tqdm import tqdm
import time


def main():
    # Set up data path
    root_dir = './data/Histology_Images_CV/'
    date_list = ['AI_P20', 'AI_P40', 'AI_P60', 'AI_P90', 'AI_P120', 'AI_P150']
    save_dir = os.path.join(root_dir, 'patch_images')

    # Set up parameters
    output_size = 1000
    shrink_ratio = 10
    shrink_size = output_size // shrink_ratio
    crops_per_image = 10
    time_limit = 180

    fail_case = []

    for date in date_list:
        print(date)
        if not os.path.isdir(os.path.join(save_dir, date)):
            os.mkdir(os.path.join(save_dir, date))

        img_list = glob(os.path.join(root_dir, date, '*', '*', '*.tif'))

        for img_path in tqdm(img_list):
            img_ID = img_path.split('/')[-1].split('.')[0]
            image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

            # Step 1: downsample image
            image_downsampled = downsample(image, shrink_ratio)
            h, w = image_downsampled.shape[:2]

            # Step 2: flood fill to turn the background white
            fill_mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(image_downsampled, fill_mask, (0, 0), (225, 225, 225))
            cv2.floodFill(image_downsampled, fill_mask, (w-1, 0), (225, 225, 225))
            cv2.floodFill(image_downsampled, fill_mask, (0, h//2), (225, 225, 225))
            cv2.floodFill(image_downsampled, fill_mask, (0, h-1), (225, 225, 225))
            cv2.floodFill(image_downsampled, fill_mask, (w-1, h-1), (225, 225, 225))
            ori_image_downsampled = image_downsampled.copy()

            # Step 3: block out the upper left and lower left corner
            crop_w = w // 4
            crop_h = h // 5
            image_downsampled[:crop_h, :crop_w, :] = (225, 225, 225)
            image_downsampled[-crop_h:, :crop_w, :] = (225, 225, 225) 

            # Step 4: reverse the image
            image_downsampled = cv2.bitwise_not(image_downsampled)

            # Step 5: generate 3*planning boxes
            box_idx = []

            lower_color = np.array([125, 150, 150])
            upper_color = np.array([175, 225, 225])
            color_mask = cv2.inRange(image_downsampled, lower_color, upper_color)
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.erode(color_mask, kernel, iterations=1)
            color_mask = cv2.dilate(color_mask, kernel, iterations=2)

            mid_h = random.randint(0, h)
            mid_w = random.randint(0, w)
            top, bottom, left, right = mid_h - shrink_size//2, mid_h + shrink_size//2, mid_w - shrink_size//2, mid_w + shrink_size//2

            start_time = time.time()
            while len(box_idx)<crops_per_image*3:
                if time.time() - start_time > time_limit:
                    print(f"Couldn't find patches in {img_ID}")
                    break
                if top >=0 and bottom <= h and left >=0 and right<=w:
                    patch = image_downsampled[top:bottom, left:right]
                    mask_patch = color_mask[top:bottom, left:right]
                    
                    if (mask_patch.sum() / (shrink_size*shrink_size*225) > 0.2) and ((patch < 10).sum() < 10):
                        box_idx.append([top, bottom, left, right, mask_patch.sum() / (shrink_size*shrink_size*225)])

                mid_h = random.randint(0, h)
                mid_w = random.randint(0, w)
                top, bottom, left, right = mid_h - shrink_size//2, mid_h + shrink_size//2, mid_w - shrink_size//2, mid_w + shrink_size//2

            if len(box_idx) == 0:
                fail_case.append(f"{date}_{img_ID}")
                continue

            # Step 6: remove overlapping boxes
            box_idx = remove_overlapping_boxes(box_idx, 0.2)
            if len(box_idx) > crops_per_image:
                box_idx.sort(key=lambda x: x[-1], reverse=True)
                box_idx = box_idx[:crops_per_image]

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
            plt.savefig(os.path.join(save_dir, date, f'map_{img_ID}.png'))
            plt.close()

            for idx, square in enumerate(box_idx):
                top, bottom, left, right, _ = square
                crop_img = image[top*shrink_ratio:bottom*shrink_ratio, left*shrink_ratio:right*shrink_ratio]
                cv2.imwrite(os.path.join(save_dir, date, f"{img_ID}_{idx}.png"), crop_img)

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
        if i in removed_indices:
            continue
        
        for j in range(i + 1, len(boxes)):
            if j in removed_indices:
                continue
            
            box2 = boxes[j]
            # Calculate overlap area between box1 and box2
            overlap_area = calculate_overlap_area(box1, box2)
            
            # Calculate overlap ratio
            overlap_ratio = overlap_area / (box_area*2 - overlap_area)
            
            # Check if overlap ratio exceeds threshold
            if overlap_ratio > overlap_threshold:
                removed_indices.add(j)
        
        if i not in removed_indices:
            boxes_to_keep.append(box1)
    
    return boxes_to_keep


if __name__ == '__main__':
    main()