import numpy as np
from finetune import fine_tune, get_bounding_box, postprocess_masks, generate_eval
import weak_helpers as utils
from transformers import SamConfig, SamProcessor, SamModel
from PIL import Image, ImageOps, ImageFilter
import torch
import cv2
import os 
import math
from scipy.interpolate import interp1d
import time 

'''
Weakly Supervised Learning:
    -Combines several methods:
        - using fine-tuned SAM + post-processing to generate pseudo-labels
    - uses pseudo-labels as real labels for fine-tuning
'''

## step 1: load unlabelled data into numpy arrays
unlabelled = []
unlabelled_dir = "./unlabelled/" ## change directory as necessary
image_files = os.listdir(unlabelled_dir)
for file in image_files: ## loop through dir
    img_filepath = unlabelled_dir + file
    if os.path.isfile(img_filepath) and all(substring not in file for substring in ['christian_lefteye','cindy_lefteye']) and file != '.DS_Store':
        img = Image.open(img_filepath)
        np_img = np.asarray(img) 
        unlabelled.append(np_img)

## step 2: run fine-tuned SAM on batch of images
start_time = time.time()
predictions = generate_eval(unlabelled, './models/iris_model_checkpoint.pth')


## step 3: post-processing of masks generated 

## helper function for below

## main post-processing loop
sam_pred_masks = []
for pred_mask in predictions: 
    max_area = -1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    eroded_mask = cv2.erode(pred_mask, kernel, iterations=2) ## remove noise from model prediction
    eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel) ## remove noise from model prediction
    iris_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    iris_center, iris_radius = None, None 
    for contour in iris_contours:
        convex_closed = cv2.convexHull(contour, False)
        perimeter = cv2.arcLength(convex_closed, True)
        area = cv2.contourArea(convex_closed)
        if perimeter == 0:
            continue
        circularity = (4*math.pi*area)/(perimeter*perimeter)
        ## check the contour circularity is high, area is big, and all points are inside bounding_box
        if circularity > 0.8 and area > max_area and utils._are_points_in_bounding_box(convex_closed, rect):
            max_area = area
            approx = cv2.approxPolyDP(convex_closed, perimeter * 0.034, True)
            iris_center, iris_radius = cv2.minEnclosingCircle(approx)
    filtered_mask = np.zeros((1920, 1080), dtype=np.uint8)
    if iris_center:
        cv2.circle(filtered_mask, (int(iris_center[0]),int(iris_center[1])), int(iris_radius), (255), -1)  
        sam_pred_masks.append(filtered_mask)
    else:
        sam_pred_masks.append(eroded_mask)

## step 4 (not included in final work): ensemble of masks with image processing algorithm

frame_size = (1080, 1920)
threshold_pred_masks = []
proj_eye_center = (frame_size[0]/2, frame_size[1]/3)
proj_eye_radius = (0.25*frame_size[0], 0.07*frame_size[1])
y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])
rect = (x_start, y_start, x_end - x_start, y_end - y_start)
## loop through each image
for image in unlabelled:
    temp = image.copy()
    ## grayscale
    red_channel = temp[:, :, 2]
    ## historgram equalization 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
    standardized = clahe.apply(red_channel)
    # crop by bounding box prompt
    filtered = standardized[y_start:y_end, x_start:x_end]
    ## retrieve optimal binary threshold and binarize
    optimal_threshold = math.ceil(utils._find_threshold(filtered)) + 30
    _, iris_mask = cv2.threshold(standardized, optimal_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, kernel)
    iris_contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    iris_center, iris_radius = None, None 
    for contour in iris_contours:
        convex_closed = cv2.convexHull(contour, False)
        perimeter = cv2.arcLength(convex_closed, True)
        area = cv2.contourArea(convex_closed)
        if perimeter == 0:
            continue
        circularity = (4*math.pi*area)/(perimeter*perimeter)
        ## check the contour circularity is high, area is big, and all points are inside bounding_box
        if circularity > 0.7 and area > max_area and utils._are_points_in_bounding_box(convex_closed, rect):
            max_area = area
            approx = cv2.approxPolyDP(convex_closed, perimeter * 0.034, True)
            iris_center, iris_radius = cv2.minEnclosingCircle(approx)
    filtered_mask = np.zeros((1920, 1080), dtype=np.uint8)
    if iris_center:
        cv2.circle(filtered_mask, (int(iris_center[0]),int(iris_center[1])), int(iris_radius), (255), -1)  
        threshold_pred_masks.append(filtered_mask)
    else:
        threshold_pred_masks.append(iris_mask)

## step 5 (not included in final work): generate ensemble
ensemble_mask = [utils.find_union_if_iou_above_threshold((threshold_pred_masks[i]/255).astype('uint8'), sam_pred_masks[i]) for i in range(len(sam_pred_masks))]
ensemble_mask = [ensemble_mask[i]*255 for i in range(len(ensemble_mask))]

## step 6: fine-tuned with pseudo-masks
fine_tune(unlabelled, sam_pred_masks, mode='iris')

end_time = time.time() - start_time
print(end_time)

