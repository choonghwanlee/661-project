import numpy as np
from transformers import SamConfig, SamProcessor, SamModel
from PIL import Image, ImageOps, ImageFilter
import torch
import cv2
import os 
import math
from scipy.interpolate import interp1d

def _find_threshold(image):
    """
    determine optimal threshold for image frame. 
    calculates quantile function of image, finds quantile value where first derivative decreases
    
    Paramaters
    ---------
    image: np.array
        np.array representation of the image
    """
    ## TODO: to include or not include CLAHE hist equalization?
    ## find the histogram of the grayscale image
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ## find the CDF of the normalized PDF
    pdf = hist / sum(hist)
    cdf = np.cumsum(pdf)
    inverse_cdf = interp1d(cdf, [i for i in range(1,257)], kind='linear', fill_value='extrapolate')
    ## define baseline optimal threshold
    optimal_threshold = min(hist.flatten())
    ## initialize values
    prev_slope = inverse_cdf(0.002) - inverse_cdf(0)/0.1
    prev_threshold = inverse_cdf(0.002)
    for i in np.arange(0.004,1,0.002):
        ## value of the quantile function
        # x.append(i)
        # y.append(inverse_cdf(i))
        potential_threshold = inverse_cdf(i)
        ## calculate first derivative using point-to-point difference 
        slope = (potential_threshold - prev_threshold)/0.1
        ## if difference in slopes < 0, set the previous threshold as the optimal threshold value   
        if slope - prev_slope < 0:
            optimal_threshold = prev_threshold 
            # print('Threshold Value:', optimal_threshold)
            return optimal_threshold
        else: 
            # set prev to current
            prev_threshold = potential_threshold
            prev_slope = slope
    return optimal_threshold


def _are_points_in_bounding_box(contour, bounding_box):
    x, y, width, height = bounding_box
    
    for point in contour:
        px, py = point[0]  # point is in the form [[x, y]]
        
        if not (x <= px <= x + width and y <= py <= y + height):
            return False
    
    return True

def calculate_iou(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    
    # Count white pixels (intersection) and total pixels (union)
    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)
    
    # Calculate IoU (Intersection over Union)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def find_union_if_iou_above_threshold(mask1, mask2, threshold=0.3):
    iou = calculate_iou(mask1, mask2)
    
    if iou > threshold:
        # Compute the union of masks
        union_mask = cv2.bitwise_or(mask1, mask2)
        return union_mask
    else:
        return mask2