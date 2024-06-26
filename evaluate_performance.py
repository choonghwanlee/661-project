import os
import cv2
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from finetune import generate_eval
from datasets import load_metric
from transformers import SamConfig, SamProcessor, SamModel
import torch

# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def compute_metrics(predicted_masks, true_masks, metric):
    with torch.no_grad():
        metrics = metric.compute(
            predictions=predicted_masks,
            references=true_masks,
            num_labels=2,
            ignore_index=255, #?? What is this? -Keith
            reduce_labels=False, #?? What is this? -Keith
        )
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics  

def evaluate_on_images(images, true_masks, modelCheckpointFilePath):
    print(f"modelCheckpoint: {modelCheckpointFilePath}")
    '''
    Parameters:
        -images a list of np arrays, where each entry in the list is the np array representation of an image
        -true_masks a list of boolean np arrays, where the ith entry represents the true mask of the ith entry in images

    Returns:
        -a list of floating point IOU values - the ith entry in the list is the IOU on image i

    '''
    assert len(images) == len(true_masks), "Length of images & masks don't match"
    predictions = generate_eval(images, modelCheckpointFilePath)
    metric = evaluate.load("mean_iou")
    iou_vals = [0 for _ in range(len(images))]
      
    for image_index in range(len(images)):
        true_mask = true_masks[image_index]
        if true_mask.ndim == 3:
            true_mask = cv2.cvtColor(true_mask, cv2.COLOR_RGB2GRAY)
        true_mask = np.reshape(true_mask, (1, true_mask.shape[0], true_mask.shape[1]))
        pred_mask =  np.reshape(predictions[image_index], (1, predictions[image_index].shape[0], predictions[image_index].shape[1]))
        metrics = compute_metrics(pred_mask, true_mask, metric=metric)

        intersection = np.logical_and(true_mask, pred_mask)
        union = np.logical_or(true_mask, pred_mask)
        iou = np.sum(intersection) / np.sum(union)
        iou_vals[image_index] = iou
        
    return iou_vals


def visualize_masks(image, masks):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the first image on the left
    axes[0].imshow(np.array(image))  # Assuming the first image is grayscale
    axes[0].set_title("Image")

    # Plot the second image on the right
    axes[1].imshow(masks, cmap='gray')  # Assuming the second image is grayscale
    axes[1].set_title("Masks")

    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Display the images side by side
    plt.show()

## example usage 

"""
ious = evaluate_on_images(test_images, test_masks, "./models/base_model_checkpoint.pth")
mean_iou = sum(ious)/len(ious)
print(mean_iou)
"""

## evaluate performance on baseline model using the test dataset