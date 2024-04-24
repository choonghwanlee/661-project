import os
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import evaluate
from datasets import load_metric
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

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


def evaluate_on_images(images, true_masks, modelCheckpointFilePath="./weights/mobile_sam.pt", model_type="vit_t", device="cpu" ):
    '''
    Parameters:
        -images a list of np arrays, where each entry in the list is the np array representation of an image
        -true_masks a list of boolean np arrays, where the ith entry represents the true mask of the ith entry in images

    Returns:
        -a list of floating point IOU values - the ith entry in the list is the IOU on image i

    '''
    #make the predictions, then call compute_metrics
    sam = sam_model_registry[model_type](checkpoint=modelCheckpointFilePath)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    
    metric = evaluate.load("mean_iou")
    iou_vals = [0 for i in range(len(images))]

                
    for image_index in range(len(images)):
        image = images[image_index]
        true_mask = true_masks[image_index]
        true_mask = np.reshape(true_mask, (1, true_mask.shape[0], true_mask.shape[1]))

        
        predictor.set_image(image)
        pred_mask, scores, logits = predictor.predict(
            multimask_output=False,
        )

        metrics = compute_metrics(true_mask, pred_mask, metric=metric)
        iou_vals[image_index] = metrics["mean_iou"]
        
    return iou_vals

