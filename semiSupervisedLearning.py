import os
import cv2
import numpy as np
from finetune import fine_tune, generate_eval
from evaluate_performance import evaluate_on_images
import random

# Define paths to images and masks folders
# Get the absolute path of the current Python script
MODE = "iris"
script_dir = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(script_dir, "models")
ckpt_folder = os.path.join(models_folder, "semiSupervised")
labeled_folder = os.path.join(script_dir, "ganglion-ece661", "ganglion-ece661", "labeled")
unlabeled_folder = os.path.join(script_dir, "unlabelled")
labeled_images_folder = os.path.join(labeled_folder, "test_images")
unlabeled_images_folder = os.path.join(unlabeled_folder, "images")
masks_folder = os.path.join(labeled_folder, "masks", MODE)

labeled_image_paths = os.listdir(labeled_images_folder)
unlabeled_image_paths = os.listdir(unlabeled_images_folder)
masks_paths = os.listdir(masks_folder)
random.shuffle(labeled_image_paths)
random.shuffle(unlabeled_image_paths)

test_image_paths = unlabeled_image_paths

def get_labeled_mask(image_name, mode="iris"):
    image_name = os.path.splitext(image_name)[0]         
    # Load corresponding mask
    mask_filename = f"{image_name}_{mode}.png"
    mask_path = os.path.join(masks_folder, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return
    return mask


# Iterate over each file in the images folder
def get_images_and_masks(image_paths, get_mask):
    images = []
    masks = []
    for filename in image_paths:
        if filename.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(labeled_images_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Unable to read image {filename}")
                continue
            
            # Extract image name without extension
            image_name = os.path.splitext(filename)[0]

            mask = get_mask(image_name, image)
            if (mask is None):
                continue
            
            # Append image and mask to lists
            images.append(image)
            masks.append(mask)
    return images, masks
        
labeled_images, labeled_masks = get_images_and_masks(labeled_image_paths, get_mask = lambda image_path, image : get_labeled_mask(image_path, mode=MODE))

TEST_SPLIT = 0.5
test_images = labeled_images[TEST_SPLIT:]
test_masks = labeled_masks[TEST_SPLIT:]
print(len(labeled_images))
print(len(labeled_masks))

model_save_ckpt = os.path.join(ckpt_folder, f"{MODE}_model_0_checkpoint.pth")

model = fine_tune(labeled_images[:TEST_SPLIT], labeled_masks[:TEST_SPLIT], model_load_ckpt=None, model_save_ckpt=model_save_ckpt, mode=MODE, num_epochs=1)
evaluate_on_images(images=test_images, true_masks=labeled_masks)

NUM_TRAIN_SPLITS = 10
for split in range(NUM_TRAIN_SPLITS):
    lower_bound = split * len(unlabeled_image_paths) / NUM_TRAIN_SPLITS
    upper_bound = lower_bound + len(unlabeled_image_paths) / NUM_TRAIN_SPLITS
    train_images, train_masks = get_images_and_masks(unlabeled_image_paths[lower_bound : upper_bound], get_mask = lambda image_path, image : generate_eval(images=[image], modelCheckpointFilePath="", model=model))
    model_load_ckpt = os.path.join(ckpt_folder, f"{MODE}_model_{str(split)}_checkpoint.pth")
    model_save_ckpt = os.path.join(ckpt_folder, f"{MODE}_model_{str(split + 1)}_checkpoint.pth")
    model = fine_tune(images=train_images, pred_masks=train_masks, model_load_ckpt=model_load_ckpt, model_save_ckpt=model_save_ckpt, mode=MODE, num_epochs=4, batch_size=16)
    evaluate_on_images(images=test_images, true_masks=labeled_masks, modelCheckpointFilePath=model_save_ckpt)    
