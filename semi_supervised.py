import os
import cv2
import numpy as np
from finetune import fine_tune, generate_eval, generate_eval_segprob
from evaluate_performance import evaluate_on_images
import random
from transformers import SamModel
import torch

# Define paths to images and masks folders
# Get the absolute path of the current Python script
MODE = "iris"
script_dir = os.path.dirname(".")
models_folder = os.path.join(script_dir, "models")
semi_supervised_ckpt_folder = os.path.join(models_folder, "semiSupervised")
labeled_folder = os.path.join(script_dir, "ganglion-ece661", "ganglion-ece661", "labeled")
unlabeled_folder = os.path.join(script_dir, "ganglion-ece661", "ganglion-ece661", "unlabelled")
labeled_images_folder = os.path.join(labeled_folder, "test_images")
unlabeled_images_folder = os.path.join(unlabeled_folder, "images")
masks_folder = os.path.join(labeled_folder, "masks", MODE)

labeled_image_paths = os.listdir(labeled_images_folder)
unlabeled_image_paths = os.listdir(unlabeled_images_folder)
l1 = len(unlabeled_image_paths)
unlabeled_image_paths = list(filter(lambda name : not (("christian_lefteye" in name) or ("cindy_lefteye" in name)), unlabeled_image_paths))
l2 = len(unlabeled_image_paths)
masks_paths = os.listdir(masks_folder)
random.shuffle(labeled_image_paths)
random.shuffle(unlabeled_image_paths)

image_path_to_image = {}
image_paths_to_add_to_training = set(unlabeled_image_paths)

def get_labeled_mask(image_name, mask_paths, mode="iris"):
    image_name = os.path.splitext(image_name)[0]
    # Load corresponding mask
    mask_filename = f"{image_name}_{mode}.png"
    if not (mask_filename in mask_paths):
        return

    mask_path = os.path.join(masks_folder, mask_filename)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return
    return mask



# Iterate over each file in the images folder
def get_images_and_masks(image_names, get_mask, base_path=labeled_images_folder):
    global image_path_to_image
    images = []
    masks = []
    for filename in image_names:
        if (len(images) % 10 == 0):
            print(len(images))
        if filename.endswith(".jpg"):
            # Load the image
            image = None
            if filename in image_path_to_image.keys():
                image = image_path_to_image[filename]
            else:
                image_path = os.path.join(base_path, filename)
                image = cv2.imread(image_path)
                image_path_to_image[filename] = image

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


        
labeled_images, labeled_masks = get_images_and_masks(labeled_image_paths, base_path=labeled_images_folder, get_mask = lambda image_path, image : get_labeled_mask(image_path, mask_paths=masks_paths , mode=MODE))
unlabeled_images, _ = get_images_and_masks(unlabeled_image_paths, base_path = unlabeled_images_folder, get_mask = lambda image_Path, image : np.array([2]))

def get_score_of_seg_prob_mask(seg_prob_mask):
    min = np.mininum(seg_prob_mask, 1.0 - seg_prob_mask)
    min_max = np.maximum(seg_prob_mask, min)
    return np.mean(min_max.flatten())



def get_training_data(model):
    global MODE, image_path_to_image
    image_path_to_pred_mask = {}
    image_path_to_pred_mask_score = {}
    scores = []
    for image_name in list(image_paths_to_add_to_training):
        #generate mask for each, take the top 1/3
        image = image_path_to_image[image_name]
        seg_prob_mask = generate_eval_segprob(images = [image], model=model, mode=MODE, modelCheckpointFilePath="")[0]
        score = get_score_of_seg_prob_mask(seg_prob_mask)
        image_path_to_pred_mask_score[image_name] = score
        scores.append(score)
        image_path_to_pred_mask[image_name] = (seg_prob_mask > 0.6).astype(np.uint8)
    scores.sort()
    threshold = scores[min(50, len(scores) - 1)]

    new_train_images = []
    new_train_masks = []
    names = list(image_paths_to_add_to_training)
    for image_name in names:
        if (image_path_to_pred_mask_score[image_name] >= threshold):
            image_paths_to_add_to_training.remove(image_name)
            new_train_images.append(image_path_to_image[image_name])
            new_train_images.append(image_path_to_pred_mask[image_name])
    return new_train_images, new_train_masks





train_images = []
train_masks = []
test_images = labeled_images
test_masks = labeled_masks


base_model_load_ckpt = os.path.join(models_folder, "iris_model_checkpoint.pth")
base_model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
base_model.load_state_dict(torch.load(base_model_load_ckpt))
model = base_model


NUM_TRAIN_SPLITS = 10
for split in range(NUM_TRAIN_SPLITS):
    print("length of training data is now " + str(len(train_images)))
    new_train_images, new_train_masks = get_training_data(model)
    #only use the masks with high confidence?
    train_images.extend(new_train_images)
    train_masks.extend(new_train_masks)
    
    model_save_ckpt = f"./models/semiSupervised_{MODE}_model_checkpoint.pth"
    fine_tune(images=train_images, pred_masks=train_masks, mode=MODE, checkpoint_info='semiSupervised', modelCheckpointFilePath = base_model_load_ckpt, BATCH_SIZE=16, num_epochs=4)
    model.load_state_dict(torch.load(model_save_ckpt))
    print("evaluating on test set")
    evaluate_on_images(images=test_images, true_masks=labeled_masks, modelCheckpointFilePath=model_save_ckpt)    
    if len(image_paths_to_add_to_training) < 20:
        break