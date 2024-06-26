import cv2
from PIL import Image
import numpy as np
from finetune import fine_tune
from generate_np import load_npz_data
import os 


## Example use case 1: loading image files 

train_masks = [] 
train_images = []

mask_dir = "./masks/iris/" ## change directory as necessary
images_dir = './images/' ## change directory as necessary
mask_files = os.listdir(mask_dir)
for file in mask_files: ## loop through dir
    name = '_'.join(file.split('.')[0].split('_')[:-1]) ## get filename
    mask_filepath = mask_dir + file
    mask_img = Image.open(mask_filepath)
    np_mask = np.asarray(mask_img) 
    train_masks.append(np_mask) 
    img_filepath = images_dir + name + '.jpg'
    image = Image.open(img_filepath)
    np_image = np.asarray(image)
    train_images.append(np_image)
        

fine_tune(train_images, train_masks, mode='iris')

## Example use case 2: loading from npz 
## TO-DO



