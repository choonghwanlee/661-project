import cv2
from PIL import Image
# from finetune import fine_tune
from generate_np import load_npz_data

### load images and masks as PIL images 
train_images = load_npz_data('./gangloooooon/labeled/images.npz')
train_masks = load_npz_data('./gangloooooon/labeled/masks.npz')
# test_images = None
# test_masks = None 

print(train_masks)
# fine_tune(train_images['images'], train_masks['iris'], mode='iris')


