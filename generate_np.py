import numpy as np
import cv2
from PIL import Image
import os
import os


"""
converts an image to a numpy array

Parameters:
    - filepath (String): path to the image, either absolute or relative to current user directory

Returns:
    - numpy array
"""
def image_to_np(filepath):
    #using opencv
    print(filepath)
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    nparray_cv2 = np.asarray(img)

    return nparray_cv2


"""
Converts a list of images to numpy arrays and saves them in a .npz file

Parameters:
    - image_filenames ([String]): an array of strings that represent the image names.
        These images should be located in the images folder and have .jpg extension
        - ie. "im" should be passed in for an image with path ./images/im.jpg
    - save_path (String): absolute or relative (to current directory) path for the .npz 
        file to be saved to. Do not include the .npz extension 

Returns:
    - String representing the compelte file path
"""
def images_to_npz(image_filenames, save_path):
    np_arrays = []

    for f in image_filenames:
        np_arrays.append(image_to_np(f"./images/{f}.jpg"))
    
    #option 1: dictionary with 1 key whose value is an array of N np arrays
    np.savez("./{save_path}", np_arrays)

    #option 2: dictionary with N keys whose value is an np array
    # np.savez("./train_data/unlabelled/test_savez", *np_arrays[:])

    return save_path+".npz"


"""
Loads data from a .npz file

Parameters:
    - filename (String): absolute filepath to the .npz file

Returns:
    - Numpy NpzFile type with keys that point to values holding arrays


example usage:
data = load_npz_data("arrays.npz")
lst = data.files
for item in lst:
    print(item)
    items = data[item]
    print(items)
"""
def load_npz_data(filepath):
    data = np.load(filepath)

    return data

"""
code to turn images, test images, and both of the test masks to npz files

dir00 = "./images" ## images that are unlabelled
dir0 = "./test_images" 
dir1 = "./masks/iris"
dir2 = "./masks/pupil"
np_arrays_train_images = []
np_arrays_images = []
np_arrays_iris = []
np_arrays_pupil = []

for filename in os.listdir(dir00):
    f = os.path.join(dir00, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_train_images.append(image_to_np(f))
                         
np.savez_compressed("./train_data/unlabelled/images", images=np_arrays_train_images)

for filename in os.listdir(dir0):
    f = os.path.join(dir0, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_images.append(image_to_np(f))
                         
np.savez_compressed("./train_data/labeled/images", images=np_arrays_images)

for filename in os.listdir(dir1):
    f = os.path.join(dir1, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_iris.append(image_to_np(f))
                         
np.savez("./train_data/labeled/iris", np_arrays_iris)

print("/n/n SAVED /n/n")

for filename in os.listdir(dir2):
    f = os.path.join(dir2, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_pupil.append(image_to_np(f))
                         
np.savez_compressed("./train_data/labeled/masks", iris=np_arrays_iris, pupil=np_arrays_pupil)

idk = load_npz_data("./train_data/labeled/masks.npz")
image = Image.fromarray(idk["iris"][0])
other = load_npz_data("./train_data/labeled/images.npz")
im = Image.fromarray(other["images"][0])

image.save("./testiris.png")
im.save("./testimage.png")

"""