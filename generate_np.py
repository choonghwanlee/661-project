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
    np_arrays = {}

    for f in image_filenames:
        np_arrays[f.split("/")[-1].split(".")[0]] = image_to_np(f)

    np.savez_compressed(f"./{save_path}", **np_arrays)
    print(len(np_arrays))

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


<<<<<<< HEAD
"""
#code to turn images, test images, and both of the test masks to npz files

imgs = "./images"
tst_imgs = "./test_images"
iris_mask = "./masks/iris"
pupil_mask = "./masks/pupil"
np_arrays_images = []
np_arrays_test_images = []
np_arrays_iris = []
np_arrays_pupil = []

for filename in os.listdir(imgs):
    f = os.path.join(imgs, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_images.append(f)
images_to_npz(np_arrays_images, "./train_data/unlabelled/images")

print("/n/n SAVED IMAGES /n/n")

for filename in os.listdir(tst_imgs):
    f = os.path.join(tst_imgs, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_test_images.append(f)
images_to_npz(np_arrays_test_images, "./train_data/labeled/images")

print("/n/n SAVED TEST IMAGES /n/n")

for filename in os.listdir(iris_mask):
    f = os.path.join(iris_mask, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_iris.append(f)
images_to_npz(np_arrays_iris, "./train_data/labeled/iris")

print("/n/n SAVED IRIS /n/n")

for filename in os.listdir(pupil_mask):
    f = os.path.join(pupil_mask, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename != ".DS_Store":
        np_arrays_pupil.append(f)
images_to_npz(np_arrays_pupil, "./train_data/labeled/pupil")

print("/n/n SAVED PUPIL /n/n")


# for filename in os.listdir(dir2):
#     f = os.path.join(dir00, filename)
#     # checking if it is a file
#     if os.path.isfile(f) and filename != ".DS_Store":
#         filenames.append(f)

#     if len(filenames) > 10:
#         break

# images_to_npz(filenames, "./train_data/unlabelled/images")
                         
# np.savez_compressed("./train_data/labeled/masks", iris=np_arrays_iris, pupil=np_arrays_pupil)






# idk = load_npz_data("./train_data/labeled/masks.npz")
# image = Image.fromarray(idk["iris"][0])
# other = load_npz_data("./train_data/labeled/images.npz")
# im = Image.fromarray(other["images"][0])

# image.save("./testiris.png")
# im.save("./testimage.png")
"""