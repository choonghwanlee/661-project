import numpy as np
import cv2
from PIL import Image


"""
converts an image to a numpy array

Parameters:
    - filepath (String): path to the image, either absolute or relative to current user directory

Returns:
    - numpy array
"""
def image_to_np(filepath):
    #using opencv
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    nparray_cv2 = np.asarray(img)
    
    #using pillow
    # image = Image.open(filename)
    # nparray_pil = np.asarray(image)

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
    # print(type(data))
    # print(data)
    # lst = data.files

    return data

# test_arrays = []
# for i in range(3):
#     arr = np.array([[i, i, i], [i, i, i], [i, i, i]])
#     test_arrays.append(arr)