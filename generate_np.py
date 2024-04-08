import numpy as np
import cv2
from PIL import Image

def image_to_np(filename):

    #using opencv
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    nparray_cv2 = np.asarray(img)
    
    #using pillow
    # image = Image.open(filename)
    # nparray_pil = np.asarray(image)

    return nparray_cv2


def images_to_np(image_filenames, save_path):
    np_arrays = []

    for f in image_filenames:
        np_arrays.append(image_to_np(f"./images/{f}.jpg"))
    
    #option 1: dictionary with 1 key whose value is an array of N np arrays
    np.savez("./{save_path}", np_arrays)

    #option 2: dictionary with N keys whose value is an np array
    # np.savez("./train_data/unlabelled/test_savez", *np_arrays[:])

    return save_path+".npz"


"""
access each value like this:
for item in lst:
    print(item)
    items = data[item]
    print(items)
"""
def load_npz_data(filename):
    data = np.load(f'./train_data/unlabelled/{filename}.npz')
    lst = data.files

    return lst


# test_arrays = []
# for i in range(3):
#     arr = np.array([[i, i, i], [i, i, i], [i, i, i]])
#     test_arrays.append(arr)