import json
import os
from PIL import Image
import numpy as np
import matplotlib.path as mpltPath

"""
Returns a mask of the shape created by a list of points as an array of 0's and 255's

Parameters:
    - poly_points ([(x, y)]): a list of tuples containing coordinate pairs that define the shape
    - w (int): width of the image
    - h (int): height of the image

Returns:
    - Nparray of size (h, w) containing 0 (not in the object) and 255 (in the object)
"""
def points_inside_polygon(poly_points, w, h):
    # Create a matplotlib path object using the polygon points
    path = mpltPath.Path(poly_points)
    
    # Initialize an empty mask
    mask = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            mask[i][j] = 255 if path.contains_point((j, i)) else 0
    
    return mask

def points_inside_circle(center, radius, w, h):
    # Create a matplotlib path object using the polygon points
    path = mpltPath.Path([(0, 0)])
    circle = path.circle(center, radius)
    
    # Initialize an empty mask
    mask = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            mask[i][j] = 255 if circle.contains_point((j, i)) else 0
    
    return mask

def points_inside_circle(center, radius, w, h):
    # Create a matplotlib path object using the polygon points
    path = mpltPath.Path([(0, 0)])
    circle = path.circle(center, radius)
    
    # Initialize an empty mask
    mask = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            mask[i][j] = 255 if circle.contains_point((j, i)) else 0
    
    return mask


"""
Creates nparray representations of masks of the desired objects in the given json file

Parameters:
    - filepath (String): absolute filepath to the json file
    - mask_objects ([String]): list of strings that are the names of the desired objects. 
        Must exactly match its name in the corresponding json file

Returns:
    - dictionary whos keys are the items in mask_objects and values a list of masks for all 
        appearances of the object in the json file
"""
def get_masks(filepath, mask_objects):
    f = open(filepath)
    data = json.load(f)
    width = data["item"]["slots"][0]["width"]
    height = data["item"]["slots"][0]["height"]
    annotations = data["annotations"]
    recording_type = data["item"]["slots"][0]["type"]
    masks = {}

    if recording_type == 'video': ## images annotated through video annotation tool
        for item in annotations: 
            name = item['name']
            frame_range = item['ranges'][0]
            frames = item['frames']
            for id, value in frames.items(): ## loop over frame key
                ellipse = value['ellipse']
                c = (ellipse["center"]["x"], ellipse["center"]["y"])
                r = (ellipse["radius"]["x"]+ ellipse["radius"]["y"]) / 2
                print(f"radius: {r}")
                if name not in masks:
                    masks[name] = []
                masks[name].append(points_inside_circle(c, r, width, height))
    else: ## images annotated frame by frame
        for item in annotations:
            if item["name"] in mask_objects:
                name = item["name"]
                if "polygon" in item:
                    path = item["polygon"]["paths"][0]
                    arr = []
                    for pt in path:
                        arr.append((pt["x"], pt["y"]))

                    nparr = np.array(arr)

                    if name not in masks:
                        masks[name] = []
                    masks[name].append(points_inside_polygon(nparr, width, height))

                elif "ellipse" in item:
                    print("ellipse!!")
                    ellipse = item["ellipse"]
                    c = (ellipse["center"]["x"], ellipse["center"]["y"])
                    r = (ellipse["radius"]["x"]+ ellipse["radius"]["y"]) / 2
                    print(f"radius: {r}")

                    if name not in masks:
                        masks[name] = []
                    masks[name].append(points_inside_circle(c, r, width, height))
    f.close()
    return masks

"""
Makes an image from the given nparray mask and saves it to image_path

Parameters: 
    - mask_array: np array of the mask
    - image_path (string): path for the image to be saved at, must include file extension
""" 
def make_image_mask(mask_array, image_path):
    im = Image.fromarray(mask_array)

    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(image_path)


# masks = get_masks("./christian_righteye_66.json", ["iris", "pupil"])
# make_image_mask(masks["iris"][0], "iris_mask.jpg")
# make_image_mask(masks["pupil"][0], "pupil_mask.png")

"""
example usage 

directory = "json_files"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        masks = get_masks(f"./{f}", ["iris", "pupil"])
        name = filename.split(".")[0]
        if "iris" in masks:
            make_image_mask(masks["iris"][0], f"./masks/iris/{name}_iris.png")
        if "pupil" in masks:
            make_image_mask(masks["pupil"][0], f"./masks/pupil/{name}_pupil.png")
"""

directory = "json_files"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        masks = get_masks(f"./{f}", ["iris", "pupil"])
        name = filename.split(".")[0]
        if "iris" in masks:
            [make_image_mask(mask, f"./masks/iris/{name}_{index}_iris.png") for index, mask in enumerate(masks['iris'])]
        if "pupil" in masks:
            [make_image_mask(mask, f"./masks/pupil/{name}_{index}_pupil.png") for index, mask in enumerate(masks['pupil'])]
