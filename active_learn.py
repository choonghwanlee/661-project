import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import metrics
from evaluate_performance import evaluate_on_images
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.stats import entropy
from transformers import SamConfig, SamModel, SamProcessor
from PIL import Image, ImageOps, ImageFilter
from finetune import fine_tune, get_bounding_box, postprocess_masks
import statistics
import time
import os

h = 1920
w = 1080
proj_eye_center, proj_eye_radius = (w/2, h/3), (0.25*w, 0.1*h)
y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])

def get_bounding_box(frame_size):
    # get bounding box from mask
    proj_eye_center, proj_eye_radius = (frame_size[0]/2, frame_size[1]/3), (0.25*frame_size[0], 0.07*frame_size[1])
    y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
    x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])
    bbox = [x_start, y_start, x_end, y_end]
    return bbox

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def predict_mask_w_point(image):
    model_type = "vit_t"
    sam_checkpoint = "./weights/mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    img = cv2.imread(image)
    height, width, _ = img.shape

    #this is for bounding box
    prompt = get_bounding_box([width, height])

    predictor = SamPredictor(mobile_sam)
    predictor.set_image(img)

    input_point = np.array([[(width/2), height/3]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    #for bounding box:
    # masks, scores, logits = predictor.predict(
    #     box=[[prompt]]
    # )

    print(masks.shape)

    for i, (mask, score, logits) in enumerate(zip(masks, scores, logits)):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


"""
stuff for acquisition functions
"""
def imageEncoder(img, model, preprocess, device):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2, model, preprocess, device):
    img1 = imageEncoder(image1, model, preprocess, device)
    img2 = imageEncoder(image2, model, preprocess, device)
    score = cosine_similarity(img1.cpu().detach().numpy(), img2.cpu().detach().numpy())*100
    return score[0][0]

def model_sim(image1, image2, model, preprocess, device):
    return generateScore(image1, image2, model, preprocess, device)

def ssim(image1, image2):
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return metrics.structural_similarity(image1_gray, image2_gray, full=True)[0]*100

def overall_similarity(img1, img2, model, preprocess, device, importance=[.5, .5]):
    if (importance[0] + importance[1]) != 1:
        print("weights must sum to 1")
        return
    
    h, w, _ = img1.shape
    proj_eye_center, proj_eye_radius = (w/2, h/3), (0.25*w, 0.1*h)
    y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
    x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])

    crop_img1 = img1[y_start:y_end, x_start:x_end]
    crop_img2 = img2[y_start:y_end, x_start:x_end]

    model_similarity_score = model_sim(crop_img1, crop_img2, model, preprocess, device)
    ssim_score = ssim(crop_img1, crop_img2)

    return model_similarity_score*importance[0] + ssim_score*importance[1]

def generate_preds(image_names, images, modelCheckpointFilePath):
    #make the predictions, then call compute_metrics
    model_config = SamConfig.from_pretrained("Zigeng/SlimSAM-uniform-50")
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

    # Create an instance of the model architecture with the loaded configuration
    my_model = SamModel(config=model_config)
    #Update the model by loading the weights from saved file.
    my_model.load_state_dict(torch.load(modelCheckpointFilePath))
    ## move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_model = my_model.to(device)
    ## prepare prompt and input
    print('model loaded')
    augmented = [Image.fromarray(im).filter(ImageFilter.GaussianBlur(radius = 3)) for im in images]
    print('image augmented')
    width, height = augmented[0].size
    prompt = get_bounding_box([width, height])
    segmentations = []
    scores = []
    print('generating predictions')
    count = 0;
    for image in images:
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        if (count%200) == 0:
            print(f'processed image {count}/{len(augmented)}')
        ## generate eval – forward pass
        my_model.eval()
        with torch.no_grad():
            outputs = my_model(**inputs, multimask_output=False)
            # scores.append(np.array(outputs.iou_scores.cpu())[0][0][0])
        predicted_masks = postprocess_masks(outputs.pred_masks.squeeze(1), inputs["reshaped_input_sizes"][0].tolist(), 
                                        inputs["original_sizes"][0].tolist(), processor.image_processor.pad_size).to(device)
        # convert soft mask to hard mask
        seg_prob = torch.sigmoid(predicted_masks)
        seg_prob = seg_prob.cpu().numpy().squeeze()
        crop_seg_prob = seg_prob[y_start:y_end, x_start:x_end].flatten()
        scores.append(statistics.mean([1-x for x in crop_seg_prob if x > 0.5]))
        
        # scores.append(entropy(seg_prob.flatten()))
        seg = (seg_prob > 0.5).astype(np.uint8)
        segmentations.append(seg)
        count += 1
    return segmentations, scores


def calc_circularity(mask):
    iris_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_area = -1
    circ = -1

    for contour in iris_contours:
        convex_closed = cv2.convexHull(contour, False)
        perimeter = cv2.arcLength(convex_closed, True)
        area = cv2.contourArea(convex_closed)
        if perimeter == 0:
            continue
        ## usually between 0 and 1, higher == more circular
        circularity = (4*math.pi*area)/(perimeter*perimeter)

        if (area > biggest_area):
            biggest_area = area
            circ = circularity

    return circ


"""
this should use uncertainty and similarity to choose best data points to label
compare all the images with low certainty with the images that have already been 
"""
def hybrid_selector(image_names, images_arr, masks, certainties, labeled_arr):
    length = len(images_arr)
    if len(image_names) != length or len(masks) != length or len(certainties) != length:
        print(f"ARRAYS MUST BE OF SAME LENGTH. GOT: \n image_names: {len(image_names)}, images_arr: {length}, masks: {len(masks)}, certainties: {len(certainties)}")
    
    print("getting uncertain images...")
    ind = 0
    bad_result_inds = []
    # uncertainty sampling -> if circularity is <0.7 or if the score is <0.7
    for ind in range(length):
        # calc circularity
        circularity = calc_circularity(masks[ind])
        # print(f"circularity: {circularity}")
        if circularity < 0.7 or certainties[ind] > 0.3:
            print(f"certainty: {certainties[ind]}")
            bad_result_inds.append(ind)
    print(bad_result_inds)
    
    print(f"bad results: {len(bad_result_inds)}")
    print("selecting images to label...")
    label_inds = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    model.to(device)
    for i in bad_result_inds:
        highest_sim = 0
        for labeled in labeled_arr:
            similarity = overall_similarity(images_arr[i], labeled, model, preprocess, device, importance=[0.75, 0.25])
            highest_sim = max(highest_sim, similarity)
        if highest_sim < 80.0:
            label_inds.append(i)
        print(f"similarity: {highest_sim}")
    print(f"label results: {len(label_inds)}")
    print("compiling images to label...")
    label_names = []
    label_arrs = []
    for index in label_inds:
        label_names.append(image_names[index])
        label_arrs.append(images_arr[index])

   # find the images with similarity below x (idk yet)
    # for all labelled images, compare with the current image
    print(f"images: {label_names}")
   # return images who don't meed the threshold
    return label_names, label_arrs

def test(modelCheckPointPath, mode='iris'):
    # this will be a dictionary with key imagename and value dictionary with 2 np arrays
    image_and_mask = {}
    # load test images
    print("\nIMAGES")
    test_image_data = np.load('./train_data/labeled/images.npz')
    lst = test_image_data.files
    print(len(lst))
    for item in lst:
        arr = test_image_data[item]
        image_and_mask[item] = {'image': arr}

    if mode == 'iris':
        print("\nIRISES")
        test_image_data = np.load('./train_data/labeled/iris.npz')
        lst = test_image_data.files
        print(len(lst))
        for item in lst:
            name = item.replace("_iris", "")
            image_and_mask[name]["mask"] = test_image_data[item]

    elif mode == 'pupil':
        print("\nPUPILS")
        test_image_data = np.load('./train_data/labeled/pupil.npz')
        lst = test_image_data.files
        print(len(lst))
        for item in lst:
            name = item.replace("_pupil", "")
            image_and_mask[name]["mask"] = test_image_data[item]

    images = []
    masks = []
    for im in image_and_mask:
        if "mask" in image_and_mask[im]:
            m_grayscale = cv2.cvtColor(image_and_mask[im]["mask"], cv2.COLOR_RGB2GRAY)
            images.append(np.array(image_and_mask[im]["image"]))
            masks.append(m_grayscale)
    
    print()
    print(len(images))
    print(len(masks))
    mious = evaluate_on_images(images, masks, modelCheckPointPath)
    return mious


manually_labeled_images = ['aden_lefteye_0', 'aden_lefteye_1600', 'aden_lefteye_1833', 'aden_lefteye_233', 'aden_lefteye_2533', 'aden_lefteye_2733', 'aden_lefteye_2900', 'aden_lefteye_3000', 'aden_lefteye_3200', 'aden_lefteye_3566', 'aden_lefteye_3766', 'aden_lefteye_4433', 'aden_lefteye_966', 'aden_righteye_1100', 'aden_righteye_1300', 'aden_righteye_1466', 'aden_righteye_1666', 'aden_righteye_2033', 'aden_righteye_2233', 'aden_righteye_266', 'aden_righteye_2966', 'aden_righteye_3066', 'aden_righteye_3266', 'aden_righteye_3500', 'aden_righteye_3700', 'aden_righteye_3933', 'aden_righteye_4133', 'aden_righteye_4866', 'aden_righteye_500', 'aden_righteye_700', 'aden_righteye_933', 'dillon_lefteye_1200', 'dillon_righteye_3500', 'jason_lefteye_1633', 'jason_lefteye_1800', 'jason_lefteye_4166', 'jason_lefteye_4366', 'jason_lefteye_4400', 'jason_lefteye_4600', 'jason_lefteye_4833', 'jason_righteye_1533', 'jason_righteye_1733', 'jason_righteye_4700', 'dillon_righteye_3466', 'dillon_righteye_266', 'jason_lefteye_2333', 'dillon_lefteye_1233']
t0 = time.time()
# load images
image_names = []
images = []
test_image_data = np.load('./train_data/unlabelled/images.npz')
t1 = time.time()
# identify which images to use (not used for training/finetuning/testing)
lst = test_image_data.files
print(len(lst))
for item in lst:
    if ("christian_lefteye" not in item) and ("cindy_lefteye" not in item) and (item not in manually_labeled_images):
        image_names.append(item)
        arr = test_image_data[item]
        images.append(arr)
print(len(image_names))
t2 = time.time()
# make predictions
masks, certainties = generate_preds(image_names, images, "./models/pass3_3_001_1e4_allaug_iris_model_checkpoint.pth")
t3 = time.time()
# select images to label
to_label_names, to_label_arrs = hybrid_selector(image_names, images_arr, masks, certainties, labeled_arr)
t4 = time.time()
print(len(to_label_names))
print(to_label_names.sort())
print("------------")
print(f"time to load images: {t1-t0}")
print(f"time to identify which images to use: {t2-t1}")
print(f"time to make predictions: {t3-t2}")
print(f"time to select images to label: {t4-t3}")

t5 = time.time()
# finetuning

new_images = []
new_masks = []

mask_directory = "train/masks"
img_directory = "train/images"
for filename in os.listdir(mask_directory):
    if ".png" in filename:
        f = os.path.join(mask_directory, filename)
        mask = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        new_masks.append(np.asarray(mask))

        name = filename.replace("_iris.png", ".jpg")
        img_f = os.path.join(img_directory, name)
        img = cv2.cvtColor(cv2.imread(img_f), cv2.COLOR_BGR2RGB)
        new_images.append(np.asarray(img))

mask_directory = "label-1/masks"
img_directory = "label-1/images"
for filename in os.listdir(mask_directory):
    if ".png" in filename:
        f = os.path.join(mask_directory, filename)
        mask = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        new_masks.append(np.asarray(mask))

        name = filename.replace("_0_iris.png", ".jpg")
        img_f = os.path.join(img_directory, name)
        img = cv2.cvtColor(cv2.imread(img_f), cv2.COLOR_BGR2RGB)
        new_images.append(np.asarray(img))
        
mask_directory = "label-2/masks"
img_directory = "label-2/images"
for filename in os.listdir(mask_directory):
    if ".png" in filename:
        f = os.path.join(mask_directory, filename)
        mask = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        new_masks.append(np.asarray(mask))

        name = filename.replace("_iris.png", ".jpg")
        img_f = os.path.join(img_directory, name)
        img = cv2.cvtColor(cv2.imread(img_f), cv2.COLOR_BGR2RGB)
        new_images.append(np.asarray(img))
        
mask_directory = "label-3/masks"
img_directory = "label-3/images"
for filename in os.listdir(mask_directory):
    if ".png" in filename:
        f = os.path.join(mask_directory, filename)
        mask = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        new_masks.append(np.asarray(mask))

        name = filename.replace("_iris.png", ".jpg")
        img_f = os.path.join(img_directory, name)
        img = cv2.cvtColor(cv2.imread(img_f), cv2.COLOR_BGR2RGB)
        new_images.append(np.asarray(img))
        
mask_directory = "label-4/masks"
img_directory = "label-4/images"
for filename in os.listdir(mask_directory):
    if ".png" in filename:
        f = os.path.join(mask_directory, filename)
        mask = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        new_masks.append(np.asarray(mask))

        name = filename.replace("_iris.png", ".jpg")
        img_f = os.path.join(img_directory, name)
        img = cv2.cvtColor(cv2.imread(img_f), cv2.COLOR_BGR2RGB)
        new_images.append(np.asarray(img))

print(len(new_images))
fine_tune(new_images, new_masks, checkpoint_info='00pass4_8_001_1e4_again_pretrained_baseaug') #, modelCheckpointFilePath=)
t6 = time.time()
print(f"time to fine_tune: {t6-t5}")

t7 = time.time()
# evaluating
new_mious = test("./models/00pass4_8_001_1e4_again_pretrained_baseaug_iris_model_checkpoint.pth")
# print(new_mious)
print(len(new_mious))
new_mious_avg = np.average(np.array(new_mious))
print(new_mious_avg)
t8 = time.time()
print(f"time to eval: {t8-t7}")



"""
PRETRAINED MODEL ON IRIS DETECTION

print(test())
    #this is the mious from running test on the pretrained model for iris (line above)
mious_base_iris = [0.4933260235662088, 0.49456124450219374, 0.4832681071169556, 0.4826067726532986, 0.49302743459249254, 0.49450037768372873, 0.494620808700202, 0.492379924346995, 0.49516633007828736, 0.49300185309489236, 0.49472957708063786, 0.4922172865305233, 0.4946474964774582, 0.49441447018634055, 0.4948954217970404, 0.4950794741333151, 0.4822401608121129, 0.492858110351498, 0.4922795151497368, 0.49464505428271005, 0.49355210889122564, 0.4928948263090596, 0.49470427065485983, 0.49403921603120315, 0.4945408063064691, 0.49509352638406495, 0.4906434092262216, 0.4941175382317415, 0.4939127746784668, 0.48426539565776666, 0.48419869746812483, 0.4951142681087964, 0.49433143974100546, 0.494359942983511, 0.4834893416072796, 0.483552298708695, 0.494387451897316, 0.4930539396706743, 0.4926704341315179, 0.4932972409897518, 0.49499850165258186, 0.4944168993362971, 0.4927518770199597, 0.49470729798445184, 0.4932050229817577, 0.49312846350068906, 0.4919527247003279, 0.4935779450300329, 0.49418717208121726, 0.4931130723832085, 0.49312041046310967, 0.4903235957797158, 0.4951823043979897, 0.48311090325232214, 0.48286711833409535, 0.48319512122413133, 0.493836611419052, 0.49407810927861046, 0.4953289618486842, 0.49524469745929656, 0.4942750031521815, 0.49280615132261424, 0.49413997515083674, 0.4942989263132682, 0.4955325296091538, 0.49306895989367944, 0.49280906847039624, 0.4926877812391831, 0.49533775484658354, 0.48357365824629667, 0.48270110693172563, 0.4945457936170959, 0.4935323809417157, 0.49522876802989896, 0.4911856970024946, 0.4945145731833889, 0.4941291204192151, 0.48984776532628904, 0.49581603868135304, 0.4830046784008855, 0.483453333066106, 0.4928671094790584, 0.4932935307107222, 0.49472893848456617, 0.4817894541535648, 0.4950706019067379, 0.4926119540563113, 0.4946688727964161, 0.49462361651737563, 0.49319024675135054, 0.494780036603536, 0.4954263201619895, 0.4912043847542669, 0.49333917010120526, 0.4919851700509609, 0.4931293099966374, 0.4937517924187928, 0.49466678724095875, 0.49422448291531684, 0.4945575964501008, 0.49525298665647793, 0.49523442139387275, 0.4944893265700939, 0.4943544726912895, 0.4834379896215508, 0.48289840552263424, 0.48391810721913503, 0.4934996428990463, 0.49467906075972917, 0.4932888977831403, 0.4833252842908761, 0.4826346752267375, 0.4946097737165558, 0.49468416813609156, 0.495349433812369, 0.4944566571872778, 0.4927815389647941, 0.4932297444568792, 0.4919912189969077, 0.492935005359349, 0.49474900195113175, 0.4950513393599152, 0.49150337409900946, 0.4946063150134946, 0.4802040316473977, 0.4934003919681594, 0.49331426810405643, 0.49308806141391476, 0.49389297181764846, 0.4944983893077785, 0.49432862727117644, 0.49427362438277894, 0.4945264451743169, 0.4951671896412109, 0.4909480625101295, 0.4953897408284104, 0.49435710052269993, 0.4840667685715849, 0.4933793194093207, 0.4931599691922502, 0.4826457051016338, 0.4930587800117784, 0.49465564665600437, 0.4945496918307671, 0.49459921978956894, 0.4938132979779691, 0.4950660175946343, 0.4955178347934919, 0.4930217528204499, 0.4928923363658262, 0.4947972094701249, 0.4927087360750154, 0.49239692383299555, 0.4930989506626945, 0.49471073297341484, 0.49447934690351364, 0.4943747235331511, 0.4949238442154905, 0.49377922946671043, 0.49267433153153506, 0.49464018392441533, 0.4938286144313373, 0.4951389075554274, 0.49323239721971135, 0.4931995198687187, 0.49543205866064455, 0.48329021446017323, 0.4827920378568814, 0.4944421638820269, 0.4952143460141446, 0.4906914356438294, 0.49491724393417047, 0.4948782806239652, 0.4951247408330332, 0.49420283222343075, 0.48429307733646, 0.4945774160442248, 0.4834710440791712, 0.4938788747714416, 0.4927337850033734, 0.49485735434752426, 0.4952234649576487, 0.49522554187214807, 0.49298695975175527, 0.492604687273495, 0.4947161359012388, 0.49438924117398847, 0.49311025107586903, 0.49300593312125773, 0.4932272634775607, 0.4928158954115532, 0.49481190972762035, 0.494698335715794, 0.4945824721140188, 0.4937807543103659, 0.4945665173616597, 0.49513089330358473, 0.48329406002876946, 0.493574378937793, 0.4926518769503391, 0.49482379996049897, 0.4867420498907514, 0.4927440768123012, 0.49274431944918934, 0.49458227289006196, 0.4947656751091157, 0.493292426744311, 0.49310593705575256, 0.49299590310338215, 0.4944940217316943, 0.49460155560822916, 0.4933872784296387, 0.49275526728097585, 0.4942075393869727, 0.48469063057039913, 0.4917165048524703, 0.49326201391269126, 0.49283422865928894, 0.4928157025400624, 0.496866532158735, 0.48307913043053174, 0.48284203434084927, 0.4938881660701587, 0.4951724741982103, 0.49539596188250756, 0.491219019074657, 0.4942890456553957, 0.4943453507655709, 0.4945636795152216, 0.4930782417282724, 0.4833377710574079, 0.49414182728478995, 0.4942900129098335, 0.491769131700707, 0.4952529882058403, 0.4927173255474964, 0.492708639015704, 0.4946958342010655, 0.4935541855629987, 0.4924498167079254, 0.4927661472619982, 0.49457522284349226, 0.4946556507498075, 0.4950605407166092, 0.495099837688544, 0.4950966539515718, 0.49477311454408995, 0.4935255058641074, 0.49305207692880626, 0.4936573637439691, 0.4932210916880608, 0.4943739969269628, 0.4950195501604893, 0.49397412790569784, 0.49436559181799533, 0.4943386973239965, 0.49028580076329675, 0.495289502040721, 0.4943060648965943, 0.48322791058665293]
print(len(mious_base_iris))
np_mious_iris = np.array(mious_base_iris)
print(np.average(np_mious_iris))
    # pretrained miou average is 0.49249325252684534

BASELINE MODEL ON IRIS DETECTION
print(test())
    # pretrained miou average is 0.49442484884953003
"""