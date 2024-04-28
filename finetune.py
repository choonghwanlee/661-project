import numpy as np
from datasets import Dataset
import cv2
from PIL import Image, ImageOps, ImageFilter
from transformers import SamProcessor, SamConfig, SamModel
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from typing import Tuple
import monai
import time
from tqdm import tqdm


class SAMDataset(TorchDataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor, transform = None, task = 'iris'):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        self.type = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ## apply random data augmentation
        if self.transform:
            image = self.transform(image)
        width, height = image.size
        ground_truth_mask = np.array(item["label"])

        # get iris box prompt or 
        prompt = get_bounding_box([width, height]) if self.type == 'iris' else get_point_prompt(ground_truth_mask)

        # prepare image and prompt for the model
        if self.type == 'iris':
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        else: 
            inputs = self.processor(image, input_points = [[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs

def fine_tune(images, pred_masks, mode='iris', checkpoint_info='base', modelCheckpointFilePath = None, BATCH_SIZE=2, num_epochs=4):
    ## images: np array of images 
    ## pred_masks: np array of masks
    print('running fine-tuning')
    data_dic = create_dataset(images, pred_masks)
    # Initialize the processor
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    augmentations = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(brightness=0.2)])
    dataset = SAMDataset(dataset=data_dic, processor=processor, transform= augmentations, task=mode)
    print('dataset initialized!')
    # Create a DataLoader instance for the training dataset
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    if not modelCheckpointFilePath:
        model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
    else:
        model_config = SamConfig.from_pretrained("Zigeng/SlimSAM-uniform-50")
        # Create an instance of the model architecture with the loaded configuration
        model = SamModel(config=model_config)
        #Update the model by loading the weights from saved file.
        model.load_state_dict(torch.load(modelCheckpointFilePath))
    print('loaded model!')
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=0.001, weight_decay=1e-4)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.FocalLoss(gamma=2.0, alpha=0.5)
    #Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    print('starting fine tuning!')
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'EPOCH: {epoch}')
        epoch_losses = []
        for batch in tqdm(train_dataloader):
        # forward pass
            if mode == 'iris':
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                input_boxes=batch["input_boxes"].to(device),
                                multimask_output=False)
            elif mode == 'pupil':
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                input_points=batch["input_points"].to(device),
                                multimask_output=False)
            # postprocess mask to original scale
            predicted_masks = postprocess_masks(outputs.pred_masks.squeeze(1), batch["reshaped_input_sizes"][0].tolist(), 
                                                batch["original_sizes"][0].tolist(), processor.image_processor.pad_size).to(device)
            # compute loss
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())
        
        print(f'Mean loss: {sum(epoch_losses)/len(epoch_losses)}')             
    # Save the model's state dictionary to a file
    print('done!')
    end_time = time.time() - start_time
    torch.save(model.state_dict(), f"./models/{checkpoint_info}_{mode}_model_checkpoint.pth")
    return end_time

def create_dataset(images, masks):
    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    
    dataset_dict = {
        "image": [Image.fromarray(img).filter(ImageFilter.MedianFilter(size = 3)) for img in images],
        "label": [ImageOps.grayscale(Image.fromarray(mask)) for mask in masks],
    }
    # Create the dataset using the datasets.Dataset class
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

#Get bounding boxes for iris prompt
def get_bounding_box(frame_size):
    # get bounding box from mask
    proj_eye_center, proj_eye_radius = (frame_size[0]/2, frame_size[1]/3), (0.25*frame_size[0], 0.07*frame_size[1])
    y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
    x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])
    bbox = [x_start, y_start, x_end, y_end]
    return bbox


def get_point_prompt(mask):
    ## find pupil center from mask
    rows, cols = np.nonzero(mask)
    center_row = np.mean(rows)
    center_col = np.mean(cols)
    return [center_row, center_col]


def postprocess_masks(masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...], image_size = Tuple[int, ...]) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.

    Args:
      masks (torch.Tensor):
        Batched masks from the mask_decoder, in BxCxHxW format.
      input_size (tuple(int, int)):
        The size of the image input to the model, in (H, W) format. Used to remove padding.
      original_size (tuple(int, int)):
        The original size of the image before resizing for input to the model, in (H, W) format.

    Returns:
      (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        (image_size['height'], image_size['width']),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

def _are_points_in_bounding_box(contour, bounding_box):
    x, y, width, height = bounding_box
    for point in contour:
        px, py = point[0]  # point is in the form [[x, y]]
        
        if not (x <= px <= x + width and y <= py <= y + height):
            return False
    
    return True

def generate_eval(images, modelCheckpointFilePath):
    #make the predictions, then call compute_metrics
    model_config = SamConfig.from_pretrained("Zigeng/SlimSAM-uniform-50")
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")


def reduce_masks(sam_predictions):
    frame_size = (1080, 1920)
    proj_eye_center = (frame_size[0]/2, frame_size[1]/3)
    proj_eye_radius = (0.25*frame_size[0], 0.07*frame_size[1])
    y_start, y_end = int(proj_eye_center[1]-proj_eye_radius[1]), int(proj_eye_center[1]+proj_eye_radius[1])
    x_start, x_end = int(proj_eye_center[0]-proj_eye_radius[0]), int(proj_eye_center[0]+proj_eye_radius[0])
    rect = (x_start, y_start, x_end - x_start, y_end - y_start)
    sam_pred_masks = [] ## your updated masks
    
    for pred_mask in sam_predictions: ## where sam_predictions are your model's output
        max_area = -1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        eroded_mask = cv2.erode(pred_mask, kernel, iterations=2) ## remove noise from model prediction
        eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel) ## remove noise from model prediction
        iris_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = -1
        iris_center, iris_radius = None, None 
        for contour in iris_contours:
            convex_closed = cv2.convexHull(contour, False)
            perimeter = cv2.arcLength(convex_closed, True)
            area = cv2.contourArea(convex_closed)
            if perimeter == 0:
                continue
            circularity = (4*math.pi*area)/(perimeter*perimeter)
            ## check the contour circularity is high, area is big, and all points are inside bounding_box
            if circularity > 0.8 and area > max_area and _are_points_in_bounding_box(convex_closed, rect):
                max_area = area
                approx = cv2.approxPolyDP(convex_closed, perimeter * 0.034, True)
                iris_center, iris_radius = cv2.minEnclosingCircle(approx)
        filtered_mask = np.zeros((1920, 1080), dtype=np.uint8)
        if iris_center:
            cv2.circle(filtered_mask, (int(iris_center[0]),int(iris_center[1])), int(iris_radius), (255), -1)  
            sam_pred_masks.append(filtered_mask)
        else:
            sam_pred_masks.append(eroded_mask)
    return sam_pred_masks

def generate_eval(images, modelCheckpointFilePath, mode='iris',model=None):
    my_model=model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (my_model is None):
        model_config = SamConfig.from_pretrained("Zigeng/SlimSAM-uniform-50")

        # Create an instance of the model architecture with the loaded configuration
        my_model = SamModel(config=model_config)
        #Update the model by loading the weights from saved file.
        my_model.load_state_dict(torch.load(modelCheckpointFilePath))
        
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    ## move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_model = my_model.to(device)
    ## prepare prompt and input
    augmented = [Image.fromarray(im).filter(ImageFilter.MedianFilter(size= 3)) for im in images]
    width, height = augmented[0].size
    prompt = get_bounding_box([width, height])
    segmentations = []
    for image in augmented:
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        ## generate eval – forward pass
        my_model.eval()
        with torch.no_grad():
            outputs = my_model(**inputs, multimask_output=False)
        predicted_masks = postprocess_masks(outputs.pred_masks.squeeze(1), inputs["reshaped_input_sizes"][0].tolist(), 
                                        inputs["original_sizes"][0].tolist(), processor.image_processor.pad_size).to(device)
        
        # convert soft mask to hard mask
        seg_prob = torch.sigmoid(predicted_masks)
        seg_prob = seg_prob.cpu().numpy().squeeze()
        seg = (seg_prob > 0.6).astype(np.uint8)
        segmentations.append(seg)
    segmentations = reduce_masks(segmentations)
    return segmentations

def generate_eval_segprob(images, modelCheckpointFilePath, mode='iris',model=None):
    my_model=model
    if (my_model is None):
        model_config = SamConfig.from_pretrained("Zigeng/SlimSAM-uniform-50")

        # Create an instance of the model architecture with the loaded configuration
        my_model = SamModel(config=model_config)
        #Update the model by loading the weights from saved file.
        my_model.load_state_dict(torch.load(modelCheckpointFilePath))
        
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    ## move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_model = my_model.to(device)
    ## prepare prompt and input
    augmented = [Image.fromarray(im).filter(ImageFilter.MedianFilter(size = 3)) for im in images]
    width, height = augmented[0].size
    prompt = get_bounding_box([width, height])
    seg_probs = []
    for image in augmented:
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        ## generate eval – forward pass
        my_model.eval()
        with torch.no_grad():
            outputs = my_model(**inputs, multimask_output=False)
        predicted_masks = postprocess_masks(outputs.pred_masks.squeeze(1), inputs["reshaped_input_sizes"][0].tolist(), 
                                        inputs["original_sizes"][0].tolist(), processor.image_processor.pad_size).to(device)
        
        # convert soft mask to hard mask
        seg_prob = torch.sigmoid(predicted_masks)
        seg_prob = seg_prob.cpu().numpy().squeeze()
        seg = (seg_prob > 0.6).astype(np.uint8)
        seg_probs.append(seg_prob)
    return seg_probs

"""
find fine_tune usage in naive_train.py
"""
