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

def fine_tune(images, pred_masks, mode='iris', checkpoint_info='base', modelCheckpointFilePath = None):
    ## images: np array of images 
    ## pred_masks: np array of masks
    print('running fine-tuning!!')
    data_dic = create_dataset(images, pred_masks)
    # Initialize the processor
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    # augmentations = transforms.Compose([transforms.RandomCrop((1920, 1080)), transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(brightness=0.2), transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2))])
    augmentations = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(brightness=0.2)])
    # dataset = SAMDataset(dataset=data_dic, processor=processor, task=mode)

    dataset = SAMDataset(dataset=data_dic, processor=processor, transform= augmentations, task=mode)
    print('dataset initialized!')
    # Create a DataLoader instance for the training dataset
    BATCH_SIZE = 2
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    if not modelCheckpointFilePath:
        model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
    else:
        model_config = SamConfig.from_pretrained("Zigeng/SlimSAM-uniform-50")
        # Create an instance of the model architecture with the loaded configuration
        model = SamModel(config=model_config)
        #Update the model by loading the weights from saved file.
        print(f"loading model from {modelCheckpointFilePath}!")
        model.load_state_dict(torch.load(modelCheckpointFilePath))
    print('loaded model!')
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    # Initialize the optimizer and the loss function
    lr = 0.001
    optimizer = Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=1e-4)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.FocalLoss(gamma=2.0, alpha=0.5)
    #Training loop
    num_epochs = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    print('starting fine tuning!')
    start_time = time.time()
    best_loss = 1
    for epoch in range(num_epochs):
        print(f'EPOCH: {epoch}')
        if epoch % 4 == 0 and epoch != 0:
            lr *= 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"decreased lr to {lr}")
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
        
        mean_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'Mean loss: {mean_loss}')
        if mean_loss < best_loss:
            print("saving model!")
            best_loss = mean_loss
            torch.save(model.state_dict(), f"./models/{checkpoint_info}_{mode}_model_checkpoint.pth")
    # Save the model's state dictionary to a file
    print('done!')
    end_time = time.time() - start_time
    if best_loss == 1:
        print("saving last one")
        torch.save(model.state_dict(), f"./models/{checkpoint_info}_{mode}_model_checkpoint.pth")
    return end_time

def create_dataset(images, masks):
    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    
    dataset_dict = {
        "image": [Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius = 3)) for img in images],
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


def generate_eval(images, modelCheckpointFilePath):
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
    augmented = [Image.fromarray(im).filter(ImageFilter.GaussianBlur(radius = 3)) for im in images]
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
        seg = (seg_prob > 0.5).astype(np.uint8)
        segmentations.append(seg)
    return segmentations