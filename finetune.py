import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
import cv2
from PIL import Image
from transformers import SamProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize


class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, task = 'iris'):
    self.dataset = dataset
    self.processor = processor
    self.type = task

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    width, height = image.size
    ground_truth_mask = np.array(item["label"])

    # get iris box prompt or 
    prompt = get_bounding_box([width, height]) if self.type == 'iris' else get_point_prompt(ground_truth_mask)

    # prepare image and prompt for the model
    if self.type == 'iris':
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
    else: 
       inputs = self.processor(image, input_labels = [[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

def fine_tune(images, pred_masks, mode='iris'):
    ## images: np array of images 
    ## pred_masks: np array of masks
    data_dic = create_dataset(images, pred_masks)
    # Initialize the processor
    processor = SamProcessor.from_pretrained("dhkim2810/MobileSAM")
    dataset = SAMDataset(dataset=data_dic, processor=processor, task=mode)
    # Create a DataLoader instance for the training dataset
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=False)
    model = SamModel.from_pretrained("dhkim2810/MobileSAM")
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
            return 
    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
   
    #Training loop
    num_epochs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
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
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')             
    # Save the model's state dictionary to a file
    torch.save(model.state_dict(), "/content/drive/MyDrive/ColabNotebooks/models/SAM/{mode}_model_checkpoint.pth")

def create_dataset(images, masks):
    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    # TO-DO: add a bit of gaussian blur to remove noise and non-robust feature
    # TO-DO: filter out for potential "empty" masks
    dataset_dict = {
        "image": [Image.fromarray(img) for img in images],
        "label": [Image.fromarray(mask) for mask in masks],
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
    ## find iris center from mask
    rows, cols = np.nonzero(mask)
    center_row = np.mean(rows)
    center_col = np.mean(cols)
    return [center_row, center_col]
