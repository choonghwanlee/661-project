from typing import Tuple
import numpy as np
from PIL import Image ImageFilter
from transformers import SamProcessor, SamConfig, SamModel
import torch
import torch.nn.functional as F
from typing import Tuple
import matplotlib.pyplot as plt

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

def predict_and_show_mask(images, modelCheckpointFilePath, imTitle):
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
    print('generating predictions')
    count = 1;
    for image in augmented:
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        # print(f'processed image {count}/{len(augmented)}')
        ## generate eval – forward pass
        my_model.eval()
        with torch.no_grad():
            outputs = my_model(**inputs, multimask_output=False)
            # print(f"outputs: {outputs}")
            # print(f"model iou: {np.array(outputs.iou_scores.cpu())[0][0][0]}")
        predicted_masks = postprocess_masks(outputs.pred_masks.squeeze(1), inputs["reshaped_input_sizes"][0].tolist(), 
                                        inputs["original_sizes"][0].tolist(), processor.image_processor.pad_size).to(device)
        # print("got predicted masks")
        # convert soft mask to hard mask
        seg_prob = torch.sigmoid(predicted_masks)
        seg_prob = seg_prob.cpu().numpy().squeeze()
        seg = (seg_prob > 0.5).astype(np.uint8)
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(seg, plt.gca())
        plt.title(imTitle)
        plt.axis('off')
        plt.show()
            
        segmentations.append(seg)
        count += 1
    return segmentations