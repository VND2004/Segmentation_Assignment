import os
import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp
import argparse

# Define transformations for input image
val_transformation = Compose([
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Function to convert mask indices to RGB colors
def mask_to_rgb(mask, color_dict):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_dict.items():
        mask_rgb[mask == label] = color
    return mask_rgb

# Main inference function
def main(image_path, output_path):
    # Load the model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights=None,          
        in_channels=3,                 
        classes=3                      
    )
    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Read and preprocess the input image
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]
    resized_img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=resized_img)
    input_img = transformed['image'].unsqueeze(0)  

    # Perform inference
    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mask = cv2.resize(output_mask, (ori_w, ori_h))  # Resize mask to original dimensions
    mask = np.argmax(mask, axis=2)

    # Define the color mapping for the segmentation mask
    color_dict = {
        0: [0, 0, 0],    
        1: [255, 0, 0],  
        2: [0, 255, 0],  
    }
    mask_rgb = mask_to_rgb(mask, color_dict)

    # Save the segmented image
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, mask_bgr)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="segmented_output.png", help="Path to save the output image")
    args = parser.parse_args()

    main(args.image_path, args.output_path)
