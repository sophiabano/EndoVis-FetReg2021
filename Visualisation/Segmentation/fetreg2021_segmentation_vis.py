"""
Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156

Visualization script for image and mask for task 1 (semantic segmentation)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_colormap():
    """
    Returns FetReg colormap
    """
    colormap = np.asarray(
        [
            [0, 0, 0],   # 0 - background 
            [255, 0, 0], # 1 - vessel
            [0, 0, 255], # 2 - tool
            [0, 255, 0], # 3 - fetus

        ]
        )
    return colormap

def plot_image_n_label(img, mask):
    """
    Plot of image and RGB mask for visualisation 
    Params
        img : Input image 
        mask: Input segmentation mask 
    Return
        plot of image and RGB mask   
    """
    
    colormap = get_colormap()
    
    mask_rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        mask_rgb[mask == cnt] = colormap[cnt]
    

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(img)
    axs[0].axis("off")


    axs[1].imshow(mask_rgb)
    axs[1].axis("off")
    fig.tight_layout()
    plt.show()
    
    return fig

def plot_image_gt_pred_labels(img, mask, pred):
    """
    Plot of image and RGB mask for visualisation 
    Params
        img : Input image 
        mask: Input segmentation mask 
    Return
        plot of image and RGB mask   
    """
    
    colormap = get_colormap()
    
    mask_rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        mask_rgb[mask == cnt] = colormap[cnt]
    
    pred_rgb = np.zeros(pred.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        pred_rgb[pred == cnt] = colormap[cnt]

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs[0].imshow(img)
    axs[0].axis("off")


    axs[1].imshow(mask_rgb)
    axs[1].axis("off")
    
    axs[2].imshow(pred_rgb)
    axs[2].axis("off")
    
    fig.tight_layout()
    # plt.show()
    
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input (images) folder", default = "sample_data/images/input")
    parser.add_argument("--mask", help="Path to segmentation masks folder corresponding to input", default = "sample_data/images/output")
    parser.add_argument("--output", help="Output path to save plot" , default = "sample_data/images/vis")
    args = parser.parse_args()

    assert os.path.isdir(args.input), f"{args.input} directory does not exist"
    
    img_path =args.input 
    mask_path = args.mask 
    assert os.path.exists(img_path), f"{img_path} images/labels do not exist."
    assert os.path.exists(mask_path), f"{mask_path} images/labels do not exist."
    

    Img_list = np.sort(os.listdir(img_path)) # List all image names
    

    for cnt in range(len(Img_list)):
        fname = Img_list[cnt]
        img_path_fname = os.path.join(img_path, fname)
        mask_path_fname = os.path.join(mask_path, fname)
       
        img = cv2.imread(img_path_fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path_fname, cv2.COLOR_BGR2GRAY)
        fig = plot_image_n_label(img, mask)
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        fname2 = fname.replace('png','jpg')
        fig.savefig(os.path.join(args.output, fname2))

