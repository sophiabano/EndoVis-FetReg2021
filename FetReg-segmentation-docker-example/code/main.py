"""
Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156

Task 1 - Segmentation - Docker dummy example showing 
the input and output folders for the submission
"""

import sys  # For reading command line arguments
from glob import glob  # For listing files
import cv2  # For handling image loading & processing
import os  # for path creation
import numpy as np

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH +' created')
    else:
        print(OUTPUT_PATH +' exists')
        
    input_file_list = glob(INPUT_PATH + "/*.png")

    for f in input_file_list:
        file_name = f.split("/")[-1]
        img = cv2.imread(f,0)       
        
        ret,img2 = cv2.threshold(img,100,1,cv2.THRESH_BINARY)
        
        img2 = np.uint8(img2)
        result = cv2.imwrite(OUTPUT_PATH + "/" + file_name, img2)
        if result==True:
            print(OUTPUT_PATH+'/' +file_name +' output mask saved')
        else:
            print('Error in saving file')
