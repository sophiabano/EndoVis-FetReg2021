"""
Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156

Task 2 - Registration - Docker dummy example showing 
the input and output folders and the output text file format for the submission

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
        
    input_file_list = sorted(glob(INPUT_PATH + "/*.png"))
    
    crop_y = 80
    crop_x = 80
    
    for i in range(len(input_file_list)-1):
    
        img1 =cv2.imread(input_file_list[i])
        img2 =cv2.imread(input_file_list[i+1])
        crop_lim = [crop_y,img1.shape[0]-crop_y,crop_x,img1.shape[1]-crop_x]
        img1 = img1[crop_lim[0]:crop_lim[1],crop_lim[2]:crop_lim[3], ]
        img2 = img2[crop_lim[0]:crop_lim[1],crop_lim[2]:crop_lim[3]]
    
    
        ShiftReg    = cv2.reg_MapperGradShift()
        PyrShiftReg = cv2.reg_MapperPyramid(ShiftReg)
        ShiftMap    = PyrShiftReg.calculate(img1, img2)
        ShiftMap    = cv2.reg.MapTypeCaster_toShift(ShiftMap)
    
        AffineMap   = cv2.reg_MapAffine(np.eye(2),ShiftMap.getShift())
    
        AffineReg    = cv2.reg_MapperGradAffine()
        PyrAffineReg = cv2.reg_MapperPyramid(AffineReg)
        AffineMap    = PyrAffineReg.calculate(img1, img2,AffineMap)
        AffineMap    = cv2.reg.MapTypeCaster_toAffine(AffineMap)
    
        H = np.eye(3)
        H[0:2,2:3] = ShiftMap.getShift()
        ProjMap = cv2.reg_MapProjec(H)
    
        ProjReg    = cv2.reg_MapperGradProj()
        PyrProjReg = cv2.reg_MapperPyramid(ProjReg)
        ProjMap    = PyrProjReg.calculate(img1, img2,ProjMap)
        ProjMap    = cv2.reg.MapTypeCaster_toProjec(ProjMap)
    
        file_name = input_file_list[i+1].split("/")[-1]
        file_name = file_name.replace('png','txt')
        result = np.savetxt(OUTPUT_PATH + "/" + file_name , ProjMap.getProjTr(),'%10.5f')
        
