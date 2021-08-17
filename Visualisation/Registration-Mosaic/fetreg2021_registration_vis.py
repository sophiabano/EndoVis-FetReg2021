"""
@author: Sophia Bano

This is the implmentation of the paper 'Deep placental Vessel Segmentation for 
Fetoscopic Mosaicing' which was presented at MICCAI2020.

Note: If you use this code, consider citing the following paper: 
    
Bano, S., Vasconcelos, F., Shepherd, L.M., Vander Poorten, E., Vercauteren, T., 
Ourselin, S., David, A.L., Deprest, J. and Stoyanov, D., 2020, October. 
Deep placental vessel segmentation for fetoscopic mosaicking. In International 
Conference on Medical Image Computing and Computer-Assisted Intervention 
(pp. 763-773). Springer, Cham.

Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156


Task 2: Registration and Mosaic - visualisation of the generated mosaic

"""
import cv2
import numpy as np
import os
import argparse


from utilsReg import *

parser = argparse.ArgumentParser(description='create video files from frames')
parser.add_argument("--video_name", type=str, default = "anon001", help="video sequence name")
parser.add_argument("--videoframes_path", type=str, default = "sample_data/images/input", help="path to folder containing video frames (in folder 'images') and fetoscope masks (in folder 'mask')")
parser.add_argument("--registraion_Hpath", type=str, default = 'sample_data/images/output', help="path of folder containing individual text files with Homography martix (output of the registration algorithm)")
parser.add_argument("--FOVmask_path", type=str, default = "sample_data/images/mask/anon001_mask.png", help="path to folder containing fetoscope circular mask")
parser.add_argument("--write_path", type=str, default = "sample_data/images/vis", help="path to write output mosaic frames and video")


args = parser.parse_args()

seq_name = args.video_name
fullImgDirPath = args.videoframes_path # frames_path + '/images'
mask_path = args.FOVmask_path # frames_path + '/mask/' + seq_name + '_mask.png'
Hpath = args.registraion_Hpath
writepath = args.write_path


transformation = "Affine"
padding_size = 2000
showImages = False

window_size = 1
frame_distance = 5
 

fullImgPaths =  [ fullImgDirPath + '/' + f  for f  in sorted(os.listdir(fullImgDirPath))]

v_crop_top = seq_exact[seq_name]["v_crop_top"]
v_crop_bottom = seq_exact[seq_name]["v_crop_bottom"]

mask_im = get_mask_im(fullImgPaths, mask_path, v_crop_top, v_crop_bottom)


seq_length = seq_exact[seq_name]["file_length"] 
seq_start = seq_exact[seq_name]["start"] 

# Read H from MICCAI2020 registration algorithm output
H_array = readHfromTXT(Hpath)
H_array = H_array[seq_start:seq_start+seq_length,:,:]

## Get registration
try:
    os.stat(writepath)
except:
    os.makedirs(writepath)

# Get affine matrix
H_affine = np.zeros( (len(H_array), 2,3))
for i in range(len(H_array)):
  H = H_array[i,:,:]
  H_affine[i] =  H[:2, :]
  
  
# Aligning to the middle frame
middle_num = seq_length/2 #len(fullImgPaths)//2
middle_num = middle_num - (middle_num%3)
H_global = getHGlobal(H_affine, fullImgPaths, middle_num)

# Plotting glocal registration onto images
do_global_registration(fullImgPaths, middle_num, seq_length, seq_start, padding_size, mask_im, H_global, writepath)

# Calling the generate_video function 
generate_video(writepath, seq_name + ".MP4", writepath + "/mosaic")




