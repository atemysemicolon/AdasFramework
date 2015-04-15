# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:24:11 2015

@author: prassanna
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:52:44 2015

@author: prassanna
"""


import glob
import cv2
import numpy as np
from prettytable import PrettyTable
import cPickle
import opencv_utils as ocv
####START EDITING HERE

######DEFINE COLOURS HERE########
#KITTI    
grey = [128,128,128];
red = [128,0,0];
pink = [128,64,128];
blue = [0,0,192];
greypurple = [64,64,128];
darkyellow = [128,128,0];
lightyellow= [192,192,128];
purple = [64,0,128];
salmon = [192,128,128];
yellowbrown = [64,64,0];
lightblue = [0,128,192];
void =[0,0,0];
       
kitti_colors = [grey, red, pink, blue, greypurple, darkyellow,
          lightyellow, purple, salmon, yellowbrown,
          lightblue, void]; #Not Real Index Values. Only to Compute Statistics.
          #For real index numbers, check modunaries2.py
          
label_list = ["Sky", "Building", "Road", "Sidewalk", "Fence", "Vegetation", "Pole", "Car", "Sign", "Pedestrian", "Cyclist", "Void"]

######OTHER DATA#######
number_classes = 12
void_class_index=11;     
color_list = kitti_colors
confusion_matrix = np.zeros((number_classes, number_classes))
compute_for_void = 1; #***Imp 1->dont compute , 0->compute
classifer_suffixes = ['RF','LR'];
gt_folder ='/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/'
confusion_write_loc = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/results/";
ext_gt = ".png_ann.xml"
ext_test="_test.png.xml" 
ext_exclude = None;
skip_last = 0; 
#Number of images to not calculate. 
#My last 3 images were shit, so I introduced this

####CONFUSION MATRIX 

filename_prefix = "camvid_confusion_";

###########END EDITING HERE



def BGRtoRGBswap(ip_color):
    temp=ip_color[2]
    ip_color[2]=ip_color[0]
    ip_color[0]=temp
    print ip_color

def skiToCvOrViceVersa(ip_image):
    """Convert from Ski-Image format to Opencv format RGB->BGR and Vice Versa. Alpha Channel Unaffected"""
    rearranged = np.zeros_like(ip_image)
    rearranged[:,:,0] = ip_image[:,:,2]
    rearranged[:,:,1] = ip_image[:,:,1]
    rearranged[:,:,2] = ip_image[:,:,0]
    return rearranged

def listImages( name_folder, extension):
    """List all the Images in the folder"""
    filenames = glob.glob(name_folder + "/*" + extension)
    return filenames
    
def imageToLabels(label_colors, img):
    annimg = np.zeros((img.shape[0], img.shape[1]))
    #converting to BGR and checking annotations          
    for i in range(0,len(label_colors)):
        color = label_colors[i]   
        a=(img[:,:,0]==color[0])
        b=(img[:,:,1]==color[1])
        c=(img[:,:,2]==color[2])      
        d=np.logical_and(a,b);
        e=np.logical_and(d,c);
        e=e.astype(int)        
        annimg = annimg + e*i;
        #im = labelsToImage(color_list, annimg);
        #cv2.imshow("Sample", im);
        #cv2.waitKey();
    return annimg;
    
def labelsToImage(label_colors, index_img,is_cv=True):
    annimg = np.zeros((index_img.shape[0], index_img.shape[1],3))
    for i in range(0,len(label_colors)):
        annimg[index_img==i]=label_colors[i];
    if(not is_cv):
        return annimg.astype(np.uint8);
    else:
        return skiToCvOrViceVersa(annimg).astype(np.uint8)
    
   


test_folder1 = '/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/'+'LR'+'/';
test_folder2  = '/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/'+'RF'+'/';

#MAIN PROGRAM STARTS HERE
print "Displaying results...."
test_images_list1 = sorted(listImages(test_folder1,ext_test))
test_images_list2 = sorted(listImages(test_folder2,ext_test))

cv2.namedWindow("GroundTruth")
cv2.namedWindow("LR")
cv2.namedWindow("RF")
for i in range(0,len(test_images_list1)-skip_last):
    print i
    gt_filename_tmp = test_images_list1[i].replace(test_folder1, gt_folder);
    gt_filename = gt_filename_tmp.replace(ext_test, ext_gt);
    
    GT = ocv.read_xml_file(gt_filename, "Ann")  
    TEST_lr = ocv.read_xml_file(test_images_list1[i], "Ann")  
    TEST_rf = ocv.read_xml_file(test_images_list2[i], "Ann")  
    
    rf_rgb=labelsToImage(kitti_colors, TEST_rf, True);
    lr_rgb=labelsToImage(kitti_colors, TEST_lr, True);
    gt_rgb =labelsToImage(kitti_colors, GT, True); 
    cv2.imshow("GroundTruth", gt_rgb)
    cv2.imshow("LR", rf_rgb);
    cv2.imshow("RF", lr_rgb);
    cv2.waitKey(0);
    
cv2.destroyAllWindows();
        
        



    
    