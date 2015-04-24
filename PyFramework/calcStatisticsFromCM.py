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
test_folder = '/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/Test/predImages/LR/'
gt_folder ='/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/Test/'
confusion_write_loc = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/results/";
ext_gt = ".png_ann.xml"
ext_test="_test.png.xml" 
ext_exclude = None;
skip_last = 0; 
#Number of images to not calculate. 
#My last 3 images were shit, so I introduced this

####CONFUSION MATRIX 

filename_conf = "camvid_confusion_crf.pkl";

###########END EDITING HERE



              
                
def display_conf(conf, label_list,acc, gb_acc, avg_acc): 
    row_headers = list(label_list)
    row_headers.insert(0,"Label names")
    row_headers.append("Accuracies")
    x = PrettyTable(row_headers)
    x.align["Label names"]="l"
    x.padding_width=1 
    for i in range(0,len(label_list)):
        row_vals = list(conf[i,:])
        name = label_list[i] + "-GT"
        row_vals.insert(0,name)
        row_vals.append(acc[i]*100)
        x.add_row(row_vals)
    print x
    print "Global Accuracy : ", (gb_acc*100)
    print "Average Accuracy : ", (avg_acc*100)
 
def final_statistics(conf):
    accuracies = [0]*len(conf) #Per Class
    
    sumpos=0;
    sumtotal=0;
    for i in range(len(conf)):
        pos = conf[i,i];
        total = sum(conf[i,:]);
        accuracies[i] = pos/total;
        if(total==0):
            accuracies[i]=0;
            
        sumpos=sumpos+pos;
        sumtotal=sumtotal+total;
        
    
    global_accuracy=sumpos/sumtotal
    average_accuracy = sum(accuracies)/(len(accuracies)-compute_for_void);
    
    results = (accuracies, global_accuracy, average_accuracy);
    return results
   

    
#MAIN PROGRAM STARTS HERE


fh  = open(confusion_write_loc+filename_conf,"r");
confusion_matrix = cPickle.load(fh)
fh.close()

acc,gb_acc,avg_acc =final_statistics(confusion_matrix)
display_conf(confusion_matrix, label_list, acc,gb_acc,avg_acc) 


    
    
