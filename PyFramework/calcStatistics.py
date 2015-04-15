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
test_folder = '/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/crf/'
gt_folder ='/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/'
confusion_write_loc = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/results/";
ext_gt = ".png_ann.xml"
ext_test="_crf.xml" 
ext_exclude = None;
skip_last = 0; 
#Number of images to not calculate. 
#My last 3 images were shit, so I introduced this

####CONFUSION MATRIX 

filename_conf = "camvid_confusion_crf.pkl";

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
    
def labelsToImage(label_colors, index_img):
    annimg = np.zeros((index_img.shape[0], index_img.shape[1],3))
    for i in range(0,len(label_colors)):
        annimg[index_img==i]=label_colors[i];
    outimg = skiToCvOrViceVersa(annimg.astype(np.uint8))
    return outimg
    
def calcStatisticsPerImage(GT_im, TEST_im):
    diff = GT_im - TEST_im
    negs = GT[diff!=0]
    poss = GT[diff==0]
    dims = GT_im.shape
     #1 -> Dont compute, 0->compute
    conf = np.zeros((number_classes, number_classes))
    for y in range(0,dims[0]):
        for x in range(0,dims[1]):
            gt = GT[y,x];
            pred = TEST_im[y,x];
            conf[gt,pred]+=1
            if(gt==void_class_index):
                conf[gt,pred]-=compute_for_void;
            
    return conf;
                
                
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
print "Calculating Statistics and confusion matrix  "; 
test_images_list = sorted(listImages(test_folder,ext_test))
GT=np.zeros([10,10])
TEST=np.zeros([10,10])
cv2.namedWindow("GroundTruth")
#cv2.namedWindow("Test Sample")
cv2.namedWindow("Predicted")
for i in range(0,len(test_images_list)-skip_last):
    
    gt_filename_tmp = test_images_list[i].replace(test_folder, gt_folder);
    gt_filename = gt_filename_tmp.replace(ext_test, ext_gt);
    print i,"-->\t On Image ",test_images_list[i], "..Comparing with..", gt_filename
    GT = ocv.read_xml_file(gt_filename, "Ann")  
    TEST = ocv.read_xml_file(test_images_list[i], "Ann")  
#    GT = imageToLabels( color_list,skiToCvOrViceVersa( cv2.imread(gt_filename) ) );
 #   TEST=imageToLabels( color_list,skiToCvOrViceVersa( cv2.imread(test_images_list[i]) ) );
    confusion_matrix+=calcStatisticsPerImage(GT, TEST)
    cv2.imshow("GroundTruth", labelsToImage(color_list,GT));
    cv2.imshow("Predicted", labelsToImage(color_list,TEST));
    q = 0#cv2.waitKey();
    if q is 113:
        break;
        
cv2.destroyAllWindows()
acc,gb_acc,avg_acc =final_statistics(confusion_matrix)
display_conf(confusion_matrix, label_list, acc,gb_acc,avg_acc) 
fh  = open(confusion_write_loc+filename_conf,"w");
cPickle.dump(confusion_matrix,fh);
fh.close();


    
    