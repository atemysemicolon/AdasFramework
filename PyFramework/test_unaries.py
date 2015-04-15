# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:08:30 2015

@author: prassanna
"""





####Modified version of Manager_train.py##########
import cv2
import numpy as np
import opencv_utils as ocv
import cPickle
import sys
import glob
from sklearn import ensemble
from scipy.stats import mode

folder_location = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/"
folder_out_rf = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/RF/"
folder_out_svm = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/SVM/"
folder_out_lr = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/LR/"
file_rf = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/RF_unary_model_only.pkl"
file_svm = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/SVM_unary_model.pkl"
file_lr = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/LR_unary_model_only.pkl"
ext_im = ".png_im.xml"
ext_ann = ".png_ann.xml"
ext_desc = ".png_desc.xml"
ext_sup = ".png_sup.xml"
ext_out = "_test.png"

#DatasetColours!!!
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
          lightblue, void];
dataset_names = ["Sky","Building","Road","Sidewal","Fence","Vegetation",
               "Pole","Car","Sign","Pedestrian","Cyclist","Void"]          
          
dataset_colours  = kitti_colors
#END Info


#Load Classifier
print "Loading Classifier....."
imp = cPickle.load(open("/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/imputer.pkl", "r"))
f = open(file_rf, "r");
clf_rf = cPickle.load(f);
f.close();

clf_svm = 0#cPickle.load(open(file_svm,"r"));
clf_lr = 0#cPickle.load(open(file_lr,"r"));
print "Done Loading."

clfs = (clf_rf,clf_lr,clf_svm);
folder_outs = (folder_out_rf, folder_out_lr, folder_out_svm);

#Helper functions
def skiToCvOrViceVersa(ip_image):
    """Convert from Ski-Image format to Opencv format RGB->BGR and Vice Versa. Alpha Channel Unaffected"""
    rearranged = np.zeros_like(ip_image)
    rearranged[:,:,0] = ip_image[:,:,2]
    rearranged[:,:,1] = ip_image[:,:,1]
    rearranged[:,:,2] = ip_image[:,:,0]
    return rearranged


def listFiles(name_folder, extension):
    """List all the Images in the folder"""
    filenames = glob.glob(name_folder + "/*" + extension)
    filenames_stripped = [filename.replace(name_folder, '') for filename in filenames]    
    return sorted(filenames),sorted(filenames_stripped)

def labelsToImage(label_colors, index_img,is_cv=True):
    annimg = np.zeros((index_img.shape[0], index_img.shape[1],3))
    for i in range(0,len(label_colors)):
        annimg[index_img==i]=label_colors[i];
    if(not is_cv):
        return annimg.astype(np.uint8);
    else:
        return skiToCvOrViceVersa(annimg).astype(np.uint8)
        
def annImagetoLabels(ann_index_image, segments):
    gt_labels = [int(mode(ann_index_image[segments==i],axis=None)[0][0]) for i in range(0,np.amax(segments)+1)]
    return gt_labels

def predLabelsToAnnIndex(segments, pred_list):
    ann_index = np.zeros_like(segments);   
    for i in range(0,len(pred_list)):
        ann_index[np.where(segments==i)] = pred_list[i];
    return ann_index;

    
filenames, filenames_desc =listFiles(folder_location, ext_desc); 
filenames_sup = [filename.replace(ext_desc, ext_sup) for filename in filenames_desc]
filenames_ann = [filename.replace(ext_desc, ext_ann) for filename in filenames_desc]
filenames_out = [filename.replace(ext_desc, ext_out) for filename in filenames_desc]
#gt_global = list();
#desc_global = list();
#cv2.namedWindow("Original");
#cv2.namedWindow("Predicted");



for i in range(0,len(filenames_sup)):
    print i, "->", filenames_sup[i]; 
    segments = ocv.read_xml_file(folder_location+filenames_sup[i], "Segments")
    desc = ocv.read_xml_file(folder_location+filenames_desc[i], "Descriptors")
    #ann = ocv.read_xml_file(folder_location+filenames_ann[i], "Ann")
    print "Transforming.."    
    desc = imp.transform(desc)
    print "Predicting.."
    for k in range(0,1):#len(clfs)):
        clf = clfs[k]
        folder_out = folder_outs[k]
        pred_labels = clf.predict(desc);
        predAnn = predLabelsToAnnIndex(segments, pred_labels);
        #annImage = labelsToImage(dataset_colours, ann);
        predImage = labelsToImage(dataset_colours, predAnn);
        #cv2.imshow("Original", annImage);
        #cv2.imshow("Predicted", predImage);
        ocv.write_xml_file(folder_out+filenames_out[i]+".xml", "Ann", predAnn)
        cv2.imwrite(folder_out+filenames_out[i], predImage);
    #cv2.waitKey(0)
    
#cv2.destroyAllWindows();
    