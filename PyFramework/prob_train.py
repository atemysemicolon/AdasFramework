# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:34:51 2015

@author: prassanna
"""

import cv2
import numpy as np
import opencv_utils as ocv
import cPickle
import sys
import glob
from sklearn import ensemble
from scipy.stats import mode
from sklearn import preprocessing

#Script to Output Probabilities from a trained classifier. Almost like test_unaries.

file_rf = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Model/RF_unary_model.pkl"
file_svm = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Model/SVM_unary_model.pkl"
file_lr = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Model/LR_unary_model.pkl"
#file_imp= "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/model/imputer.pkl"
folder_out = ["/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/unaries/RF/Train/", "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/unaries/LR/Train/"]

ext_im = ".png_im.xml"
ext_ann = ".png_ann.xml"
ext_desc = ".png_desc.xml"
ext_sup = ".png_sup.xml"

#Load Classifier
print "Loading Classifier....."
f = open(file_rf, "r");
clf1,imp = cPickle.load(f);
f.close()

clf_svm =0 #cPickle.load(open(file_svm,"r"));
clf_lr,imp = cPickle.load(open(file_lr,"r"));
clfs = [clf1, clf_lr]#, clf_svm];

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
    
def annImagetoLabels(ann_index_image, segments):
    gt_labels = [int(mode(ann_index_image[segments==i],axis=None)[0][0]) for i in range(0,np.amax(segments)+1)]
    return gt_labels
    

folder_location = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump4/Train/"

#Getting filenames for Random Forests
filenames, filenames_desc =listFiles(folder_location, ext_desc); 
filenames_sup = [filename.replace(ext_desc, ext_sup) for filename in filenames_desc]
filenames_ann = [filename.replace(ext_desc, ext_ann) for filename in filenames_desc]
filename_extless = [filename.replace(ext_desc, "") for filename in filenames_desc] 
gt_global = list();
desc_global = list();


for i in range(0,1):#len(filenames_sup)):
    print i, "->", filenames_sup[i]; 
    for j in range(0,len(clfs)):
        print "On classifier : ",j
        
        desc = ocv.read_xml_file(folder_location+filenames_desc[i], "Descriptors")
        desc = imp.transform(desc);
        pred = clfs[j].predict_log_proba(desc);
        #predEnergy = clf.predict_log_proba(desc);
        ocv.write_xml_file(folder_out[j]+filename_extless[i]+"_prob.xml", "prob", pred)
    #ocv.write_xml_file(folder_out+filenames_extless[i]+"_energy.xml", "energy", predEnergy)