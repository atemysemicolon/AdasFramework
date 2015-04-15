# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:43:12 2015

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

file_rf = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/RF_unary_model_only.pkl"
file_svm = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/SVM_unary_model.pkl"
file_lr = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/LR_unary_model_only.pkl"
file_imp= "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/imputer.pkl"
folder_out = ["/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/unaries/RF/Test/", "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/unaries/LR/Test/"]

ext_im = ".png_im.xml"
ext_ann = ".png_ann.xml"
ext_desc = ".png_desc.xml"
ext_sup = ".png_sup.xml"

#Load Classifier
print "Loading Classifier....."
imp =cPickle.load(open(file_imp,"r"));
f = open(file_rf, "r");
clf1 = cPickle.load(f);
f.close()
clf_svm =0 #cPickle.load(open(file_svm,"r"));
clf_lr = cPickle.load(open(file_lr,"r"));
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
    

#folder_in_rf = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/RF/"
#folder_in_svm = 0#"/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/SVM/"
#folder_in_lr = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/predImages/LR/"
#folder_locations = [folder_in_rf, folder_in_lr, folder_in_svm]
folder_location = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/"

#Getting filenames for Random Forests
filenames, filenames_desc =listFiles(folder_location, ext_desc); 
filenames_sup = [filename.replace(ext_desc, ext_sup) for filename in filenames_desc]
filenames_ann = [filename.replace(ext_desc, ext_ann) for filename in filenames_desc]
filename_extless = [filename.replace(ext_desc, "") for filename in filenames_desc] 
gt_global = list();
desc_global = list();


for i in range(0,len(filenames_desc)):
    print i, "->", filenames_sup[i]; 
    for j in range(0,len(clfs)):
        print "On classifier : ",j
        
        desc = ocv.read_xml_file(folder_location+filenames_desc[i], "Descriptors")
        desc = imp.transform(desc);
        pred = clfs[j].predict_log_proba(desc);
        #predEnergy = clf.predict_log_proba(desc);
        ocv.write_xml_file(folder_out[j]+filename_extless[i]+"_prob.xml", "prob", pred)