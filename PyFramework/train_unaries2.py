# -*- coding: utf-8 -*-
"""
Created on Wed Apr  16 13:48:59 2015

@author: prassanna
"""



####Modified version of Manager_train.py##########
import cv2
import numpy as np
import opencv_utils as ocv
import cPickle
import sys
import glob
from sklearn import ensemble,linear_model, svm, preprocessing
from scipy.stats import mode


folder_location = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump5/Train/"
folder_out ="/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump5/Model/"
ext_im = ".png_im.xml"
ext_ann = ".png_ann.xml"
ext_desc = ".png_desc.xml"
ext_sup = ".png_sup.xml"


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
    gt_labels_temp = [ np.any(ann_index_image[segments==i])for i in range(0,np.amax(segments)+1)]
    gt_labels = [int(mode(ann_index_image[segments==i],axis=None)[0][0]) if gt_labels_temp[i]==True else 11  for i in range(0,np.amax(segments)+1) ]
    return gt_labels
    
filenames, filenames_desc =listFiles(folder_location, ext_desc); 
filenames_sup = [filename.replace(ext_desc, ext_sup) for filename in filenames_desc]
filenames_ann = [filename.replace(ext_desc, ext_ann) for filename in filenames_desc]

gt_global = list();
desc_global = list();
fnames  = list();
for i in range(0,len(filenames_sup)):
    print i, "->", folder_location+filenames_sup[i]; 
    fnames.append(filenames_sup[i]);
    segments = ocv.read_xml_file(folder_location+filenames_sup[i], "Segments")
    desc = ocv.read_xml_file(folder_location+filenames_desc[i], "Descriptors")
    ann = ocv.read_xml_file(folder_location+filenames_ann[i], "Ann")
    gt_labels = annImagetoLabels(ann, segments);
    gt_global.extend(gt_labels);
    desc_global.append(desc);

        

n_states=12
clf=ensemble.RandomForestClassifier(class_weight="auto")
#Now Making data look good
imp=preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0)

X = np.vstack(desc_global)
imp.fit(X)
X=imp.transform(X)

cPickle.dump((X,gt_global,fnames), open(folder_out+"Data_unary_model.pkl","w"))
del desc_global

print "Saving imputer.."
cPickle.dump(imp, open(folder_out+"imp_unary_model_only.pkl","w"))
del imp

print "fitting Random forest...."
clf.fit(X,gt_global);
cPickle.dump(clf, open(folder_out+"RF_unary_model_only.pkl","w"))
del clf


print "fitting Logisitic Regression...."
clf2 = linear_model.LogisticRegression(class_weight="auto")
clf2.fit(X,gt_global)
cPickle.dump(clf2, open(folder_out+"LR_unary_model_only.pkl","w"))
del clf2

print "Done!"