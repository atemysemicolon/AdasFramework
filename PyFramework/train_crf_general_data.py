# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:56:42 2015

@author: prassanna
"""
import SuperpixelAdjacency
import opencv_utils as ocv
import pystruct
import numpy as np
import cPickle
import sys
import glob
from scipy import stats

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

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
#g1=list()    
def annImagetoLabels(ann_index_image, segments):
    #global g1
    gt_labels_temp = [ np.any(ann_index_image[segments==i])for i in range(0,np.amax(segments)+1)]
    gt_labels = [int(stats.mode(ann_index_image[segments==i],axis=None)[0][0]) if gt_labels_temp[i]==True else 11  for i in range(0,np.amax(segments)+1) ]
    #g1=gt_labels
    return gt_labels


#Save Data
print "Loading Data...."
(X_train, Y_train)=cPickle.load(open(folder_out+"crf_general_data_200.pkl","r"))

print "Initializing CRF"
nr_states= 12;
class_counts = np.bincount(np.hstack(Y_train))
class_frequency = 1./ class_counts;
class_weights = class_frequency*(nr_states/np.sum(class_frequency))

C = 0.01
experiment_name = "edge_features_one_slack_trainval_%f" % C
model = crfs.GraphCRF( n_states = nr_states, inference_method='max-product',
                                 class_weight=class_weights)
                                 #symmetric_edge_features=[0])
                                 
ssvm = learners.OneSlackSSVM(
    model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10)                                 

#Fit CRF
ssvm.fit(X_train, Y_train)

cPickle.dump(ssvm, open(folder_out+"trained_general_crf.pkl", "w"))

#Use Potts Model -> Manually

#Modify for edge_function
