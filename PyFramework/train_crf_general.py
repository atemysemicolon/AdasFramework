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

#Collect the data
    
folder_energy_train = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/unaries/LR/Train/";
folder_energy_test =  "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/unaries/LR/Test/";
folder_data_train = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/train/"
folder_out = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/"

ext_desc = ".png_desc.xml"
ext_sup =".png_sup.xml"
ext_nbr= ".png_nbrs.xml"
ext_ann=".png_ann.xml"
ext_energy = "_prob.xml"

filenames,filenames_desc = listFiles(folder_data_train,ext_desc);
filenames_sup = [filename.replace(ext_desc, ext_sup) for filename in filenames_desc]
filenames_ann = [filename.replace(ext_desc, ext_ann) for filename in filenames_desc]
filenames_nbr = [filename.replace(ext_desc, ext_nbr) for filename in filenames_desc]
bla, filenames_energy = listFiles(folder_energy_train,ext_energy);


#Loop over data
    #Get Probabilities
    #Get Ground Truth
    #Get adjacency list of superpixels
X_train = list();
Y_train=list();

for i in range(0,len(filenames_energy)):
        print i,"->",filenames[i]
        segments = ocv.read_xml_file(folder_data_train+filenames_sup[i], "Segments")
        ann = ocv.read_xml_file(folder_data_train+filenames_ann[i], "Ann")
        nbrs = ocv.read_xml_file(folder_data_train+filenames_nbr[i], "Neighbours") 
        energy = ocv.read_xml_file(folder_energy_train+filenames_energy[i], "prob")
        label_list=annImagetoLabels(ann,segments)
        edge_list = SuperpixelAdjacency.convertToAdjacencyList(nbrs)
        #edge_features.shape=len(edge_features),1
        x = (energy, edge_list)
        Y_train.append(np.array(label_list));
        X_train.append(x);

#For Pedestrian, who isn't detected at all        
desc_temp = np.asarray(np.zeros_like(energy)+1, 'float64');
x_temp=(desc_temp, edge_list)
X_train.append(x_temp);
gt_temp =np.zeros((len(energy),)) +9
Y_train.append(np.asarray(gt_temp,'int64'))


#Save Data
print "Saving Data"
cPickle.dump((X_train,Y_train),open(folder_out+"crf_general_data_200.pkl","w"))

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
