# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:30:03 2015

@author: prassanna
"""

import numpy as np
import opencv_utils as ocv
import itertools
from scipy.stats import mode

bias_energy=0.2

def PySuperpixelNeibhours(nbrs):
    nbrs_dict = dict();
    for i in range(0,len(nbrs)):
        nbr_i = nbrs[i,:]+1; #to get -1 as 0
        nbrs_dict[i]=list(np.trim_zeros(nbr_i,trim='b')-1);
        #Need to trim trailing 1s        
    return nbrs_dict
    
def convertToAdjacencyList(nbrs):
    nbrs_list  = list();
    for i in range(0,len(nbrs)):
        nbr_i = nbrs[i,:];
        for j in range(0,len(nbr_i)):
            if(nbr_i[j]<0): #REmoving Trailing 0s
                break;
            else:
                a=(i<nbr_i[j])*i + (i>=nbr_i[j])*nbr_i[j]
                b=(i<nbr_i[j])*nbr_i[j] + (i>=nbr_i[j])*i
                nbrs_list.append([a,b]);
    nbrs_list.sort()
    nbrs_list=list(nbr for nbr,_ in itertools.groupby(nbrs_list));
    return np.asarray(nbrs_list,'int64')

def annImagetoLabels(ann_index_image, segments):
    gt_labels = [int(mode(ann_index_image[segments==i],axis=None)[0][0]) for i in range(0,np.amax(segments)+1)]
    return gt_labels     

def convertToAdjacencyListPotts(nbrs,segments,label_list):
    nbrs_list  = list();
    edge_features = list();
    
    for i in range(0,len(nbrs)):
        nbr_i = nbrs[i,:];
        for j in range(0,len(nbr_i)):
            if(nbr_i[j]<0): #REmoving Trailing 0s
                break;
            else:
                a=(i<nbr_i[j])*i + (i>=nbr_i[j])*nbr_i[j]
                b=(i<nbr_i[j])*nbr_i[j] + (i>=nbr_i[j])*i
                nbrs_list.append([a,b]);
                #val =(label_list[a]==label_list[b])*(1-bias_energy) + (label_list[a]!=label_list[b])*(bias_energy)
                
    nbrs_list.sort()
    nbrs_list=list(nbr for nbr,_ in itertools.groupby(nbrs_list));
    edge_features = [(float(label_list[l[0]]==label_list[l[1]])*bias_energy + int(label_list[l[0]]!=label_list[l[1]])*(1-bias_energy)) for l in nbrs_list]
    edge_features = np.transpose(np.asarray(edge_features,'float64'))
    edge_features.shape =len(edge_features),1
    return np.asarray(nbrs_list,'int64'), edge_features


def annImagetoLabels(ann_index_image, segments):
    gt_labels = [int(mode(ann_index_image[segments==i],axis=None)[0][0]) for i in range(0,np.amax(segments)+1)]
    return gt_labels
nbr = ocv.read_xml_file("/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/train/0001TP_006750.png_nbrs.xml", "Neighbours")        
segments = ocv.read_xml_file("/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/train/0001TP_006750.png_sup.xml", "Segments")        
labels =annImagetoLabels(ocv.read_xml_file("/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/train/0001TP_006750.png_ann.xml", "Ann"), segments)

l,e = convertToAdjacencyListPotts(nbr,segments,labels);
        