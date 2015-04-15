# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:38:00 2015

@author: prassanna
"""
from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger
import cPickle
import numpy as np
import glob
import opencv_utils as ocv
import cv2
import SuperpixelAdjacency
from scipy.stats import mode
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
          
label_list  = kitti_colors

def listFiles(name_folder, extension):
    """List all the Images in the folder"""
    filenames = glob.glob(name_folder + "/*" + extension)
    filenames_stripped = [filename.replace(name_folder, '') for filename in filenames]   
    return sorted(filenames),sorted(filenames_stripped)

def skiToCvOrViceVersa(ip_image):
    """Convert from Ski-Image format to Opencv format RGB->BGR and Vice Versa. Alpha Channel Unaffected"""
    rearranged = np.zeros_like(ip_image)
    rearranged[:,:,0] = ip_image[:,:,2]
    rearranged[:,:,1] = ip_image[:,:,1]
    rearranged[:,:,2] = ip_image[:,:,0]
    return rearranged
    
def predLabelsToAnnIndex(segments, pred_list):
    ann_index = np.zeros_like(segments);   
    for i in range(0,len(pred_list)):
        ann_index[np.where(segments==i)] = pred_list[i];
    return ann_index;

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

file_ssvm = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/trained_crf.pkl"
fh = open(file_ssvm,"r")
ssvm = cPickle.load(fh)
fh.close();
crf = ssvm.model
#CRF trained on LR energy
folder_energy ="/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/unaries/LR/Test/"
folder_superpixels = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/test/"
ext_superpixels = ".png_sup.xml"
ext_nbr = ".png_nbrs.xml"
ext_energy="_prob.xml"
ext_ann=".png_ann.xml"
ext_out_image = "_crf.png"
ext_out_ann = "_crf.xml"
folder_output = "/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/crf/"
fh,filenames_ext = listFiles(folder_energy, ext_energy);
filenames = [f.replace(ext_energy,"") for f in filenames_ext]

for i in range(0,len(filenames)):
    print i, "->",filenames[i]
    print "Loading all files..."
    energies = ocv.read_xml_file(folder_energy+filenames[i]+ext_energy, "prob");
    segments = ocv.read_xml_file(folder_superpixels+filenames[i]+ext_superpixels, "Segments")
    nbrs = ocv.read_xml_file(folder_superpixels+filenames[i]+ext_nbr, "Neighbours")
    ann = ocv.read_xml_file(folder_superpixels+filenames[i]+ext_ann, "Ann")
    gt_labels = annImagetoLabels(ann, segments);
    
    print "Necessary Transformations..."
    edge_list,edge_features= SuperpixelAdjacency.convertToAdjacencyListPotts(nbrs,segments,gt_labels)
    #edge_features.shape=len(edge_features),1
    x=(energies,edge_list,edge_features)
    
    print "Inference...."
    predlabelList = crf.inference(x, ssvm.w, relaxed=True);   
    
    print "Final Operations..."
    predAnn = predLabelsToAnnIndex(segments,predlabelList)
    predImage = labelsToImage(label_list, predAnn, True);
    ocv.write_xml_file(folder_output+filenames[i]+ext_out_ann,"Ann",predAnn);
    cv2.imwrite(folder_output+filenames[i]+ext_out_image, predImage);