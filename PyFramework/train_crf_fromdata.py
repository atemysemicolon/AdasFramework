# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:55:31 2015

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


#Save Data
print "Saving Data"
(X_train,Y_train) = cPickle.load(open("/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/crf_data_200.pkl","r"))

#Do Some Data modifying operations
#Energy * -1 perhaps?

print "Initializing CRF"
nr_states= 12;
class_counts = np.bincount(np.hstack(Y_train))
class_frequency = 1./ class_counts;
class_weights = class_frequency*(nr_states/np.sum(class_frequency))

C = 0.01
experiment_name = "edge_features_one_slack_trainval_%f" % C
model = crfs.EdgeFeatureGraphCRF( n_states = nr_states, inference_method='max-product',
                                 class_weight=class_weights,
                                 symmetric_edge_features=[0])
ssvm = learners.OneSlackSSVM(
    model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10)                                 

#Fit CRF
ssvm.fit(X_train, Y_train)

cPickle.dump(ssvm, open("/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/trained_crf_fromdata.pkl", "w"))



