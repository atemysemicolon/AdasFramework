# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:18:22 2015

@author: prassanna
"""
import cPickle
from sklearn import linear_model,ensemble
folder_model ="/home/prassanna/Development/workspace/CamVid_scripts/FrameworkDump3/model/"

(X,gt_global,fnames) = cPickle.load(open(folder_model+"Data_unary_model.pkl","r"))


print "fitting Random Forest...."
clf = ensemble.RandomForestClassifier(class_weight="auto")
clf.fit(X,gt_global)
cPickle.dump(clf, open(folder_model+"RF_unary_model_only.pkl","w"))
del clf