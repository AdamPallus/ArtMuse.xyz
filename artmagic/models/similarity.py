#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:52:42 2018

@author: adam
"""
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage

import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
import glob, os
import operator

import pandas as pd


def cosine_distance(a,b):
    return(np.inner(a, b) / (norm(a) * norm(b)))

def euclidian_distance(a,b):
    return(norm(a-b)*-1)
    
def find_matches(pred, collection_features, images, nimages=8, distance='cosine'): 
#    img = kimage.load_img(imageurl, target_size=(224, 224))
#    x = kimage.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    img = preprocess_input(img)
#    pred = model.predict(img)
    pred = pred.flatten()
    
    nimages = len(collection_features)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        if distance=='cosine':
            sims[i]= cosine_distance(pred.flatten(),collection_features[i].flatten())
        else:
            sims[i]= euclidian_distance(pred.flatten(),collection_features[i].flatten())
    similar_images=dict(zip(images,sims))
    topmatches=sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True)[0:nimages+1]
    return(topmatches)
#    return(similar_images)
    
def find_matches2(pred, collection_features, images, nimages=8, distance='cosine'): 
#    img = kimage.load_img(imageurl, target_size=(224, 224))
#    x = kimage.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    img = preprocess_input(img)
#    pred = model.predict(img)
    pred = pred.flatten()
    
    nimages = len(collection_features)
    sims = np.zeros((nimages, 1))
    for i in range(0,nimages):
        if distance=='cosine':
            sims[i]= cosine_distance(pred.flatten(),collection_features[i].flatten())
        else:
            sims[i]= euclidian_distance(pred.flatten(),collection_features[i].flatten())
    topmatches=pd.DataFrame(imgfile=images, simscore=sims).sort_values(sims,ascending=False).head(nimages)
    return(topmatches)
#    return(similar_images)
            
