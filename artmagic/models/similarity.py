#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:52:42 2018

@author: adam
"""
from scipy.spatial import distance

import pandas as pd


#def cosine_distance(a,b):
#    return(np.inner(a, b) / (norm(a) * norm(b)))
#
#def euclidian_distance(a,b):
#    return(norm(a-b)*-1)
#
def hamming_distance(a,b):
    '''
    compares distance for binary arrays
    returns number of features that are not the same
    '''
    if max(a)>1:
      a[a>0]=1
      b[b>0]=1
    return(distance.hamming(a,b))
    
def find_matches(pred, #features from user selected image
                 collection_features,  #list of features in the collection
                 images, #list of filenames associated with the features
                 dist='cosine' #distance metric - only cosine is good
                 ): 
    '''
    Finds matches for the features of the selected image, 
    according to the distance metric specified.
    Distance metrics use the scipy package
    '''   
    pred = pred.flatten()
    
    nimages = len(collection_features)
    sims = []
    for i in range(0,nimages):
        if dist=='euclidean':
            sims.append(distance.euclidean(pred.flatten(),
                                           collection_features[i].flatten()))
        elif dist=='hamming':
            sims.append(hamming_distance(pred.flatten(),
                                         collection_features[i].flatten()))
        else: #default to cosine
            sims.append(distance.cosine(pred.flatten(),
                                        collection_features[i].flatten()))
    print('max sim = ' +str(max(sims)))
    similar_images=pd.DataFrame({'imgfile':images,
                                 'simscore':sims})
    return(similar_images)


            
