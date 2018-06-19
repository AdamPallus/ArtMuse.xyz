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
    a[a>0]=1
    b[b>0]=1
    return(distance.hamming(a,b))
    
def find_matches(pred, collection_features, images, dist='cosine'): 

    pred = pred.flatten()
    
    nimages = len(collection_features)
    sims = []
    for i in range(0,nimages):
        if dist=='cosine':
#            sims[i]= cosine_distance(pred.flatten(),collection_features[i].flatten())
            sims.append(distance.cosine(pred.flatten(),collection_features[i].flatten()))
        elif dist=='hamming':
            sims.append(hamming_distance(pred.flatten(),collection_features[i].flatten()))
        else:
            sims.append(distance.euclidean(pred.flatten(),collection_features[i].flatten()))
    print('max sim = ' +str(max(sims)))
    similar_images=pd.DataFrame({'imgfile':images,
                                 'simscore':sims})
    return(similar_images)


            
