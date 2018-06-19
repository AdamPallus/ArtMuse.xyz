#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:56:53 2018

@author: adam
"""

from artmagic import app
from artmagic.models.similarity import find_matches

import flask
import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing import image as kimage
import tensorflow as tf
from werkzeug.utils import secure_filename
import pandas as pd
from PIL import ExifTags, Image


collection_features = np.load(os.path.join(app.config['DATA_FOLDER'],
                                           'collection_features_6-17.npy'))
files_and_titles=pd.read_csv(os.path.join(app.config['DATA_FOLDER'],
                                          'files_and_titles_6-17.csv'))

app.secret_key = 'adam'

#I was getting an error because the model was losing track of the graph
#defining graph here lets me keep track of it later as things move around
graph = tf.get_default_graph()

model = VGG16(include_top=False, weights='imagenet')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def autorotate_image(filepath):
    
    '''Phones rotate images by changing exif data, 
    but we really need to rotate them for processing'''
    
    image=Image.open(filepath)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
            exif=dict(image._getexif().items())
    
        if exif[orientation] == 3:
            print('ROTATING 180')
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            print('ROTATING 270')
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            print('ROTATING 90')
            image=image.rotate(90, expand=True)
        image.save(filepath)
        image.close()
    except (AttributeError, KeyError, IndexError):
    # cases: image don't have getexif   
        pass
    return(image)
    
@app.route('/',  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
        # Get method type
    method = flask.request.method
    print(method)


    if method == 'GET':
        return flask.render_template('index.html')
    
    if method == 'POST':
        # No file found in the POST submission
        if 'file' not in flask.request.files:
            print("FAIL")
            return flask.redirect(flask.request.url)

        # File was found
        file = flask.request.files['file']
        if file and allowed_file(file.filename):


            img_file = flask.request.files.get('file')
            
            print('Rotated!')
            #secure file name so stop hackers
            img_name = secure_filename(img_file.filename)

            # Write image to tmp folder so it can be shown on the next page 
            imgurl=os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            file.save(imgurl)
            #check and rotate cellphone images
            img_file = autorotate_image(imgurl)
                
            #load image for processing through the model
            img = kimage.load_img(imgurl, target_size=(224, 224))
            img = kimage.img_to_array(img)
            img = np.expand_dims(img, axis=0)  
            
            #there's an issue with the model losing track of the graph
            #I found this fix by searching for the error I was getting
            #see above
            global graph
            with graph.as_default():
                pred=model.predict(img)
            matches=find_matches(pred, collection_features, 
                                 files_and_titles['imgfile'],dist='cosine')
            
            showresults=files_and_titles.set_index('imgfile',drop=False).join(matches.set_index('imgfile'))
            showresults.sort_values(by='simscore',ascending=True,inplace=True)

            original_url = img_name
            return flask.render_template('results2.html',matches=showresults,original=original_url)
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)
