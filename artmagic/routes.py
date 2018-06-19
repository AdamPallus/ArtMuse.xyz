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

#collection_features = np.load('/home/adam/artnetwork/saved_collection_features.npy')
#collection_features = np.load('/home/adam/artnetwork/collection_features2.npy')

collection_features = np.load(os.path.join(app.config['DATA_FOLDER'],'collection_features_6-17.npy'))
files_and_titles=pd.read_csv(os.path.join(app.config['DATA_FOLDER'],'files_and_titles_6-17.csv'))

#files_and_titles=pd.read_csv('/home/adam/Downloads/files_and_titles.csv')

#files_and_titles.sort_values(by='imgfile',inplace=True)
#files_and_titles.reset_index(inplace=True)

#imagespath= "/home/adam/artnetwork/fineartamericaspider/output/full"

app.secret_key = 'adam'

#os.chdir(imagespath)
#images=glob.glob("*.jpg")

graph = tf.get_default_graph()
model = VGG16(include_top=False, weights='imagenet')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def autorotate_image(filepath):
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
            print('SUCCESS')
                    # Image info

            img_file = flask.request.files.get('file')
            
            print('Rotated!')
#            img_name = img_file.filename
            img_name = secure_filename(img_file.filename)

            # Write image to static directory 
            #os.getcwd()
            imgurl=os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            file.save(imgurl)
            #check and rotate cellphone images
            img_file = autorotate_image(imgurl)

                
            img = kimage.load_img(imgurl, target_size=(224, 224))
            img = kimage.img_to_array(img)
            img = np.expand_dims(img, axis=0)    
            global graph
            with graph.as_default():
                pred=model.predict(img)
            matches=find_matches(pred, collection_features, files_and_titles['imgfile'],nimages=50)
            matches = pd.DataFrame(matches, columns=['imgfile', 'simscore'])
            matches['simscore']=matches.simscore.astype('double')
            showresults=files_and_titles.set_index('imgfile',drop=False).join(matches.set_index('imgfile'))
            showresults=showresults.sort_values(by='simscore',ascending=False)
            # Delete image when done with analysis
#            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            original_url = img_name
#            return flask.render_template('results2.html',matches=showresults,original=img_name)
            return flask.render_template('results2.html',matches=showresults,original=original_url)
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)
