#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:55:13 2018

@author: adam
"""


from flask import Flask
#import os

#UPLOAD_FOLDER = "uploads/"
#if not os.path.isdir(UPLOAD_FOLDER):
#    os.mkdir(UPLOAD_FOLDER)

app = Flask(__name__)
app.config.from_object(__name__) 
app.config.update(dict(
#        UPLOAD_FOLDER = os.path.basename('tmp'),
#        UPLOAD_FOLDER = "/home/adam/artnetwork/mvpapp/app/static/img/tmp",
        UPLOAD_FOLDER = "artmagic/static/img/tmp/",
        DATA_FOLDER = "artmagic/models/",
#        UPLOAD_FOLDER = UPLOAD_FOLDER,
        ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
))


from artmagic import routes