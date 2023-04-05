from PIL import Image
import requests
from io import BytesIO
from numpy import *
import numpy as np
import keras
from django.utils.timezone import datetime
from django.conf import settings
import math

def gri_severity(gri_img,category):
    #gri_img = CloudinaryImage(gri_img).image( format = "png")
    response = requests.get(gri_img.url)
    gri_img = Image.open(BytesIO(response.content))
    print(gri_img)
    gri_gray_img = gri_img.resize((200,200)).convert('L') #resize and convert to gray scale
    gri_gray_img_matrix = array([array(gri_gray_img).flatten()],'f')
    gri_gray_img_matrix = gri_gray_img_matrix.reshape(gri_gray_img_matrix.shape[0],200,200,1).astype('float32')
    gri_gray_img_matrix = gri_gray_img_matrix / 255
    if category == "Garbage":
        severity = 0
        garbage_severity_model = keras.models.load_model("ml_models/garbage_reduce_severity.h5")
        garbage_severity_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        severity = garbage_severity_model.predict(gri_gray_img_matrix)
        severity = np.argmax(severity[0])
    elif category == "POTHOLE":
        #print("No ML model for given category, default severity will: 0")
        severity = 0
        pothole_severity_model = keras.models.load_model("ml_models/pothole_reduce_severity.h5")
        pothole_severity_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        severity = pothole_severity_model.predict(gri_gray_img_matrix)
        severity = np.argmax(severity[0])
    elif category == "Fallen Tree":
        #print("No ML model for given category, default severity will: 0")
        severity = 0
        FallenTree_severity_model = keras.models.load_model("ml_models/fallenTree_reduce_severity.h5")
        FallenTree_severity_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        severity = FallenTree_severity_model.predict(gri_gray_img_matrix)
        severity = np.argmax(severity[0])
    else:
        print("No ML model for given category, default severity will: 0")
        severity = 0
    return severity

def gri_priority(gri_obj):
    noUpvotes = int(gri_obj.gri_upvote)
    categoryValue = int(gri_obj.gri_category.cat_value)
    noDaysPast = (datetime.now().date() - gri_obj.gri_timeStamp.date()).days
    if noDaysPast>settings.MAX_DAY:
        noDaysPast = settings.MAX_DAY
    severity = int(gri_obj.gri_severity)
    print([severity,noUpvotes,categoryValue,noDaysPast])
    normalize_noUpvotes = normalizeValue(noUpvotes,0,settings.MAX_UPVOTE)
    normalize_noDaysPast = normalizeValue(noDaysPast,1,settings.MAX_DAY)
    normalize_categoryValue = normalizeValue(categoryValue,1,settings.MAX_CATEGORY_VALUE)
    normalize_severity = normalizeValue(severity,0,settings.MAX_SEVERITY)

    GRI_VALUES_NORMALIZE = [normalize_severity,normalize_noUpvotes,normalize_categoryValue, normalize_noDaysPast]
    print(GRI_VALUES_NORMALIZE)
    score = np.sum([GRI_VALUES_NORMALIZE[i] * settings.WEIGHTS[i] for i in range(len(GRI_VALUES_NORMALIZE))])
    print(score)
    return 10 - math.ceil(score)

def normalizeValue(val,minVal,maxVal):
  return ((val-minVal)/(maxVal-minVal))*10
    
