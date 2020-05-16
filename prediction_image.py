# -*- coding: utf-8 -*-
import cv2
import os
import time
import numpy as np
import argparse

from face_mask_model import get_labels
from face_mask_model import get_colors
from face_mask_model import get_weights
from face_mask_model import get_config
from face_mask_model import load_model
from face_mask_model import get_prediction

#arguments to pass command line 
ap=argparse.ArgumentParser()

ap.add_argument("-y","--yolo_type", required=True,help='choose type of model: "tiny" or "yolov3"')

ap.add_argument("-p","--path", required=True,help='enter image path:')

args=vars(ap.parse_args())


print(args)

if args["yolo_type"]=='tiny':
    
    ruta_yolo='./tiny_yolo/'
    model = 'yolov3-tiny_custom_final.weights'
    config = 'yolov3-tiny_custom.cfg'

elif args["yolo_type"]=='yolov3':

    ruta_yolo='./yolo/'
    model = 'yolov3_custom_train_final.weights'
    config = 'yolov3_custom_train.cfg'
    
def frame_detection(path,config,model,ruta_yolo):
    
    # load our input image
    image = cv2.imread(path)
    labelspath="yolo.names"
    cfgpath=config
    wpath=model
    Lables=get_labels(ruta_yolo+labelspath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(ruta_yolo+CFG,ruta_yolo+Weights)
    Colors=get_colors(Lables)
    res=get_prediction(image,nets,Lables,Colors)
    #main    
    #plt.imshow(res)
    cv2.imwrite(ruta_yolo+'prediction.JPG', res)#res
    
    return

#path="./mask2.JPG"
frame_detection(args["path"],config,model,ruta_yolo)
