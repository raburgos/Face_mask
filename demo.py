# -*- coding: utf-8 -*-
import cv2
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

labelspath="yolo.names"
cfgpath=config
wpath=model
Lables=get_labels(ruta_yolo+labelspath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(ruta_yolo+CFG,ruta_yolo+Weights)
Colors=get_colors(Lables)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("We cannot open webcam")

while True:
    ret, frame = cap.read()
    # resize our captured frame if we need
    frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

    # detect object on our frame
    res=get_prediction(frame,nets,Lables,Colors)

    # show us frame with detection
    cv2.imshow("Web cam input", res)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break