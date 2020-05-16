# -*- coding: utf-8 -*-
import cv2
import os
import time
import numpy as np

start = time.time()

confthres=0.5 #0.5
nmsthres=0.4 #0.4

def get_labels(labels_path):
    # load file with your class names
    lpath=os.path.sep.join([labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([weights_path])
    
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([config_path])
    
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on Mask detection dataset (3 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
 
    return net

def get_prediction(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]
    
    dim = (416, 416)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    resized= resized.astype('float32')
    
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(resized, 1/255, (416, 416),[0, 0, 0],#should be image
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    print("[INFO] FPS rate {:.6f} seconds".format(1/(end - start)))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:

            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            #print(classID)
            confidence = scores[classID]
            #print(confidence)
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                
                print(np.round(detection,6))
                
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            try:
                letras=list(map(lambda x: cv2.imread('./digitos_rectangulo/{}_2.png'.format(ord(x))),list(LABELS[classIDs[i]])))
                f=lambda x: np.place(x, x>150, COLORS[classIDs[i]])
                list(map(lambda x: f(x), letras))
                serie_full= letras
                cont_y=y
                cont_x=x
                for i in serie_full:
                    image[max(0,cont_y-5-i.shape[0]):cont_y-5,max(0,cont_x):cont_x+i.shape[1]]=i
                    cont_x=cont_x+i.shape[1]
            except:
                pass
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2) #default 10

    return (image)