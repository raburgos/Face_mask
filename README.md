# Face_mask
 Face Mask - Object Detection

This repository it´s about Object Detection using YOLOv3 and tiny yolov3, with custom training data to detect if people is using mask or not (also I train class label 'semi-mask'; in this case isn´t work well due to lack of images in the initial training process).

# **Setting up everything!**

* Git clone this repository

* If you have Anaconda distribution in your machine you can create virtual environment, in order to not have problems with any library dependencies (you can directly use this excellent tutorial https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) or use next steps:
 * Go to Anaconda prompt in your machine:
  ![step1](/misc/step1.png)
 * Create virtual environment using "conda create -n env_name python=3.7"
 ![step2](/misc/step2.png)
 * Then you have to activate your virtual environment (in this case I create face_mask env). Command: conda activate env_name.
 ![step3](/misc/step3.png)
 * Now you can install opencv (version 4.1.0.25) using pip install opencv-contrib-python==4.1.0.25. This will install opencv library and all necessary dependencies (like numpy) as well. Opencv is responsable to make predictions using YOLO weights and cfg (trained using Darknet framework)
 ![step4](/misc/step8.png)
* Now you can change directory to the path where you git clone this repository.
![step5](/misc/step6.png)
* User your webcam to give a try (you can use yolov3 or tiny yolo as well!
![step7](/misc/step7.png)
