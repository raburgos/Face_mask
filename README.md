# Face_mask
## Face Mask - Object Detection

This repository it´s about Object Detection using YOLOv3 and tiny yolov3, with custom training data to detect if people is using mask or not (also I trained class label 'semi-mask'; in this case isn´t work well due to lack of images in the initial training process).

## Special thanks and references

* First of all thanks to all open source community! Many functions in face_mask_model.py were extracted from Chandan Dwivedi https://gist.github.com/Jargon4072.
* Also I used as reference to start with the module dnn from opencv: https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/ and https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
* In order to train YOLOv3 and Tiny YOLOv3 I used as reference to Quang Nguyen https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1

## **Setting up everything!**

1. Git clone this repository

1. Download YOLOv3 weights [here](http://www.mediafire.com/file/5r7ooamujxgo5pk/yolov3_custom_train_final.weights/file) and place the file in "yolo" folder. Also if you want tiny yolo model you can download weights [here](http://www.mediafire.com/file/dmqsp10bu7rqh9p/yolov3-tiny_custom_final.weights/file) and place the file in "tiny_yolo" folder.

1. If you have Anaconda distribution in your machine you can create virtual environment, in order to not have problems with any library dependencies (you can directly use this excellent tutorial https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) or use next steps:
   1. Go to Anaconda prompt in your machine:
      ![step1](/misc/step1.png)
   1. Create virtual environment using "conda create -n env_name python=3.7"
      ![step2](/misc/step2.png)
   1. Then you have to activate your virtual environment (in this case I create face_mask env). Command: conda activate env_name.
      ![step3](/misc/step3.png)
   1. Now you can install opencv (version 4.1.0.25) using pip install opencv-contrib-python==4.1.0.25. This will install opencv library       and all necessary dependencies (like numpy) as well. Opencv is responsable to make predictions using YOLO weights and cfg (trained       using Darknet framework)
   ![step4](/misc/step8.png)
   1. Now you can change directory to the path where you git clone this repository.
      ![step5](/misc/step6.png)
1. User your webcam to give a try (you can use yolov3 or tiny yolo as well!)
      ![step7](/misc/step7.png)<br />
   Result:<br />
   ![demo](/misc/demo.gif)
1. You can make predictions over images. Example: python prediction_image.py -y yolov3 -p 1.JPG.
   ![step10](/misc/step10.png)
   After '-y' you have to specify model (yolov3 or tiny). After '-p' you have to specify path to the image, in this case we use image     1.JPG.<br />
   Result:
   ![predicion_image_example](/misc/prediction_image_example.JPG)
