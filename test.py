import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
import cv2
from net import FRCNN
import pandas as pd
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import time


def im_read(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img,(2048,1216))

    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_im



img_folder = "C:\Train"

        
rcnn = FRCNN()
rcnn.built = True
rcnn.load_weights("G:\\BG\\FR_model\\V4\\32.h5")
im_file_name = "C:\\Test\\00006.ppm"

rgb_tensor = tf.convert_to_tensor(im_read(im_file_name), dtype= tf.float32)
rgb_tensor = tf.expand_dims(rgb_tensor, axis=0)
rgb_tensor = preprocess_input(rgb_tensor)

boxes = rcnn(rgb_tensor)
boxes = tf.cast(boxes, dtype=tf.int32)

img = cv2.imread(im_file_name)
img = cv2.resize(img,(1360,800))
cv2.imshow('original', img)


img_draw = img.copy()
for box in boxes.numpy():
    img_draw = cv2.rectangle(img_draw, (int(box[1]/1.506),int(box[0]/1.52)), (int(box[3]/1.506), int(box[2]/1.52)), (0,255,0) , 1)

cv2.imshow('detection' , img_draw)
cv2.waitKey(0)
