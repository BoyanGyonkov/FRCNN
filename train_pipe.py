import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from config import Config
from net import FRCNN
import cv2
import pandas as pd
import numpy as np
from datasetGen import Dataset


img_folder = "C:\Train"

df = pd.read_csv("C:\Train\gt.txt", sep=';', header=None, names=['fname' , 'x1' , 'y1', 'x2' , 'y2' , 'cls'])

file_paths = []
gt_boxes = []
classes = []
class_types = [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16] , [33, 34, 35, 36, 37, 38, 39, 40] , [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] , [6, 12, 13, 14, 17, 32 , 41 , 42] ]

first = True
for i in df.index:
    if(first):
        first = False
        file_paths.append(os.path.join("C:\Train" , df['fname'][i]))
        gt_boxes.append([[df['y1'][i] , df['x1'][i] , df['y2'][i] , df['x2'][i]]])
        for cls_type in range(len(class_types)):
            if df['cls'][i] in class_types[cls_type]:
                classes.append([cls_type+1])
                break
    elif(os.path.join("C:\Train", df['fname'][i]) != file_paths[-1]):
        file_paths.append(os.path.join("C:\Train" , df['fname'][i]))
        gt_boxes.append([[df['y1'][i] , df['x1'][i] , df['y2'][i] , df['x2'][i]]])
        for cls_type in range(len(class_types)):
            if df['cls'][i] in class_types[cls_type]:
                classes.append([cls_type+1])
                break
    else:
        gt_boxes[-1].append([df['y1'][i] , df['x1'][i] , df['y2'][i] , df['x2'][i]])
        for cls_type in range(len(class_types)):
            if df['cls'][i] in class_types[cls_type]:
                classes[-1].append(cls_type+1)
                break


dataset = Dataset(file_paths, gt_boxes)

        
rcnn = FRCNN()

rcnn.joint_training(dataset, classes)

#rcnn.save_weights("G:\\BG\\FRCNN2\\Model\\ver6.h5")


