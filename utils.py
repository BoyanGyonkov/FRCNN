import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
import numpy as np

def multiple_bb_iou(boxes1, boxes2):
    boxes1 = tf.cast(tf.convert_to_tensor(boxes1), dtype=tf.float32)
    boxes2 = tf.cast(tf.convert_to_tensor(boxes2), dtype=tf.float32)


    ious = []
    
    for i in range(boxes2.shape[0]):
        box2 = boxes2[i]
        
        xA = tf.math.maximum(boxes1[:,1] , box2[1])
        yA = tf.math.maximum(boxes1[:,0] , box2[0])
        xB = tf.math.minimum(boxes1[:,3] , box2[3])
        yB = tf.math.minimum(boxes1[:,2] , box2[2])

        interArea = tf.math.maximum(0,xB - xA +1) * tf.math.maximum(0,yB - yA +1)

        box1Area = (boxes1[:,3] - boxes1[:,1] + 1) * (boxes1[:,2] - boxes1[:,0] +1)
    
        box2Area = (box2[3] - box2[1] + 1) * (box2[2] - box2[0] +1)
    
        ious.append( interArea/ tf.cast(box1Area + box2Area - interArea, dtype=tf.float32))

    ious = tf.stack(ious)
        

    return ious.numpy()
