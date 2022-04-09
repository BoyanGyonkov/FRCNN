import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from tensorflow import keras
from config import Config
from utils import multiple_bb_iou
import numpy as np

class ProposalLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__(name='proposal_layer')


    def __call__(self, bbox_locs, object_scores, train_or_test = 'train'):
        if(train_or_test == 'train'):
            n_post_nms = Config.train_n_nms
        else:
            n_post_nms = Config.test_n_nms


        #convert bbox_locs to format: y1,x1,y2,x2 from y,x,h,w
        y1 = bbox_locs[:,0] - 0.5 * bbox_locs[:,2]
        x1 = bbox_locs[:,1] - 0.5 * bbox_locs[:,3]
        y2 = bbox_locs[:,0] + 0.5 * bbox_locs[:,2]
        x2 = bbox_locs[:,1] + 0.5 * bbox_locs[:,3]


        #Check if box has sides bigger than a predefined minimum, remove those which don't
        large_enough_index = tf.where(tf.math.logical_and(tf.math.greater((y2-y1), Config.prop_min_size) , tf.math.greater( (x2-x1), Config.prop_min_size )))
        bbox_reform = tf.transpose([y1 , x1 , y2 , x2])
        bbox_reform = tf.reshape(bbox_reform,[-1,4])
        bbox_reform = tf.gather(bbox_reform, large_enough_index)
        bbox_reform = tf.reshape(bbox_reform,[-1,4])

      
        object_scores = tf.gather(object_scores, large_enough_index)
        object_scores = tf.reshape(object_scores , [-1])

        #clip bboxes to fit in image
        bbox_cliped = []
        bbox_cliped.append(tf.clip_by_value(bbox_reform[:,0], clip_value_min = 0, clip_value_max = Config.input_shape[0] ))
        bbox_cliped.append(tf.clip_by_value(bbox_reform[:,1], clip_value_min = 0, clip_value_max = Config.input_shape[1] ))
        bbox_cliped.append(tf.clip_by_value(bbox_reform[:,2], clip_value_min = 0, clip_value_max = Config.input_shape[0] ))
        bbox_cliped.append(tf.clip_by_value(bbox_reform[:,3], clip_value_min = 0, clip_value_max = Config.input_shape[1] ))
        bbox_reform = tf.transpose(tf.stack(bbox_cliped))
    

        selected_indexes = tf.image.non_max_suppression(bbox_reform, object_scores, n_post_nms, Config.nms_thresh)

        return tf.reshape(tf.gather(bbox_reform,selected_indexes) , [-1,4]) 
