import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from tensorflow import keras
import math

class AdaptivePooling(keras.layers.Layer):
    def __init__(self, output_shape):
        super().__init__(name='roi_pooling')
        self.h_bins = output_shape[0]
        self.w_bins = output_shape[1]

    def __call__(self, inputs):
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        
        split_size_rows = [height//self.h_bins for i in range(self.h_bins)]
        for i in tf.range(height % self.h_bins):
            split_size_rows[i] +=1
     
 
        split_size_cols = [width//self.w_bins for i in range(self.w_bins)]
        for i in range(width % self.w_bins):
            split_size_cols[i] +=1


   
        split_rows = tf.split(inputs, split_size_rows, axis=1)
        
        for i in range(self.h_bins):
            split_rows[i] = tf.reduce_max(split_rows[i], axis=1)

        split_rows = tf.stack(split_rows, axis =1)
   

        split_cols = tf.split(split_rows, split_size_cols, axis=2)
        for i in range(self.w_bins):
            split_cols[i] = tf.reduce_max(split_cols[i], axis=2)
        split_cols = tf.stack(split_cols, axis = 2)

  
        return split_cols
