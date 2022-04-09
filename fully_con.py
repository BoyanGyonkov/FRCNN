import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from tensorflow import keras

class FC(keras.layers.Layer):
    def __init__(self,in_features,out_features, weight_name, bias_name):
        super().__init__(name='dense')
        self.b = tf.Variable(tf.random.normal([out_features], stddev=0.01), dtype=tf.float32,name= bias_name, trainable = True)
        self.out_features = out_features
        #self.is_built = False
        self.w = tf.Variable(tf.random.normal([in_features, self.out_features] , stddev=0.01), dtype=tf.float32,name= weight_name ,trainable = True)

 
    def __call__(self, x):
        #if not self.is_built:
            #self.w = tf.Variable(tf.random.normal([x.shape[-1], self.out_features] , stddev=0.01), dtype=tf.float32, trainable = True)
            #self.is_built = True
      
        return tf.matmul(x,self.w) + self.b

    
