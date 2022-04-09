import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from config import Config

def get_fe_extractor():

    #load CNN layers only
    base_model = tf.keras.applications.VGG16(input_tensor=tf.keras.Input(shape=Config.input_shape), include_top = False, weights='imagenet')
    
    #allow fine-tuning of weights
    base_model = tf.keras.Model(base_model.inputs, base_model.layers[-2].output)

    #base_model.summary()
    for i in range(1,7):
        base_model.layers[-i].trainable = True

    base_model.compile()
    return base_model

