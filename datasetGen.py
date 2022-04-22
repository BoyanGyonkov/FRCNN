import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import numpy as np
from config import Config
from tensorflow.keras.applications.vgg16 import preprocess_input

class Dataset(object):
    def __init__(self,filepaths, gt_boxes):
        self.generator_func = read_and_augment(filepaths,gt_boxes)
        self.filepaths = filepaths
        self.gt_boxes = gt_boxes
  
    def __iter__(self):
        try:
            while True:
                yield self.generator_func.__next__()

        except StopIteration:
            self.generator_func = read_and_augment(self.filepaths,self.gt_boxes)

def read_and_augment(filenames,gt_boxes):
    i = 0
     
    while( i < len(filenames)):
        # Reads iamge and resizes it to specified dimensions
        img = cv2.imread(filenames[i])
        img = cv2.resize(img,(Config.input_shape[1],Config.input_shape[0]))

        #Bounding box annotations are calculated for the resized image
        scale_resize_y = Config.input_shape[0]/800
        scale_resize_x = Config.input_shape[1]/1360
        gt_boxes_for_im = np.array(gt_boxes[i])
        gt_boxes_for_im[:,0] = gt_boxes_for_im[:,0] * scale_resize_y
        gt_boxes_for_im[:,1] = gt_boxes_for_im[:,1] * scale_resize_x
        gt_boxes_for_im[:,2] = gt_boxes_for_im[:,2] * scale_resize_y
        gt_boxes_for_im[:,3] = gt_boxes_for_im[:,3] * scale_resize_x

        
        rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_tensor = tf.convert_to_tensor(rgb_im ,dtype=tf.float32)

        
        rgb_tensor = preprocess_input(rgb_tensor)


        #Augmentation: image is upscaled and cropped to fit the original specified dimensions 
        scale_augment = np.random.default_rng().uniform(1.,1.05,1) # (No more than 5% upscale)
        orig_height = tf.shape(rgb_tensor)[0].numpy()
        orig_width = tf.shape(rgb_tensor)[1].numpy()
        new_height = int(orig_height * scale_augment)
        new_width = int(orig_width * scale_augment)
        
        rgb_tensor = tf.image.resize(rgb_tensor, [new_height, new_width])
        rgb_tensor = tf.image.resize_with_crop_or_pad(rgb_tensor, orig_height, orig_width)

        #Augmentation: image is translated along the y axis 
        y_trans = np.random.randint(-50,50) # (No more than 50 pixels to the left or right)
        rgb_tensor = tfa.image.translate(rgb_tensor, [0, y_trans], fill_mode='nearest')
        

        # Bounding Box annotations are adapted to the augmentations
        gt_boxes_for_im = gt_boxes_for_im * scale_augment
        gt_boxes_for_im[:,1] = gt_boxes_for_im[:,1] - (new_width-orig_width)/2
        gt_boxes_for_im[:,3] = gt_boxes_for_im[:,3] - (new_width-orig_width)/2
        gt_boxes_for_im[:,0] = gt_boxes_for_im[:,0] + y_trans - (new_height-orig_height)/2
        gt_boxes_for_im[:,2] = gt_boxes_for_im[:,2] + y_trans  - (new_height-orig_height)/2


        # randomly flips the image and the corresponding annotations
        flip_flag = np.random.randint(2) # 0 or 1 

        if flip_flag:
            rgb_tensor = tf.image.flip_left_right(rgb_tensor)
            img_center = Config.input_shape[1]//2

            gt_boxes_for_im[:,1] = 2*img_center -gt_boxes_for_im[:,1]
            gt_boxes_for_im[:,3] = 2*img_center -gt_boxes_for_im[:,3]
            box_w = abs(gt_boxes_for_im[:,1] - gt_boxes_for_im[:,3])
            gt_boxes_for_im[:,1] -= box_w
            gt_boxes_for_im[:,3] += box_w
            

        yield tf.expand_dims(rgb_tensor,axis=0), gt_boxes_for_im.astype(int)
        i +=1
