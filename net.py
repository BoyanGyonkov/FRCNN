import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from tensorflow import keras
import base_model
from config import Config
from rpn import RPN
from prop import ProposalLayer
from fastRCNN import FastRCNN
import cv2


class FRCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.rpn = RPN()
        self.roiLayer = ProposalLayer()
        self.fastRCNN = FastRCNN()
        self.feature_extractor = base_model.get_fe_extractor()


    def joint_training(self, dataset , classes):
        opt = tf.keras.optimizers.Adam(learning_rate = Config.learning_rate)
        
        for i in range(Config.num_epochs):
            for ind,(img,gt_boxes) in enumerate(dataset):
                with tf.GradientTape() as t:

                    feature_map = self.feature_extractor(img)
                    proposals,scores = self.rpn(feature_map)
                    loss_rpn = self.rpn.train_on_img(feature_map,gt_boxes)
                    proposals_pruned = self.roiLayer(proposals,scores, 'train')

                    proposals_pruned = tf.cast(proposals_pruned, dtype=tf.int32)
                    loss_rcnn = self.fastRCNN.train(feature_map, proposals_pruned , gt_boxes , classes[ind])
               
                    grads = t.gradient(loss_rpn+loss_rcnn, self.trainable_variables)
    
                    opt.apply_gradients(zip(grads, self.trainable_variables))

            if(i >-1):
                self.save_weights(os.path.join("G:\\BG\\FR_model\\V5", (str(i+1)+".h5") ) )
                
            print("Epoch " , i+1)

    def train_rpn(self, img , gt_boxes, learning_rate , feature_extractor , train_vars):
        with tf.GradientTape() as t:
            feature_map = feature_extractor(img)

            curr_loss = self.rpn.train_on_img(feature_map, gt_boxes)

            grads = t.gradient(curr_loss, train_vars)
            opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
            opt.apply_gradients(zip(grads,train_vars))

            return curr_loss
        
  
    def rpn_call(self, img, train_or_test='train'):
        feature_map = self.feature_extractor(img)

        proposals,scores = self.rpn(feature_map)

        proposals_pruned = self.roiLayer(proposals,scores, train_or_test)
      
        
        return proposals_pruned, feature_map

   
    def __call__(self , img):
        
        proposals, f_map = self.rpn_call(img , 'test')
        proposals = tf.cast(proposals, dtype=tf.int32)
        proposals, scores =  self.fastRCNN(f_map,proposals)
        proposals = self.roiLayer(proposals, scores, 'test')
        
        return proposals

