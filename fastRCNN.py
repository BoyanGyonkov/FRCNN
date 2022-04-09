import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import Config
from utils import multiple_bb_iou
from pool import AdaptivePooling
from fully_con import FC

class FastRCNN(keras.layers.Layer):
    def __init__(self):
        super().__init__(name='fastRCNN')
        self.adaptivePooling = AdaptivePooling((2,2))
        self.fc1 = FC(2*2*512,200, 'fc1' ,'b1')
        self.fc2 = FC(200,200, 'fc2' , 'b2')

        self.reg_fc = FC(200,4*4 , 'fc3' , 'b3')
        self.class_fc = FC(200,5 , 'fc4' , 'b4')

    def train(self,feature_map,proposal_coords,gt_boxes,gt_classes):
        #calculate IOU for all (proposal, ground truth box) combinations

        ious = multiple_bb_iou(proposal_coords,gt_boxes)
        indices_max_iou = tf.argmax(ious, axis=0)
        values_max_iou = ious[indices_max_iou, np.arange(tf.shape(ious)[1])]

        #Label each proposal as a negative or positive sample
        labels = np.zeros(tf.shape(values_max_iou)[0])
        labels[np.where(values_max_iou > Config.positive_iou_threshold)] = 1
        self.non_zero_label_count = np.count_nonzero(labels)
        labels[np.where(values_max_iou < Config.negative_iou_threshold)] = -1

        
        #generate a list of of ground_truth boxes,where each box corresponds to a proposal
        gt_box_for_prop = np.array([gt_boxes[i] for i in indices_max_iou])
        
        #repeat for classes
        gt_class_for_prop = np.array([gt_classes[i] for i in indices_max_iou])
        
        #negative proposals are labeled as class 0(background)
        gt_class_for_prop[np.where(labels == 0)[0]] = 0
        gt_class_for_prop[np.where(labels == -1)[0]] = 0

        #Randomly choose positive and negative samples for training
        if(self.non_zero_label_count > Config.pos_samples_rcnn):
            
            pos_samples = np.random.choice(np.where(labels == 1)[0] , size = Config.pos_samples_rcnn)
            neg_samples = np.random.choice(np.where(labels == -1)[0] , size = Config.neg_samples_rcnn)
        else:      
            if(self.non_zero_label_count != 0):
                
                pos_samples = np.random.choice(np.where(labels == 1)[0] , size = self.non_zero_label_count)
                neg_samples = np.random.choice(np.where(labels == -1)[0] , size = (Config.neg_samples_rcnn + Config.pos_samples_rcnn - self.non_zero_label_count))
            
            else:
                pos_samples = np.random.choice(np.where(labels == -1)[0] , size = Config.pos_samples_rcnn)
                neg_samples = np.random.choice(np.where(labels == -1)[0] , size = Config.neg_samples_rcnn)


        batch_samples = np.concatenate((pos_samples, neg_samples))
        
        #reduce proposals, gt_boxes, gt_classes,labels(pos or neg sample) to those chosen for training
        proposal_coords = tf.gather(proposal_coords, batch_samples)
        gt_box_for_prop = gt_box_for_prop[batch_samples]
        gt_class_for_prop = gt_class_for_prop[batch_samples]


        #for each proposal: do a forward pass
        proposals = []
        for coord in proposal_coords:
            
            start_y = min(coord[0]//Config.sub_sampling,74)
            start_x = min(coord[1]//Config.sub_sampling,126)

            end_y = tf.cast(tf.math.ceil(coord[2]/Config.sub_sampling) , dtype =tf.int32)
            end_x = tf.cast(tf.math.ceil(coord[3]/Config.sub_sampling) ,dtype =tf.int32)
            
            proposals.append(tf.squeeze(tf.slice(feature_map, begin=[0,start_y,start_x,0] , size=[1,max(end_y-start_y , 2) , max(end_x - start_x,2),-1 ]) ,axis=0 ) )
            
        pred_reg, pred_cls = self.forward_pass(proposals)
        
        #convert to shape: batch_size, classes, 4
        pred_reg = tf.reshape(pred_reg, [tf.shape(pred_reg)[0] , -1, 4] )

        return self.loss(gt_box_for_prop,gt_class_for_prop, proposal_coords, pred_reg, pred_cls)

    def loss(self, gt_boxes, gt_classes, proposal_coords, pred_reg, pred_cls):
        l1 = self.loss_cls(pred_cls, gt_classes)
        l2 = self.loss_reg(gt_boxes, gt_classes, proposal_coords, pred_reg)

        lamda = 10
        return l1 + lamda * l2

    def loss_cls(self, pred_cls , gt_classes):
        gt_classes = tf.convert_to_tensor(gt_classes, dtype=tf.float32)
            
        pred_cls = tf.reshape(pred_cls, [-1,5])
        pred_cls = pred_cls + 1e-7

        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(gt_classes, pred_cls)

        
    def loss_reg(self, gt_boxes, gt_classes, proposal_coords, pred_reg):

        if(self.non_zero_label_count > 0):
            props_trained = np.where(gt_classes != 0)[0]
        
            gt_boxes = tf.convert_to_tensor(gt_boxes[props_trained] , dtype=tf.float32)
            gt_classes = tf.convert_to_tensor(gt_classes[props_trained] , dtype =tf.int32)
            
        
            pred_reg = tf.gather(pred_reg, props_trained)
        
            gt_classes = tf.cast(gt_classes, dtype=tf.int32)
            pred_reg_for_class = tf.convert_to_tensor([pred_reg[count,class_num-1,:] for count,class_num in enumerate(gt_classes)])

        
            proposal_coords = tf.gather(proposal_coords, props_trained)

            boxes_height = gt_boxes[:,2] - gt_boxes[:,0]
            boxes_width = gt_boxes[:,3] - gt_boxes[:,1]
            boxes_y = gt_boxes[:,0] + 0.5 * boxes_height
            boxes_x = gt_boxes[:,1] + 0.5 * boxes_width

            proposal_coords = tf.cast(proposal_coords ,dtype=tf.float32)
            props_height = proposal_coords[:,2] - proposal_coords[:,0]
            props_width = proposal_coords[:,3] - proposal_coords[:,1]
            props_y = proposal_coords[:,0] + 0.5 * props_height
            props_x = proposal_coords[:,1] + 0.5 * props_width

            t_x = (boxes_x - props_x)/props_width
            t_y = (boxes_y - props_y)/props_height
            t_h = tf.math.log(tf.clip_by_value(( boxes_height/props_height),1e-7,10))
            t_w = tf.math.log(tf.clip_by_value( (boxes_width/props_width ) ,1e-7,10))


            gt_reformed = tf.concat(tf.transpose([t_y , t_x , t_h , t_w]), axis=0)

            return tf.keras.losses.Huber()(gt_reformed, pred_reg_for_class)
        else:
            return 0.0

  
    def forward_pass(self, proposals):
        #roi pooling
        features = []
        for prop in proposals:
            prop = tf.expand_dims(tf.convert_to_tensor(prop) ,axis=0)
            features.append(self.adaptivePooling(prop))
        features = tf.stack(features)
        
        #flatten
        features = tf.reshape(features, [tf.shape(features)[0], -1])
        
        # forward pass through fully connected layers
        features = self.fc1(features)
        features = tf.nn.relu(features)
        
        features = self.fc2(features)
        features = tf.nn.relu(features)
        
        reg_locs = self.reg_fc(features)

        classes = self.class_fc(features)
        
        return (reg_locs, classes)

    def __call__(self, feature_map, proposal_coords):

        proposals = []
        for coord in proposal_coords:
            start_y = min(coord[0]//Config.sub_sampling, 74)
            start_x = min(coord[1]//Config.sub_sampling, 126)

            end_y = tf.cast(tf.math.ceil(coord[2]/Config.sub_sampling) , dtype =tf.int32)
            end_x = tf.cast(tf.math.ceil(coord[3]/Config.sub_sampling) ,dtype =tf.int32)

            
            proposals.append(tf.squeeze(tf.slice(feature_map, begin=[0,start_y,start_x,0] , size=[1,max(end_y-start_y , 2) , max(end_x - start_x,2),-1 ]) ,axis=0 ) )


        pred_reg, pred_cls = self.forward_pass(proposals)

        #get predicted class
        pred_cls = tf.reshape(pred_cls,[-1,5])
        most_probable_class = np.argmax(pred_cls,axis=1)
        
        print(most_probable_class)

        pred_cls = tf.nn.softmax(pred_cls)
        object_scores = tf.math.reduce_max(pred_cls[:,1:] , axis=1).numpy()
 
        non_background_props = np.where(most_probable_class !=0)[0]
    
        pred_reg = tf.gather(pred_reg, non_background_props)

        proposal_coords = np.array(proposal_coords)[non_background_props]
        object_scores = object_scores[non_background_props]
        most_probable_class =most_probable_class[non_background_props]
        
        pred_reg = tf.reshape(pred_reg, [tf.shape(pred_reg)[0] , -1, 4] )

        pred_reg_for_class = tf.convert_to_tensor([pred_reg[ind,val-1,:] for ind,val in enumerate(most_probable_class)])
       
        pred_reg = pred_reg_for_class
        
        proposal_coords = tf.cast(proposal_coords ,dtype=tf.float32)
        props_height = proposal_coords[:,2] - proposal_coords[:,0]
        props_width = proposal_coords[:,3] - proposal_coords[:,1]
        props_y = proposal_coords[:,0] + 0.5 * props_height
        props_x = proposal_coords[:,1] + 0.5 * props_width

        p_x = props_width * pred_reg[:,1] + props_x
        p_y = props_height * pred_reg[:,0] + props_y
        p_w = props_width * tf.clip_by_value(tf.math.exp(pred_reg[:,3]) , 0.001, 20)
        p_h = props_height* tf.clip_by_value(tf.math.exp(pred_reg[:,2]) , 0.001, 20)
       
        return (tf.transpose([p_y, p_x, p_h, p_w]) , object_scores[:])
    
    
