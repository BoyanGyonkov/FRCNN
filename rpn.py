import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/tools/cuda/bin")
import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import Config 
from utils import multiple_bb_iou

class RPN(keras.layers.Layer):
    def __init__(self):
        super().__init__(name="rpn")
        self.anchor_ratios = Config.anchor_ratios
        self.anchor_scales = Config.anchor_scales
        self.sub_sample = Config.sub_sampling
        self.input_shapes = Config.input_shape 

        self.anchors_per_point = len(self.anchor_ratios) * len(self.anchor_scales) 
        self.anchors = self.init_anchors()
        
        self.w1 = tf.Variable(tf.random.normal([5,5,512,512], stddev=0.01),dtype=tf.float32, trainable=True, name = "rpn_w1")
        self.w2 = tf.Variable(tf.random.normal([3,3,512,512], stddev=0.01),dtype=tf.float32, trainable=True, name = "rpn_w2")
        self.reg_filter = tf.Variable(tf.random.normal([1,1,512,self.anchors_per_point * 4], stddev=0.01),dtype=tf.float32, trainable=True, name = "rpn_reg")
        self.cls_filter = tf.Variable(tf.random.normal([1,1,512,self.anchors_per_point *2], stddev=0.01),dtype=tf.float32, trainable=True , name="rpn_cls")


    def train_on_img(self,feature_map, gt_boxes):
        #calculate intersection over union for all (anchor box, ground_truth box) combinations
        ious = multiple_bb_iou(self.anchors, gt_boxes)
        indices_max_iou = np.argmax(ious, axis =0)
        values_max_iou = ious[indices_max_iou, np.arange(ious.shape[1])]
       
        ind = np.argmax(values_max_iou)
        
        gt_box_for_anchor = np.array([gt_boxes[i] for i in indices_max_iou])

        pos_samples_per_batch = Config.pos_samples_rpn
        neg_samples_per_batch = Config.neg_samples_rpn

        labels = np.zeros(self.anchors.shape[0])
        labels[np.where(values_max_iou > 0.5)] = 1
        pos_samples_number = np.count_nonzero(labels)
        labels[np.where((values_max_iou < 0.1))] = -1
        #print("Pos_samples " , pos_samples_number)
        
        
        if pos_samples_per_batch > pos_samples_number:
            if(pos_samples_number !=0):
                pos_samples = np.random.choice(np.where(labels == 1)[0], size=(pos_samples_number))
                neg_samples = np.random.choice(np.where(labels == -1)[0], size=(neg_samples_per_batch + pos_samples_per_batch - pos_samples_number))
                
            else:
                pos_samples = np.random.choice(np.where(labels == -1)[0], size=(pos_samples_per_batch))
                neg_samples = np.random.choice(np.where(labels == -1)[0], size=(neg_samples_per_batch))

        else:
            pos_samples = np.random.choice(np.where(labels == 1)[0], size=(pos_samples_per_batch))
            neg_samples = np.random.choice(np.where(labels == -1)[0], size=(neg_samples_per_batch))


        batch_samples = np.concatenate((pos_samples, neg_samples))
        
        reg_locs, cls_scores = self.conv(feature_map)
        reg_locs = tf.reshape(reg_locs,[1,-1,tf.shape(reg_locs)[3]])
        reg_locs = tf.reshape(reg_locs, [1,tf.shape(reg_locs)[1], -1 , 4])
        reg_locs = tf.reshape(reg_locs, [1,-1,4])
        

        return self.loss(gt_box_for_anchor, labels, reg_locs, cls_scores, batch_samples)
    
    def loss(self,gt_boxes, gt_labels, reg_locs, cls_scores, batch_samples):
        l1 = self.loss_cls(cls_scores, batch_samples, gt_labels)
        l2 = self.loss_reg(gt_boxes, reg_locs, gt_labels)
        lmda = 1
        #print("RPN_l1 " ,l1)
        
        return l1 +l2
    
    def loss_cls(self, cls_scores, batch_samples, gt_labels):
        gt_labels = gt_labels[batch_samples]
        gt_labels[np.where(gt_labels == -1)] = 0
        gt_labels = gt_labels.astype('float32')

        cls_scores = tf.reshape(cls_scores, [-1,2])
        cls_scores = tf.gather(cls_scores, self.valid_anchors_index,axis=0)
 
   
        cls_scores = tf.gather(cls_scores, batch_samples, axis=0)
        cls_scores = cls_scores + 1e-7

        
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(gt_labels, tf.nn.softmax(cls_scores))

    def loss_reg(self, gt_boxes, reg_locs, gt_labels):
     
        anchors_trained = np.where(gt_labels == 1)[0]
        gt_boxes = tf.convert_to_tensor(gt_boxes[anchors_trained], dtype=tf.float32)

        reg_locs = tf.reshape(reg_locs,[-1,4])
        reg_locs = tf.gather(reg_locs,self.valid_anchors_index , axis=0)
        reg_locs = tf.gather(reg_locs, anchors_trained,axis=0)
        reg_locs = tf.reshape(reg_locs,[-1,4])
        anchors_used = tf.convert_to_tensor(self.anchors[anchors_trained] , dtype=tf.float32)

        # Convert coordinates to: center x,y , height , width
        boxes_height = gt_boxes[:,2] -gt_boxes[:,0]
        boxes_width = gt_boxes[:,3] - gt_boxes[:,1]
        boxes_y = gt_boxes[:,0] + 0.5 * boxes_height
        boxes_x = gt_boxes[:,1] + 0.5 * boxes_width

        anchors_height = anchors_used[:,2] -anchors_used[:,0]
        anchors_width = anchors_used[:,3] - anchors_used[:,1]
        anchors_y = anchors_used[:,0] + 0.5 * anchors_height
        anchors_x = anchors_used[:,1] + 0.5 * anchors_width

        t_x = (boxes_x - anchors_x)/anchors_width
        t_y = (boxes_y - anchors_y)/anchors_height
        t_h = tf.math.log(tf.clip_by_value(( boxes_height/anchors_height),1e-7,10))
        t_w = tf.math.log(tf.clip_by_value( (boxes_width/anchors_width ) ,1e-7,10))
        

        gt_reformed = tf.transpose([t_y , t_x , t_h , t_w])
       
        return tf.keras.losses.Huber()(gt_reformed, reg_locs)

 
    def conv(self, feature_map):
        conv1 = tf.nn.conv2d(feature_map, self.w1, [1,1,1,1],padding='SAME')
        conv2 = tf.nn.conv2d(conv1, self.w2, [1,1,1,1],padding='SAME')
        reg_locs = tf.nn.conv2d(conv2, self.reg_filter, [1,1,1,1], padding= 'SAME')
        cls_scores = tf.nn.conv2d(conv2, self.cls_filter, [1,1,1,1] , padding = 'SAME')
        
        return (reg_locs, cls_scores) 


    def __call__(self, feature_map):
        reg_locs,cls_scores = self.conv(feature_map)

        reg_locs = tf.reshape(reg_locs,[-1,4])
        reg_locs = tf.gather(reg_locs,self.valid_anchors_index)
        reg_locs = tf.reshape(reg_locs, [-1,4])

        cls_scores = tf.reshape(cls_scores,[-1,2])
        cls_scores = tf.gather(cls_scores, self.valid_anchors_index)
        cls_scores = tf.reshape(cls_scores,[-1,2])

        anchors_height = self.anchors[:,2] - self.anchors[:,0]
        anchors_width = self.anchors[:,3] - self.anchors[:,1]
        anchors_y = self.anchors[:,0] + 0.5 * anchors_height
        anchors_x = self.anchors[:,1] + 0.5 * anchors_width

 
        #p_x = anchors_width * reg_locs[:,1] + anchors_x
        #p_y = anchors_height * reg_locs[:,0] + anchors_y
        #p_w = anchors_width * tf.math.exp(reg_locs[:,3])
        #p_h = anchors_height * tf.math.exp(reg_locs[:,2])

        p_x = anchors_x
        p_y = anchors_y
        p_w = anchors_width
        p_h = anchors_height

        #Output in format: predicted: y,x,height,width
        return (tf.concat(tf.transpose([p_y , p_x , p_h , p_w]), axis=0),tf.nn.softmax(cls_scores)[:,1])
        
    def init_anchors(self): 

        number_of_feature_points = (self.input_shapes[0]//self.sub_sample)* (self.input_shapes[1]//self.sub_sample)
        
        anchors = np.zeros(( number_of_feature_points ,self.anchors_per_point, 4), dtype=np.float32)
        anchor_base = np.zeros((self.anchors_per_point, 4), dtype=np.float32)
        
        ctr_x = self.sub_sample / 2.
        ctr_y = self.sub_sample / 2.

        for i in range(len(self.anchor_ratios)):
            for j in range(len(self.anchor_scales)):
                height = self.sub_sample * self.anchor_scales[j] * np.sqrt(self.anchor_ratios[i])
                width = self.sub_sample * self.anchor_scales[j] * np.sqrt(1./self.anchor_ratios[i])

                index = i* len(self.anchor_scales) + j

                anchor_base[index, 0] = ctr_y - height/2.
                anchor_base[index, 1] = ctr_x - width/2.
                anchor_base[index, 2] = ctr_y + height/2.
                anchor_base[index, 3] = ctr_x + width/2.

        anchors = np.broadcast_to(anchor_base,(self.input_shapes[0]//self.sub_sample+1, self.input_shapes[1]//self.sub_sample+1, anchor_base.shape[0], anchor_base.shape[1])).copy()
        for i in range(anchors.shape[0]):
            anchors[i,:,:,0] += self.sub_sample * (i)
            anchors[i,:,:,2] += self.sub_sample * (i)
            
        for j in range(anchors.shape[1]):
            anchors[:,j,:,1] += self.sub_sample * (j)
            anchors[:,j,:,3] += self.sub_sample * (j)

        anchors = np.reshape(anchors, (-1, 4))
        
        return self.get_valid_anchors(anchors)

    def get_valid_anchors(self, anchors):
        self.valid_anchors_index = np.where(
            (anchors[:,0] >= 0) &
            (anchors[:,1] >= 0) &
            (anchors[:,2] <= self.input_shapes[0]) &
            (anchors[:,3] <= self.input_shapes[1])
        )[0]
        
        return anchors[self.valid_anchors_index]
