class Config:
    #base net architecture
    base_net = "VGG16"
    #pixel_values : feature_map_points ratio
    sub_sampling =16
    #input size of image
    input_shape = (1216,2048,3)
  

    anchor_ratios = [1]
    anchor_scales = [2,3,4,5,6,8,10,13]

    #general training params
    learning_rate = 0.00001
    num_epochs = 40

    # hyperparameters for non maximum supperssion
    nms_thresh = 0.4
    train_n_nms = 2000
    test_n_nms = 300
    prop_min_size = 32 #minimum size of a box

    #hyperparameters for training RPN
    pos_samples_rpn = 20
    neg_samples_rpn = 10

    #hyperparameters for training Fast R-CNN
    pos_samples_rcnn = 5
    neg_samples_rcnn = 10
    positive_iou_threshold = 0.45
    negative_iou_threshold = 0.01
    
    
