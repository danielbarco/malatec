# Nuclei Segmentation - UNet using Tensorflow 2
# Intro
# - Dataset used is from NIH ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/NIH-NLM-ThinBloodSmearsPf/
# - The architecture used is [U-Net](https://arxiv.org/abs/1505.04597), which is very common for image segmentation problems such as this.
# - This notebook is inspired from the great kernel [Keras U-net starter - LB 0.277](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277) by Kjetil Åmdal-Sævik.

import os, sys
from os.path import isfile, join, basename
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, concatenate, Dropout, LeakyReLU, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
import cv2  
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Custom Keras layer

class SpaceToDepth(keras.layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                             reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
        return t

    def compute_output_shape(self, input_shape):
        shape =  (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                  input_shape[3] * self.block_size **2)
        return tf.TensorShape(shape)
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'block_size': self.block_size,
        })
        return config

def get_paths(img_path):
    '''Get paths of images png, jpg or jpeg'''
    list_paths_img = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if isfile(join(root, file)) and file_extension in [".png",'.jpg','.jpeg'] and not filename.startswith("._"):
                    list_paths_img.append(join(root,file))
    return list_paths_img


def get_imgs(img_path, img_height = 256, img_width = 256, img_channels = 3):
    '''get images and resizes them
    img_path = folder holding images
    img_height = 256 (default)
    img_width = 256 (default) 
    img_channels = 3 (default)
    '''
    list_paths_img = get_paths(img_path)
    imgs = np.zeros((len(list_paths_img), img_height, img_width, img_channels), dtype=np.uint8)
    # Get and resize images
    sizes_imgs = []
    for idx, image_path in enumerate(list_paths_img):
        #Read images iteratively
        img = imread(image_path)[:,:,:img_channels]

        #Resize image to match training data
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        
        #Get test size
        sizes_imgs.append([img.shape[0], img.shape[1]])
        
        #Append image to numpy array for test dataset
        imgs[idx] = img
    return imgs, sizes_imgs    

def slice_all_imgs(input_dir, output_dir, resize_factor = 1,
                  slice_height = 256, slice_width = 256,
                  zero_frac_tresh = 0.8, overlap = 0, pad = 0, verbose = False):
    '''slices all images into smaller windows    
    input_img,
    output_dir, 
    resize_factor = 1,
    slice_height = 256, 
    slice_width = 256,
    zero_frac_tresh = 0, 
    overlap = 0, 
    pad = 0, 
    verbose = False,
    '''
    list_paths_img = get_paths(PATH)
    for image in list_paths_img:
        slice_img(image, output_dir, resize_factor = resize_factor,
                    slice_height = slice_height, slice_width = slice_width,
                    zero_frac_tresh = zero_frac_tresh, overlap = overlap, pad = pad, verbose = verbose)


def slice_img(input_img, output_dir, resize_factor = 1,
                  slice_height = 256, slice_width = 256,
                  zero_frac_tresh = 0, overlap = 0, pad = 0, verbose = False):
    '''
    slices image into smaller windows
    input_img,
    output_dir, 
    resize_factor = 1,
    slice_height = 256, 
    slice_width = 256,
    zero_frac_tresh = 0, 
    overlap = 0, 
    pad = 0, 
    verbose = False,
    '''
    #adapted from https://github.com/CosmiQ/simrdwn/blob/9f91eac5d0769400f89eefc145d67f0ee209d8fc/simrdwn/core/slice_im.py

    img = cv2.imread(input_img, 1)  # color
    resized_img = cv2.resize(img, (int(round(img.shape[1] * resize_factor)), int(round(img.shape[0] * resize_factor))))
    im_h, im_w = resized_img.shape[:2]
    win_size = slice_height * slice_width
    filename = os.path.basename(input_img)

    try:
        os.makedirs(output_dir)   
        if verbose:
            print("Directory " , output_dir ,  " Created ")
    except FileExistsError:
        if verbose:
           print("Directory " , output_dir ,  " already exists")  

    # if slice sizes are large than image, pad the edges
    if slice_height > im_h:
        pad = slice_height - im_h
    if slice_width > im_w:
        pad = max(pad, slice_width - im_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        resized_img = cv2.copyMakeBorder(resized_img, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=border_color)

    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * slice_width)
    dy = int((1. - overlap) * slice_height)
    if verbose:
        print('dx', dx)
        print('dy', dy)


    for y in range(0, im_h, dy):  # sliceHeight):
        for x in range(0, im_w, dx):  # sliceWidth):
            n_ims += 1
            # extract image
            # make sure we don't go past the edge of the image
            if y + slice_height > im_h:
                y0 = im_h - slice_height
            else:
                y0 = y
            if x + slice_width > im_w:
                x0 = im_w - slice_width
            else:
                x0 = x

            window_c = resized_img[y0:y0 + slice_width, x0:x0 + slice_width]
            win_h, win_w = window_c.shape[:2]
    
            outname_part = 'slice_' + filename + \
            '_' + str(y0) + '_' + str(x0) + \
            '_' + str(win_h) + '_' + str(win_w) + \
            '_' + str(pad)

            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

            # find threshold of image that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            # print ("zero_frac", zero_fra
            # skip if image is mostly empty
            if zero_frac >= zero_frac_tresh:
                if verbose:
                    print("Zero frac too high at:", zero_frac)
                continue
                
            #  save
            outname_im = os.path.join(output_dir, outname_part + '.png')
         
            # save yolt ims
            if verbose:
                print("image output:", outname_im)
            cv2.imwrite(outname_im, window_c)


    if verbose:
        print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull,
              "sliceHeight", slice_height, "sliceWidth", slice_width)


def unet_predict(unet_model, imgs, sizes_imgs):
    ''' returns a unet predicted masks
    unet_model = tensorflow2 model checkpoint .h5
    imgs = a numpy list list of images
    sizes_imgs = a list with all image sizes'''

    # Predict 
    unet_model = load_model(unet_model)
    pred_masks = unet_model.predict(imgs, verbose=1)

    # Threshold predictions (if prediction larger than 0.5 indicates cell)
    pred_masks_t = (pred_masks > 0.5).astype(np.uint8)

    return pred_masks_t



def display_yolo(image, yolo_model, score_threshold, iou_threshold,\
                 train_batch_size =16, grid_h =8, grid_w =8, image_h =256, image_w =256, anchors =[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],\
                 plot = False):
    '''
    Display predictions from YOLO model.

    Parameters
    ----------
    - file : string list : list of images path.
    - yolo_model : YOLO model.
    - score_threshold : threshold used for filtering predicted bounding boxes.
    - iou_threshold : threshold used for non max suppression.
    '''

    # load image
    
    input_image = image[:,:,::-1]
    input_image = image / 255.
    input_image = np.expand_dims(input_image, 0)

    # prediction
    y_pred = yolo_model.predict_on_batch(input_image)

    # post prediction process
    # grid coords tensor
    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0,2,1,3,4))
    coords = tf.tile(tf.concat([coord_x,coord_y], -1), [train_batch_size, 1, 1, 5, 1])
    dims = K.cast_to_floatx(K.int_shape(y_pred)[1:3])
    dims = K.reshape(dims,(1,1,1,1,2))
    # anchors tensor
    anchors = np.array(anchors)
    anchors = anchors.reshape(len(anchors) // 2, 2)
    # pred_xy and pred_wh shape (m, grid_w, grid_h, Anchors, 2)
    pred_xy = K.sigmoid(y_pred[:,:,:,:,0:2])
    pred_xy = (pred_xy + coords)
    pred_xy = pred_xy / dims
    pred_wh = K.exp(y_pred[:,:,:,:,2:4])
    pred_wh = (pred_wh * anchors)
    pred_wh = pred_wh / dims
    # pred_confidence
    box_conf = K.sigmoid(y_pred[:,:,:,:,4:5])  
    # pred_class
    box_class_prob = K.softmax(y_pred[:,:,:,:,5:])

    # Reshape
    pred_xy = pred_xy[0,...]
    pred_wh = pred_wh[0,...]
    box_conf = box_conf[0,...]
    box_class_prob = box_class_prob[0,...]

    # Convert box coords from x,y,w,h to x1,y1,x2,y2
    box_xy1 = pred_xy - 0.5 * pred_wh
    box_xy2 = pred_xy + 0.5 * pred_wh
    boxes = K.concatenate((box_xy1, box_xy2), axis=-1)

    # Filter boxes
    box_scores = box_conf * box_class_prob
    box_classes = K.argmax(box_scores, axis=-1) # best score index
    box_class_scores = K.max(box_scores, axis=-1) # best score
    prediction_mask = box_class_scores >= score_threshold
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    # Scale box to image shape
    boxes = boxes * image_h

    # Non Max Supression
    selected_idx = tf.image.non_max_suppression(boxes, scores, 50, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, selected_idx)
    scores = K.gather(scores, selected_idx)
    classes = K.gather(classes, selected_idx)
    
    if plot:
        # Draw image
        plt.figure(figsize=(2,2))
        f, (ax1) = plt.subplots(1,1, figsize=(10, 10))
        ax1.imshow(image[:,:,::-1])
        count_detected = boxes.shape[0]
        ax1.set_title('Detected objects count : {}'.format(count_detected))
        for i in range(count_detected):
            box = boxes[i,...]
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            classe = classes[i].numpy()
            if classe == 0:
                color = (0, 1, 0)
            else:
                color = (1, 0, 0)
            rect = patches.Rectangle((x.numpy(), y.numpy()), w.numpy(), h.numpy(), linewidth = 3, edgecolor=color,facecolor='none')
            ax1.add_patch(rect)
            
    return boxes, scores, classes