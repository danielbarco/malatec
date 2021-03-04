# Nuclei Segmentation - UNet using Tensorflow 2
# Intro
# - Dataset used is from NIH ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/NIH-NLM-ThinBloodSmearsPf/
# - The architecture used is [U-Net](https://arxiv.org/abs/1505.04597), which is very common for image segmentation problems such as this.
# - This notebook is inspired from the great kernel [Keras U-net starter - LB 0.277](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277) by Kjetil Åmdal-Sævik.

import os, sys
from os.path import isfile, join
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import cv2


def get_paths(PATH):
    '''Get paths of images png, jpg or jpeg'''
    list_paths_img = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if isfile(join(root, file)) and file_extension in [".png",'.jpg','.jpeg'] and not filename.startswith("._"):
                    list_paths_img.append(join(root,file))
    return list_paths_img


def get_imgs(PATH, IMG_HEIGHT = 256, IMG_WIDTH = 256, IMG_CHANNELS = 3):
    '''get images and resizes them
    PATH = folder holding images
    IMG_HEIGHT = 256 (default)
    IMG_WIDTH = 256 (default) 
    IMG_CHANNELS = 3 (default)
    '''
    list_paths_img = get_paths(PATH)
    imgs = np.zeros((len(list_paths_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    # Get and resize images
    imgs = np.zeros((len(list_paths_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_imgs = []
    for idx, image_path in enumerate(list_paths_img):
        #Read images iteratively
        img = imread(image_path)[:,:,:IMG_CHANNELS]
        #Get test size
        sizes_imgs.append([img.shape[0], img.shape[1]])
        
        #Resize image to match training data
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        
        #Append image to numpy array for test dataset
        imgs[idx] = img
    return imgs, sizes_imgs    

def slice_img(INPUT_IMG, OUTPUT_DIR, RESIZE_FACTOR = 1,
                  SLICE_HEIGHT = 256, SLICE_WIDTH = 256,
                  ZERO_FRAC_TRESH = 0, OVERLAP = 0, PAD = 0, VERBOSE = False,
                 ):
    '''
    slices image into smaller windows
    INPUT_IMG,
    OUTPUT_DIR, 
    RESIZE_FACTOR = 1,
    SLICE_HEIGHT = 256, 
    SLICE_WIDTH = 256,
    ZERO_FRAC_TRESH = 0, 
    OVERLAP = 0, 
    PAD = 0, 
    VERBOSE = False,
    '''
    #adapted from https://github.com/CosmiQ/simrdwn/blob/9f91eac5d0769400f89eefc145d67f0ee209d8fc/simrdwn/core/slice_im.py

    img = cv2.imread(INPUT_IMG, 1)  # color
    resized_img = cv2.resize(img, (int(round(img.shape[1] * RESIZE_FACTOR)), int(round(img.shape[0] * RESIZE_FACTOR))))
    im_h, im_w = resized_img.shape[:2]
    win_size = SLICE_HEIGHT * SLICE_WIDTH

    # if slice sizes are large than image, pad the edges
    if SLICE_HEIGHT > im_h:
        pad = SLICE_HEIGHT - im_h
    if SLICE_WIDTH > im_w:
        pad = max(pad, SLICE_WIDTH - im_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        resized_img = cv2.copyMakeBorder(resized_img, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=border_color)

    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - OVERLAP) * SLICE_WIDTH)
    dy = int((1. - OVERLAP) * SLICE_HEIGHT)
    if VERBOSE:
        print('dx', dx)
        print('dy', dy)


    for y in range(0, im_h, dy):  # sliceHeight):
        for x in range(0, im_w, dx):  # sliceWidth):
            n_ims += 1
            # extract image
            # make sure we don't go past the edge of the image
            if y + SLICE_HEIGHT > im_h:
                y0 = im_h - SLICE_HEIGHT
            else:
                y0 = y
            if x + SLICE_WIDTH > im_w:
                x0 = im_w - SLICE_WIDTH
            else:
                x0 = x

            window_c = resized_img[y0:y0 + SLICE_WIDTH, x0:x0 + SLICE_WIDTH]
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
            if zero_frac >= ZERO_FRAC_TRESH:
                if VERBOSE:
                    print("Zero frac too high at:", zero_frac)
                continue
                
            #  save
            outname_im = os.path.join(OUTPUT_DIR, outname_part + '.png')
         
            # save yolt ims
            if VERBOSE:
                print("image output:", outname_im)
            cv2.imwrite(outname_im, window_c)


    if VERBOSE:
        print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull,
              "sliceHeight", SLICE_HEIGHT, "sliceWidth", SLICE_WIDTH)
        print("Time to slice", input_im, time.time()-t0, "seconds")

    return dict_yolo, dict_bbx, max_annot

def unet_predict(MODEL, IMGS, SIZES_IMGS):
    ''' returns a unet predicted masks
    MODEL = tensorflow2 model checkpoint .h5
    IMGS = a numpy list list of images
    SIZES_IMGS = a list with all image sizes'''

    # Predict 
    model = load_model(MODEL)
    pred_masks = model.predict(IMGS, verbose=1)

    # Threshold predictions (if prediction larger than 0.5 indicates cell)
    pred_masks_t = (pred_masks > 0.5).astype(np.uint8)

    # # Create list of upsampled test masks
    # preds_test_upsampled = []
    # for i in range(len(pred_masks_t)):
    #     preds_test_upsampled.append(resize(np.squeeze(pred_masks_t[i]), 
    #                                     (SIZES_IMGS[i][0], SIZES_IMGS[i][1]), 
    #                                     mode='constant', preserve_range=True))
    return pred_masks_t