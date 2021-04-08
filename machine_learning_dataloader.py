import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from machine_learning_select_features import select_features
import util
from machine_learning_extract_features import (cal_img_features_all,
                                               extract_feature_marina,
                                               extract_feature_matin,
                                               extract_feature_sift_kmeans,
                                               extract_feature_dong)

train_data_dir = '/home/zhangyangsong/IMA205/Train/Train/'
train_gt_dir = '/home/zhangyangsong/IMA205/metadataTrain.csv'
test_data_dir = '/home/zhangyangsong/IMA205/Test/Test/'

random.seed(0)


"""
read training data

binary: True or False means binary classification or multiclassification

extract_feature: can be None(use every pixels as features), "martin", "marina" or "sift_kmeans". 
the details of "martin", "marina" and "sift_kmeans" feature extraction methodes are in the files machine_learning_extract_features.py

mask_mode: True or False. this means whether to use the masks or not
For the marin and marina methods, it's necessary to use masks.

sift_orb: "sift" or "orb" for the method sift_kmeans

num_clusters: used only for the method sift_kmeans
"""
def read_train_data(img_size = 256, label_name = "ABNORMAL", extract_feature = None, mask_mode = True, sift_orb = "sift", num_clusters = 60) :
    print("read training data")
    imgs = []
    gts = []
    
    gt_data = pd.read_csv(train_gt_dir)

    for idx, img_name in enumerate(gt_data["ID"]):
        gts.append(gt_data[label_name][idx])

        img_path = train_data_dir + str(img_name) + '.bmp'
        img = cv2.imread(img_path)

        if extract_feature != "sift_kmeans":
            img = cv2.resize(img, (img_size, img_size))

        if mask_mode == True:
            seg1_path = train_data_dir + str(img_name) + '_segCyt.bmp'
            seg2_path = train_data_dir + str(img_name) + '_segNuc.bmp'
        
            img_seg1 = cv2.imread(seg1_path)
            img_seg2 = cv2.imread(seg2_path)

            if extract_feature != "sift_kmeans":
                img_seg1 = cv2.resize(img_seg1, (img_size, img_size))
                img_seg2 = cv2.resize(img_seg2, (img_size, img_size))
        
        # First mode: use every pixels as features
        if extract_feature == None:
            # if we use masks, we focus on the cell
            if mask_mode == True:
                img = img * ((img_seg1 > 0) + (img_seg2 > 0))
            img = img.astype(np.uint8)
            imgs.append(img.reshape(-1))

        # Second mode: use martin method to extract features
        elif extract_feature == "martin":
            feature = extract_feature_matin(img, img_seg1, img_seg2)
            imgs.append(feature)

        # Third mode: use marina method to extract features
        elif extract_feature == "marina":
            feature1 = extract_feature_marina(img, img_seg1)
            feature2 = extract_feature_marina(img, img_seg2)
            feature = np.hstack((feature1, feature2))
            imgs.append(feature)

        # Forth mode: use sift_kmeans to extract features
        elif extract_feature == "sift_kmeans":
            img = sharpening(img)
            img = augment_contrast(img)
            # if we use masks, we focus on the cell
            if mask_mode == True:
                img = img * ((img_seg1 > 0) + (img_seg2 > 0)) 
                img = img.astype(np.uint8)
            imgs.append(img)
        
        # fifth mode: use dong method to extract features
        elif extract_feature == "dong":
            feature = extract_feature_dong(img, img_seg1, img_seg2)
            imgs.append(feature)

        # sixth mode: combine martin, marina and dong method to extract features
        elif extract_feature == "martin_marina_dong":
            feature1 = extract_feature_matin(img, img_seg1, img_seg2)
            feature2 = extract_feature_marina(img, img_seg1)
            feature3 = extract_feature_marina(img, img_seg2)
            feature4 = extract_feature_dong(img, img_seg1, img_seg2)
            feature = np.hstack((feature1, feature2, feature3, feature4)) 
            imgs.append(feature)

    imgs = np.array(imgs)
    gts = np.array(gts)
    gts = gts.reshape(-1,1)

    # Forth mode: use sift_kmeans to extract features
    code_book = None
    if extract_feature == "sift_kmeans":
        imgs, code_book = extract_feature_sift_kmeans(imgs, sift_orb, num_clusters)
    
    print("training data shape: " + str(imgs.shape))
    return imgs, gts, code_book     




"""
read test data

extract_feature: can be None(use every pixels as features), "martin", "marina" or "sift_kmeans". 
the details of "martin", "marina" and "sift_kmeans" feature extraction methodes are in the files machine_learning_extract_features.py

mask_mode: True or False. this means whether to use the masks or not
For the marin and marina methods, it's necessary to use masks.

code_book: used only for the method sift_kmeans to extract features

sift_orb: "sift" or "orb" for the method sift_kmeans
"""
def read_test_data(img_size = 256, extract_feature = None, mask_mode = True, code_book = None, sift_orb = "sift") :
    print("\nread test data")
    imgs = []

    img_names = util.io.ls(test_data_dir, '.bmp')

    imgs = []
    imgs_path = []
    for idx, img_name in enumerate(img_names):
        name, _ = os.path.splitext(img_name)
        if (name.isdigit()) :
            img_path = test_data_dir + name+'.bmp'
            imgs_path.append(img_path)
            
            img = cv2.imread(img_path)

            if extract_feature != "sift_kmeans":
                img = cv2.resize(img, (img_size, img_size))

            if mask_mode == True:
                seg1_path = test_data_dir + name + '_segCyt.bmp'
                seg2_path = test_data_dir + name + '_segNuc.bmp'
            
                img_seg1 = cv2.imread(seg1_path)
                img_seg2 = cv2.imread(seg2_path)

                if extract_feature != "sift_kmeans":
                    img_seg1 = cv2.resize(img_seg1, (img_size, img_size))
                    img_seg2 = cv2.resize(img_seg2, (img_size, img_size))
            
            # First mode: use every pixels as features
            if extract_feature == None:
                
                # if we use masks, we focus on the cell
                if mask_mode == True:
                    img = img * ((img_seg1 > 0) + (img_seg2 > 0))
                img = img.astype(np.uint8)
                imgs.append(img.reshape(-1))

            # Second mode: use martin method to extract features
            elif extract_feature == "martin":
                feature = extract_feature_matin(img, img_seg1, img_seg2)
                imgs.append(feature)

            # Third mode: use marina method to extract features
            elif extract_feature == "marina":
                feature1 = extract_feature_marina(img, img_seg1)
                feature2 = extract_feature_marina(img, img_seg2)
                feature = np.hstack((feature1, feature2))
                imgs.append(feature)

            # Forth mode: use sift_kmeans to extract features
            elif extract_feature == "sift_kmeans":
                img = sharpening(img)
                img = augment_contrast(img)
                # if we use masks, we focus on the cell
                if mask_mode == True:
                    img = img * ((img_seg1 > 0) + (img_seg2 > 0)) 
                    img = img.astype(np.uint8)
                imgs.append(img)
            
            # fifth mode: use dong method to extract features
            elif extract_feature == "dong":
                feature = extract_feature_dong(img, img_seg1, img_seg2)
                imgs.append(feature)

            # sixth mode: combine martin, marina and dong method to extract features
            elif extract_feature == "martin_marina_dong":
                feature1 = extract_feature_matin(img, img_seg1, img_seg2)
                feature2 = extract_feature_marina(img, img_seg1)
                feature3 = extract_feature_marina(img, img_seg2)
                feature4 = extract_feature_dong(img, img_seg1, img_seg2)
                feature = np.hstack((feature1, feature2, feature3, feature4))
                imgs.append(feature)

    imgs = np.array(imgs)

    # Forth mode: use sift_kmeans to extract features
    if extract_feature == "sift_kmeans":
        imgs = cal_img_features_all(imgs, code_book, sift_orb)

    print("test data shape: " + str(imgs.shape))
    return imgs, imgs_path




# pre-processing
# select_feature can be {None, "pca", "kpca", "spca", "select_best", "RF", "ExtraTrees", "shap", "RFECV", "SFS", "permutation"}
def pre_process(X_train, X_test, y_train, select_feature = None, n_component = 20) :
    print("\npre processing")

    print("scaling and shuffling")
    # Scale data (each feature will have average equal to 0 and unit variance)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    # shuffle
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices, :]
    y_train = y_train[indices, :]

    X_train, X_test = select_features(X_train, X_test, y_train, select_feature = select_feature, n_component = n_component)

    print("After pre processing, training data shape: " + str(X_train.shape))
    print("After pre processing, test data shape: " + str(X_test.shape))
    return X_train, X_test, y_train




# write the final results
def write_data(y_pre, label_name, data_path, save_path):
    # write results
    print("write results")
    util.io.write_lines(save_path, "ID"+","+label_name+'\n', 'w')
    for idx, result in enumerate(y_pre):
        image_name = data_path[idx].split('/')[-1].split('.')[0]
        util.io.write_lines(save_path, image_name+","+str(result)+'\n', 'a')


###################################################################################################################
# The next 2 functions are only used for the SIFT method

# Contrast augmentation. For the SIFT method, we have to augment the contrast, else it cannot find the key points
def augment_contrast(img):
    max_pixel = np.max(img, axis=(0, 1))
    min_pixel = np.min(img, axis=(0, 1))

    img = (img - min_pixel) / (max_pixel - min_pixel) * 255

    img = img.astype('uint8')
    return img

# Image sharpening, for the SIFT method, else it cannot find the key points
def sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst
