import datetime
import math
import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import vq
from scipy.stats import kurtosis, moment, skew
from skimage.exposure import histogram
from skimage.measure import label, regionprops
from sklearn import preprocessing, svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold  # for K-fold cross validation
from sklearn.model_selection import cross_val_predict  # prediction
from sklearn.model_selection import cross_val_score  # score evaluation
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler, scale
from sklearn.utils import shuffle
from skimage.feature import greycomatrix, greycoprops
from skimage import color, img_as_ubyte
import util

random.seed(0)
###################################################################################################################
#####################################   First method to extract features ##########################################
###################################################################################################################

# Idea to extrait features is from the appendix in this paper
# https://www.researchgate.net/publication/265873515_Pap-smear_Benchmark_Data_For_Pattern_Classification

# 20 features:
# Column    Feature                         Name
# B         Nucleus area                    Narea
# C         Cytoplasm area                  Carea
# D         N/C ratio                       N/C
# E         Nucleus brightness              Ncol
# F         Cytoplasm brightness            Ccol
# G         Nucleus shortest diameter       Nshort
# H         Nucleus longest diameter        Nlong
# I         Nucleus elongation              Nelong
# J         Nucleus roundness               Nround
# K         Cytoplasm shortest diameter     Cshort
# L         Cytoplasm longest diameter      Clong
# M         Cytoplasm elongation            Celong
# N         Cytoplasm roundness             Cround
# O         Nucleus perimeter               Nperim
# P         Cytoplasm perimeter             Cperim
# Q         Nucleus position                Npos
# R         Maxima in nucleus               Nmax
# S         Minima in nucleus               Nmin
# T         Maxima in cytoplasm             Cmax
# U         Minima in cytoplasm             Cmin

def extract_feature_matin(img, seg_cyt, seg_nuc): 
    feature = np.zeros((20))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find contours which is very useful later
    mask_nuc = cv2.cvtColor(seg_nuc,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask_nuc,127,255,cv2.THRESH_BINARY)
    _, contours_nuc, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours_nuc) == 0) : contours_nuc = []
    elif (len(contours_nuc) == 1) : contours_nuc = contours_nuc[0]
    else : 
        max_contour = 0
        max_area = 0
        for contour in contours_nuc:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        contours_nuc = max_contour
    
    mask_cyt = cv2.cvtColor(seg_cyt,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask_cyt,127,255,cv2.THRESH_BINARY)
    _, contours_cyt, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours_cyt) == 1) : contours_cyt = contours_cyt[0]
    else : 
        max_contour = 0
        max_area = 0
        for contour in contours_cyt:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        contours_cyt = max_contour

    # B         Nucleus area                    Narea
    if len(contours_nuc) == 0:
        feature[0] = 0
    else : feature[0] = cv2.contourArea(contours_nuc)

    # C         Cytoplasm area                  Carea
    feature[1] = cv2.contourArea(contours_cyt) - feature[0]

    # D         N/C ratio                       N/C   Size of nucleus relative to cell size
    feature[2] = feature[0] / (feature[0] + feature[1])

    # E         Nucleus brightness              Ncol
    if len(contours_nuc) == 0:
        feature[3] = 0
    else : 
        img_mean = cv2.mean(img,mask = mask_nuc)
        feature[3] = 0.299 * img_mean[0] + 0.587 * img_mean[1] + 0.114 * img_mean[2]

    # F         Cytoplasm brightness            Ccol
    img_mean = cv2.mean(img,mask = mask_cyt)
    feature[4] = 0.299 * img_mean[0] + 0.587 * img_mean[1] + 0.114 * img_mean[2]

    # G         Nucleus shortest diameter       Nshort
    if len(contours_nuc) == 0:
        feature[5:9] = 0
    else : 
        (x,y),(MA,ma),angle = cv2.fitEllipse(contours_nuc)
        feature[5] = ma

        # H         Nucleus longest diameter        Nlong
        feature[6] = MA

        # I         Nucleus elongation              Nelong
        feature[7] = feature[5] / feature[6]

        # J         Nucleus roundness               Nround
        feature[8] = feature[0] / (math.pi / 4 * feature[5]**2)

    # K         Cytoplasm shortest diameter     Cshort
    (x,y),(MA,ma),angle = cv2.fitEllipse(contours_cyt)
    feature[9] = ma

    # L         Cytoplasm longest diameter      Clong
    feature[10] = MA

    # M         Cytoplasm elongation            Celong
    feature[11] = feature[9] / feature[10]

    # N         Cytoplasm roundness             Cround
    feature[12] = feature[1] / (math.pi / 4 * feature[10]**2)

    # O         Nucleus perimeter               Nperim
    if len(contours_nuc) == 0:
        feature[13] = 0
    else: feature[13] = cv2.arcLength(contours_nuc,True)

    # P         Cytoplasm perimeter             Cperim
    feature[14] = cv2.arcLength(contours_cyt,True)

    # Q         Nucleus position                Npos
    # first calculate the center of this contour
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if len(contours_nuc) == 0:
        feature[15:18] = 0
    else : 
        Mn = cv2.moments(contours_nuc)
        xn = int(Mn['m10']/Mn['m00'])
        yn = int(Mn['m01']/Mn['m00'])

        Mc = cv2.moments(contours_cyt)
        xc = int(Mn['m10']/Mn['m00'])
        yc = int(Mn['m01']/Mn['m00'])   

        feature[15] = 2 * np.sqrt((xn-xc)**2 + (xc-yc)**2) / feature[10]

        # R         Maxima in nucleus               Nmax
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img,mask = mask_nuc)
        min_val = int(min_val)
        max_val = int(max_val)

        # instead of calculating pixels in a circle, I calculte the pixels in a small square
        feature[16] = np.sum((img[max(0, max_loc[1] - 3) : min(img.shape[0]-1, max_loc[1]+3), max(0, max_loc[0] - 3) : min(img.shape[1]-1, max_loc[0]+3)] == max_val))

        # S         Minima in nucleus               Nmin
        feature[17] = np.sum((img[max(0, min_loc[1] - 3) : min(img.shape[0]-1, min_loc[1]+3), max(0, min_loc[0] - 3) : min(img.shape[1]-1, min_loc[0]+3)] == min_val))
    

    # T         Maxima in cytoplasm             Cmax
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask = mask_cyt)
    min_val = int(min_val)
    max_val = int(max_val)

    # instead of calculating pixels in a circle, I calculte the pixels in a small square
    feature[18] = np.sum((img[max(0, max_loc[1] - 3) : min(img.shape[0]-1, max_loc[1]+3), max(0, max_loc[0] - 3) : min(img.shape[1]-1, max_loc[0]+3)] == max_val))

    # U         Minima in cytoplasm             Cmin
    feature[19] = np.sum((img[max(0, min_loc[1] - 3) : min(img.shape[0]-1, min_loc[1]+3), max(0, min_loc[0] - 3) : min(img.shape[1]-1, min_loc[0]+3)] == min_val))
    
    return feature


###################################################################################################################
#####################################   Second method to extract features #########################################
###################################################################################################################

# Idea to extrait features is from this paper
# https://ieeexplore.ieee.org/document/8451588
"""
26 features:
In every region of interest, we calculate 26 features concerning the intensity (average intensity, average contrast) 
and the texture (smoothness, uniformity, third moment, entropy) in all three color channels. 
Also, some shape features for each area were calculated (area, major and minor axis length, eccentricity, 
orientation, equivalent diameter, solidity and extent).
"""
def extract_feature_marina(img, seg):
    feature = np.zeros((26))

    # find contours which is very useful later
    mask = cv2.cvtColor(seg,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0): return feature
    elif (len(contours) == 1) : contours = contours[0]
    else : 
        max_contour = 0
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        contours = max_contour
    
    # intensity (average intensity, average contrast) and the 
    # texture (smoothness, uniformity, third moment, entropy) in all three color channels.
    feature[0:6] = statxture(masked_pixels(img[:,:,0], mask))
    feature[6:12] = statxture(masked_pixels(img[:,:,1], mask))
    feature[12:18] = statxture(masked_pixels(img[:,:,2], mask))

    # area
    feature[18] = cv2.contourArea(contours)
    
    # major and minor axis length, 
    (x,y),(MA,ma),angle = cv2.fitEllipse(contours)
    feature[19] = MA
    feature[20] = ma
    # eccentricity, 
    a = ma / 2
    b = MA / 2
    feature[21] = np.sqrt(a ** 2 - b ** 2) / a

    # orientation, 
    feature[22] = angle

    # equivalent diameter, 
    feature[23] = np.sqrt(4*feature[18]/np.pi)

    # solidity 
    hull = cv2.convexHull(contours)
    hull_area = cv2.contourArea(hull)
    feature[24] = feature[18]/hull_area

    # extent
    x,y,w,h = cv2.boundingRect(contours)
    rect_area = w*h
    feature[25] = feature[18]/rect_area
    
    return feature

# calculate intensity (average intensity, average contrast) and 
# the texture (smoothness, uniformity, third moment, entropy) in one channel. 

# the function of texture is from:
# https://github.com/joefutrelle/oii/blob/49d5f9dbd1675cf2c336dbb7df9c8195d087a3b1/ifcb2/features/texture.py
def statxture(pixels):
    """computes a variety of texture stats from
    the image histogram.
    See Digital Image Processing Using MATLAB, ch. 11"""
    average_gray_level = np.mean(pixels)
    average_contrast = np.std(pixels)

    H = histogram(pixels)[0]
    H = H / (1. * len(pixels))
    L = len(H)

    d = (L - 1.)**2

    normvar = np.var(pixels) / d
    smoothness = 1. - 1. / (1. + normvar)

    third_moment = moment(pixels,3) / d

    uniformity = np.sum(H**2)

    eps = np.finfo(float).eps
    entropy = 0. - np.sum(H * np.log2(H + eps)) 
    
    return average_gray_level, average_contrast, smoothness, \
        third_moment, uniformity, entropy

def masked_pixels(image,mask):
    return image[np.where(mask)]


###################################################################################################################
#####################################   Third method to extract features ##########################################
###################################################################################################################

"""
Previous methods all use the masks provided by the dataset, but in reality we don't have these masks. Instead of segmenting
the image manually, I think we can also try some unsupervised machine learning methods like ORB and k-means to automatically extract 
the features. And then we use some classical classifiers to predict the results. 

reference:
https://liverungrow.medium.com/sift-bag-of-features-svm-for-classification-b5f775d8e55f
"""

# mode can be "sift" or "orb", k is the number of clusters which will be used in k-means
def extract_feature_sift_kmeans(imgs, mode = "sift", k = 60):
    
    # calculate descriptrors of all images
    des = cal_descriptor_all(imgs, mode)
    
    # k-means to find clusters
    code_book = cal_codebook(des, k)

    # calculate the features
    return cal_img_features_all(imgs, code_book, mode), code_book


def cal_orb(img):
    """
    Calculate the ORB descriptors for an image.
    Args:
        img (BGR matrix): The image that will be used.
    Returns:
        list of floats array: The descriptors found in the image.
    """
    orb = cv2.ORB_create(edgeThreshold=2, patchSize=2)

    _, des = orb.detectAndCompute(img, None)

    return des

def cal_sift(img):
    """
    Calculate the ORB descriptors for an image.
    Args:
        img (BGR matrix): The image that will be used.
    Returns:
        list of floats array: The descriptors found in the image.
    """
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=20, sigma = 0.5)

    _, des = sift.detectAndCompute(img, None)

    return des


def cal_descriptor_all(imgs, mode = "sift"):
    """
    Calculate ORB descriptors of all images in training set.
    :param imgs: list of training set image. mode = "sift" or "orb"
    :return: numpy array, 
    """
    print('\nCalculating descriptor for all images:')
    descriptors_list = []
    for idx, img in enumerate(imgs):
        try:
            if mode == "sift":
                des = cal_sift(img)
            else:
                des = cal_orb(img)
            descriptors_list.append(des)  

        except cv2.error as e:
            print('Image {} error! '.format(idx), e)
    descriptors = np.concatenate(descriptors_list, axis=0)
    print('descriptors.shape: {}'.format(descriptors.shape))
    print('Calculating descriptors for all images completed!')
    return descriptors


def cal_codebook(all_descriptors, k = 20):
    """
    Tune sklearn kmeans to get optimal cluster size, which is the codebook size
    :param all_descriptors:
    :return: code book
    """
    print('\nStart calculating code book using K-means')

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(all_descriptors)

    print('Calculating code book completed!')
    return kmeans.cluster_centers_

def cal_img_features(img, codebook, mode = "sift"):
    """
    Calculate the features of a single image given the codebook(vocabulary) generated by clustering method (kmeans), each column is the center of the cluster.
    :param img:
    :param codebook:
    :return:
    """
    features = np.zeros((1, codebook.shape[0]))
    if mode == "sift":
        des = cal_sift(img)
    else :
        des = cal_orb(img)
    code, _ = vq(des, codebook)
    for i in code:
        features[0, i] += 1
    return(features)

# calculate the features of all images
def cal_img_features_all(imgs, codebook, mode = "sift"):
    print('\nStart calculating all image features:')
    features_all_list = []
    for img in imgs:
        this_features = cal_img_features(img, codebook, mode)
        features_all_list.append(this_features)
    features_all = np.concatenate(features_all_list, axis=0)
    print('features all shape is: {}'.format(features_all.shape))
    print('Calculating all image features completed!')
    return features_all

###################################################################################################################
#####################################   Forth method to extract features ##########################################
###################################################################################################################

# idea is from this paper
# https://doi.org/10.1007/s12652-020-02256-9

# the formula to calculate these features is from
# https://www.mdpi.com/2072-6694/12/12/3564/s1
# https://hal.inria.fr/hal-01420292/document
"""
Category and Features
Color features (6)              Mean, variance, skewness, kurtosis, energy1, entropy1
Texture features (8)            Energy2, entropy2, moment of inertia, correction, inverse moment, roughness, contrast, direction
Morphological features (6)      Area, perimeter, aspect ratio, circularity, rectangularity, nuclear-cytoplasmic ratio
"""
def extract_feature_dong(img, seg_cyt, seg_nuc):
    feature = np.zeros(20)

    # find contours which is very useful later
    mask = cv2.cvtColor(seg_cyt+seg_nuc,cv2.COLOR_BGR2GRAY)
    mask_cyt = cv2.cvtColor(seg_cyt,cv2.COLOR_BGR2GRAY)
    mask_nuc = cv2.cvtColor(seg_nuc,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0): return feature
    elif (len(contours) == 1) : contours = contours[0]
    else : 
        max_contour = 0
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        contours = max_contour
    

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ROI_pixels = masked_pixels(gray_img, mask)

    # Mean, variance
    feature[0] = np.mean(ROI_pixels)
    feature[1] = np.var(ROI_pixels)
    
    # skewness, kurtosis
    feature[2] = skew(ROI_pixels - feature[0])
    feature[3] = kurtosis(ROI_pixels - feature[0])

    # energy
    feature[4] = np.sum(ROI_pixels**2)

    # entropy
    H, bin = histogram(ROI_pixels)
    H = H / (1. * len(ROI_pixels))
    eps = np.finfo(float).eps
    feature[5] = 0. - np.sum(H * np.log2(H + eps)) 

    # Energy2, entropy2, moment of inertia, correction, inverse moment
    feature[6:11] = glcm_features(gray_img*(mask > 0))

    # Area, perimeter, aspect ratio, circularity, rectangularity, nuclear-cytoplasmic ratio
    feature[11] = cv2.contourArea(contours)

    feature[12] = cv2.arcLength(contours,True)

    (x,y),(MA,ma),angle = cv2.fitEllipse(contours)
    feature[13] = MA / ma

    # solidity 
    hull = cv2.convexHull(contours)
    hull_area = cv2.contourArea(hull)
    feature[14] = feature[11]/hull_area

    # extent
    x,y,w,h = cv2.boundingRect(contours)
    rect_area = w*h
    feature[15] = feature[11]/rect_area

    # nuclear-cytoplasmic ratio
    feature[16] = np.sum((mask_nuc > 0)) / np.sum((mask_cyt > 0))

    # roughness, contrast, direction
    feature[17] = feature[11] / (math.pi / 4 * MA**2)

    feature[18] = contrast(gray_img)

    feature[19] = angle

    return feature

def contrast(image):
	image = np.array(image)
	image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
	m4 = np.mean(np.power(image - np.mean(image),4))
	v = np.var(image)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v,2)
	fcon = std / np.power(alfa4, 0.25)
	return fcon

# Energy2, entropy2, moment of inertia, correction, inverse moment
def glcm_features(img):
    
    image = img_as_ubyte(img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds, [1], [0], levels=max_value, normed=True, symmetric=False)[:,:,0,0]
    
    # energy, entropy
    energy = np.sum(matrix_coocurrence**2)
    entropy  = np.sum(-matrix_coocurrence * np.log(matrix_coocurrence + 1e-6))

    # moment of inertia, correction, Dissimilarity
    inertia = 0
    correction = 0
    dissimilarity = 0
    for i in range(1, matrix_coocurrence.shape[0]+1):
        for j in range(1, matrix_coocurrence.shape[1]+1):
            inertia += (i-j)**2 * matrix_coocurrence[i-1][j-1]
            correction += i*j * matrix_coocurrence[i-1][j-1]
            dissimilarity += abs(i - j) * matrix_coocurrence[i-1][j-1]
    
    # print(energy, entropy, inertia, correction, dissimilarity )
    return energy, entropy, inertia, correction, dissimilarity


if __name__ == '__main__': 
    a = cv2.imread("Train/Train/3654.bmp")
    b = cv2.imread("Train/Train/3654_segCyt.bmp")
    c = cv2.imread("Train/Train/3654_segNuc.bmp")
    
    feature = extract_feature_dong(a,b,c)
    print(feature)

    

    

