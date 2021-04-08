import argparse
import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import util
from machine_learning_dataloader import augment_contrast, sharpening
from machine_learning_extract_features import cal_descriptor_all

# The part is just to find the best number of cluster for k-means method


# use elbow method to find the best k for k-means
def tune_kmeans(all_descriptors, mode = "sift", save_folder = None):
    """
    Tune sklearn kmeans to get optimal cluster size, which is the codebook size
    :param all_descriptors, mode
    :return:
    """
    
    k_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    sse = []
    for i, k in enumerate(k_list):
        start_ts = datetime.datetime.now()
        print('\nRunning kmeans with cluster {}:'.format(k))
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=500)
        kmeans.fit(all_descriptors)
        sse.append(kmeans.inertia_)
        print('cluster {}: sse is {}'.format(k, sse[i]))
        end_ts = datetime.datetime.now()
        print('time of running : {}'.format(end_ts - start_ts))
    np.save(save_folder+'/sse_'+mode+'.npy', sse)
    plt.plot(k_list, sse)
    plt.plot(k_list, sse, 'o')
    plt.grid()
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.savefig(save_folder+'/tune_kmeans_'+mode+'.png')

def main(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # find the best k
    train_data_dir = 'Train/Train/'
    train_gt_dir = 'metadataTrain.csv'
    mode = args.mode
    save_folder = args.save_folder
    imgs = []
    
    gt_data = pd.read_csv(train_gt_dir)

    for idx, img_name in enumerate(gt_data["ID"]):
        img_path = train_data_dir + str(img_name) + '.bmp'    
        img = cv2.imread(img_path)
        img = sharpening(img)
        img = augment_contrast(img)
        imgs.append(img)

    imgs = np.array(imgs)
    
    descriptors = cal_descriptor_all(imgs, mode=mode)
    tune_kmeans(descriptors, mode = mode, save_folder = save_folder)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--mode', nargs='?', type=str, default='sift', 
                        help='sift or orb')
    # for saving the results
    parser.add_argument('--save_folder', nargs='?', type=str, default="./machine_learning_plot",    
                        help='Path to the folder to save the results, type=str, default="./machine_learning_plot"')
    args = parser.parse_args()

    main(args)
