import argparse
import os
import warnings
import ast
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

# functions to train and predict data
from machine_learning_classifier import (KNN_train_prediction,
                                         RF_train_prediction,
                                         SVM_train_prediction,
                                         bagging_train_prediction,
                                         logistic_train_prediction,
                                         PSO_SVM_train_prediction,
                                         XGBoost_train_prediction,
                                         MLP_train_prediction,
                                         AutoML_train_prediction)
# functions to read data and pre process
from machine_learning_dataloader import (pre_process, read_test_data,
                                         read_train_data, write_data)


def main(args):
    
    if args.classifier == "SVM":
        classifiers = [SVM_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "RF":
        classifiers = [RF_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "Bagging":
        classifiers = [bagging_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "Logistic":
        classifiers = [logistic_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "KNN":
        classifiers = [KNN_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "PSO_SVM":
        classifiers = [PSO_SVM_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "XGBoost":
        classifiers = [XGBoost_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "MLP":
        classifiers = [MLP_train_prediction]
        classifier_names = [args.classifier]
    elif args.classifier == "AutoML":
        classifiers = [AutoML_train_prediction]
        classifier_names = [args.classifier]
    else:
        classifiers = [KNN_train_prediction, SVM_train_prediction, bagging_train_prediction, logistic_train_prediction, RF_train_prediction, PSO_SVM_train_prediction, XGBoost_train_prediction, MLP_train_prediction]
        classifier_names = ["KNN", "SVM", "Bagging", "Logistic", "RF", "PSO_SVM", "XGBoost", "MLP", "AutoML"]


    if args.binary == True:
        label_name = "ABNORMAL"
    else:
        label_name = "GROUP"

    if args.extract_feature == None:
        feature_name = "pixel"
    else:
        feature_name = args.extract_feature

    if args.select_feature == None:
        select_name = "None"
    else:
        select_name = args.select_feature


    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)


    # read training data 
    X_train, y_train, code_book = read_train_data(img_size = args.img_size, label_name = label_name, extract_feature = args.extract_feature, mask_mode = args.mask_mode, sift_orb = args.sift_orb, num_clusters = args.num_clusters)
    
    # read test data
    X_test, X_path = read_test_data(img_size = args.img_size, extract_feature = args.extract_feature, mask_mode = args.mask_mode, code_book = code_book, sift_orb = args.sift_orb)

    # pre processing (for example, scale, PCA)
    X_train, X_test, y_train = pre_process(X_train, X_test, y_train, select_feature = args.select_feature, n_component = args.num_components)

    # train and prediction
    for i, classifier in enumerate(classifiers):
        y_pre = classifier(X_train, X_test, y_train, binary = args.binary, cv_mode = args.cv_mode)
        write_data(y_pre, label_name, X_path, args.save_folder + '/'+label_name +'_'+ classifier_names[i] + '_' + feature_name + '_' + args.sift_orb + '_' + select_name + '_'+ str(args.mask_mode) + '_' + args.cv_mode + '.csv')




if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Hyperparams')

    # choose a classifier
    parser.add_argument('--classifier', nargs='?', type=str, default=None, 
                        help='Choose classifier between [None, "SVM", "RF", "Bagging", "Logistic", "KNN", "PSO_SVM", "XGBoost", "MLP", "AutoML"]. None means run all classifiers. type=str, default=None')
    
    # hyper parameter for reading data, extract features and select features
    parser.add_argument('--img_size', nargs='?', type=int, default=256, 
                        help='reshape the image to this size(except the sift methods). type=int, default=256')
    parser.add_argument('--binary', nargs='?', type=ast.literal_eval, default=True, 
                        help='True or False: binary classification or multiclassification. type=bool, default=True')
    parser.add_argument('--extract_feature', nargs='?', type=str, default=None, 
                        help='Choose feature mode between [None, "martin", "marina", "sift_kmeans", "dong", "martin_marina_dong"], None means consider all pixels as features. type=str, default=None')
    parser.add_argument('--select_feature', nargs='?', type=str, default=None, 
                        help='choose between [None, "pca", "kpca", "spca", "select_best", "RF", "ExtraTrees", "shap", "RFECV", "SFS", "permutation"]. type=str, default=None')
    parser.add_argument('--mask_mode', nargs='?', type=ast.literal_eval, default=True, 
                        help='True or False: Whether to use the masks or not. For the "martin" and "marina" feature mode, the mask_mode must be True. type=bool, default=True')
    parser.add_argument('--sift_orb', nargs='?', type=str, default="sift", 
                        help='for the feature mode "sift_kmeans", choose between ["sift", "orb"], type=str, default="sift"')
    parser.add_argument('--num_clusters', nargs='?', type=int, default=60, 
                        help='for the feature mode "sift_kmeans", choose number of clusters for k-means(We can obtain this parameter by the tune_kmeans function)  type=int, default=60')
    parser.add_argument('--num_components', nargs='?', type=int, default=20, 
                        help='choose number of features for feature selections methods. type=int, default=20')
    
    # for training
    parser.add_argument('--cv_mode', nargs='?', type=str, default="Grid",    
                        help='Choose between ["Grid", "Randomized", means GridSearchCV or RandomizedSearchCV. type=str, default="Grid"')
    
    # for saving the results
    parser.add_argument('--save_folder', nargs='?', type=str, default="./machine_learning_submission",    
                        help='Path to the folder to save the results, for example "./submission", then we will save the results to this folder. type=str, default="./machine_learning_submission"')
    
    args = parser.parse_args()

    main(args)



