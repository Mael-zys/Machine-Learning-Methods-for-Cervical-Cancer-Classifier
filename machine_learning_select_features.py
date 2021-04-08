import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_selection import (RFE, RFECV, SelectKBest, chi2,
                                       f_classif, f_regression)
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.utils import shuffle
import shap
import util


# feature selection
# select_feature can be {None, "pca", "kpca", "spca", "select_best", "RF", "ExtraTrees", "shap", "RFECV", "SFS", "permutation"}
def select_features(X_train, X_test, y_train, select_feature = None, n_component = 20) :
    
    if select_feature is not None:
        print("feature selection mode is: " + select_feature)
    
    # pca
    if select_feature == "pca":
        pca = PCA(n_components=n_component,svd_solver='randomized', whiten=True)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    # Kernel Pca
    elif select_feature == "kpca":
        kpca = KernelPCA(n_components=n_component, kernel='rbf', gamma=2, n_jobs=8)
        kpca.fit(X_train)

        X_train = kpca.transform(X_train)
        X_test = kpca.transform(X_test)

    # Sparse Pca
    elif select_feature == "spca":
        spca = SparsePCA(n_components=n_component, n_jobs=8)
        spca.fit(X_train)

        X_train = spca.transform(X_train)
        X_test = spca.transform(X_test)

    # Variable Ranking
    elif select_feature == "select_best":
        bestfeatures = SelectKBest(score_func=f_classif, k=n_component)
        bestfeatures.fit(X_train, y_train.ravel())

        X_train = bestfeatures.transform(X_train)
        X_test = bestfeatures.transform(X_test)

    # Built-in Feature Importance: RF
    elif select_feature == "RF":
        model = RandomForestClassifier(n_jobs=8, random_state=0)
        model.fit(X_train, y_train.ravel())

        feature_importance = model.feature_importances_
        index = np.argsort(feature_importance)[-1:-n_component-1:-1]

        X_train = X_train[:, index]
        X_test = X_test[:, index]

    # Built-in Feature Importance: ExtraTreesClassifier
    elif select_feature == "ExtraTrees":
        model = ExtraTreesClassifier(n_jobs=8, random_state=0)
        model.fit(X_train, y_train.ravel())

        feature_importance = model.feature_importances_
        index = np.argsort(feature_importance)[-1:-n_component-1:-1]

        X_train = X_train[:, index]
        X_test = X_test[:, index]

    # shap
    # https://github.com/slundberg/shap
    elif select_feature == "shap":
        model = RandomForestClassifier(n_jobs=8, random_state=0)
        model.fit(X_train, y_train.ravel())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        shap_values_mean = np.mean(np.abs(shap_values[0]),axis=0)
        index = np.argsort(np.abs(shap_values_mean))[-1:-n_component-1:-1]

        X_train = X_train[:, index]
        X_test = X_test[:, index]

    # Recursive Feature Elimination
    elif select_feature == "RFECV":
        # Instantiate RFECV visualizer with a random forest classifier
        rfecv = RFECV(RandomForestClassifier(), n_jobs=8)

        rfecv.fit(X_train, y_train.ravel()) # Fit the data to the visualizer

        X_train = rfecv.transform(X_train)
        X_test = rfecv.transform(X_test)

    # Sequential Feature Selection
    elif select_feature == "SFS":
        # Build RF classifier to use in feature selection
        clf = RandomForestClassifier()

        # Sequential Forward Selection
        SFS = sfs(clf,
                k_features=n_component, 
                forward=True,
                floating=False,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=8,
                cv=5)

        SFS = SFS.fit(X_train, y_train.ravel())

        X_train = SFS.transform(X_train)
        X_test = SFS.transform(X_test)

    # Permutation importance
    elif select_feature == "permutation":
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        result = permutation_importance(rf, X_train, y_train.ravel(), n_repeats=10,
                                        random_state=42, n_jobs=8)
        sorted_idx = result.importances_mean.argsort()[-1:-n_component-1:-1]

        X_train = X_train[:, index]
        X_test = X_test[:, index]

    return X_train, X_test
