import warnings

import numpy as np
from ailearn.Swarm import PSO

warnings.filterwarnings("ignore")

import random

import sklearn
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from supervised.automl import AutoML
random.seed(0)

def SVM_train_prediction(X_train, X_test, y_train, binary = True, cv_mode = "Grid") :
    print("\nSVM estimator")
    
    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # gaussian kernel
    p_grid_lsvm = {'C': [1e-3,1e-2,0.05,1e-1,0.5,1,2,5,1e1,1e2,15,20,40,60,80,120,140],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1,2,4,8,10,15,20,30,40,50,60], }
    # Lsvm = SVC(kernel='rbf', class_weight="balanced")
    Lsvm = SVC(kernel='rbf')
    if cv_mode == "Grid":
        grid_lsvm = GridSearchCV(estimator=Lsvm, param_grid=p_grid_lsvm, scoring=scoring, cv=5, n_jobs=8)
    else:
        grid_lsvm = RandomizedSearchCV(estimator=Lsvm, param_distributions=p_grid_lsvm, scoring=scoring, cv=5, n_jobs=8)
    grid_lsvm.fit(X_train, y_train.ravel())
    print("Best training Score: {}".format(grid_lsvm.best_score_))
    print("Best training params: {}".format(grid_lsvm.best_params_))
    y_pre = grid_lsvm.predict(X_test)
    return y_pre

def RF_train_prediction(X_train, X_test, y_train, binary = True, cv_mode ='Grid'):
    print("\nRandom forest estimator")

    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # Random forest
    # RF=RandomForestClassifier(random_state=0, class_weight="balanced")
    RF=RandomForestClassifier(random_state=0)
    p_grid_RF = {'n_estimators': [10,15,20,25,30], 'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_leaf':[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'max_features': ['sqrt','log2']}   

    if cv_mode == "Grid":
        grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF, scoring=scoring, cv=5, n_jobs=8)
    else:
        grid_RF = RandomizedSearchCV(estimator=RF, param_distributions=p_grid_RF, scoring=scoring, cv=5, n_jobs=8)
    grid_RF.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_RF.best_score_))
    print("Best params: {}".format(grid_RF.best_params_))
    y_pre = grid_RF.predict(X_test)
    return y_pre

def bagging_train_prediction(X_train, X_test, y_train, binary=True, cv_mode = "Grid"):
    print("\nBagging estimator")

    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # Tree = DecisionTreeClassifier(random_state=0, class_weight="balanced")
    Tree = DecisionTreeClassifier(random_state=0)
    p_grid_tree = {'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_leaf':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]} 
    
    if cv_mode == "Grid":
        grid_tree = GridSearchCV(Tree, p_grid_tree, cv=5, scoring = scoring, n_jobs = 8)
    else:
        grid_tree = RandomizedSearchCV(Tree, p_grid_tree, cv=5, scoring = scoring, n_jobs = 8)
    
    grid_tree.fit(X_train, y_train.ravel())
    best_params = grid_tree.best_params_
    
    # Tree = DecisionTreeClassifier(min_samples_leaf=best_params["min_samples_leaf"],min_samples_split=best_params["min_samples_split"], random_state=0, class_weight="balanced")
    Tree = DecisionTreeClassifier(min_samples_leaf=best_params["min_samples_leaf"],min_samples_split=best_params["min_samples_split"], random_state=0)
    p_grid_bagging = {'n_estimators': [5,10,15,20]}      
    bag=BaggingClassifier(base_estimator=Tree, random_state=0)
    
    if cv_mode == "Grid":
        grid_bagging = GridSearchCV(estimator=bag, param_grid=p_grid_bagging, cv=5, scoring=scoring, n_jobs=8)
    else:
        grid_bagging = RandomizedSearchCV(estimator=bag, param_distributions=p_grid_bagging, cv=5, scoring=scoring, n_jobs=8)

    grid_bagging.fit(X_train, y_train.ravel())
    print("Best Validation Score: {}".format(grid_bagging.best_score_))
    print("Best params: {}".format(grid_bagging.best_params_))
    y_pre = grid_bagging.predict(X_test)
    return y_pre

def logistic_train_prediction(X_train, X_test, y_train, binary = True, cv_mode = "Grid") :
    print("\nLogistic estimator")
    
    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # Logistic
    # logi=LogisticRegression(class_weight="balanced")
    logi=LogisticRegression(random_state=0, max_iter=10000)
    p_grid_log = {'C': [1e-3,1e-2,0.05,1e-1,0.5,1,2,5,1e1,1e2,15,20,40]}  

    if cv_mode == "Grid":
        grid_log = GridSearchCV(estimator=logi, param_grid=p_grid_log, scoring=scoring, cv=5, n_jobs = 8)
    else:
        grid_log = RandomizedSearchCV(estimator=logi, param_distributions=p_grid_log, scoring=scoring, cv=5, n_jobs = 8)

    grid_log.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_log.best_score_))
    print("Best params: {}".format(grid_log.best_params_))
    y_pre = grid_log.predict(X_test)
    return y_pre

def KNN_train_prediction(X_train, X_test, y_train, binary=True, cv_mode = "Grid") :
    print("\nKNN estimator")

    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # KNN
    KNN=KNeighborsClassifier()
    p_grid_knn = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,15]}

    if cv_mode == "Grid":
        grid_knn = GridSearchCV(estimator=KNN, param_grid=p_grid_knn, scoring=scoring, cv=5, n_jobs = 8)
    else:
        grid_knn = RandomizedSearchCV(estimator=KNN, param_distributions=p_grid_knn, scoring=scoring, cv=5, n_jobs = 8)

    grid_knn.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_knn.best_score_))
    print("Best params: {}".format(grid_knn.best_params_))
    y_pre = grid_knn.predict(X_test)
    return y_pre

def PSO_SVM_train_prediction(X_train, X_test, y_train, binary = True, cv_mode = "Grid") :
    print("\nPSO-SVM estimator")
    
    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    def svm_score(C, gamma):
        model = SVC(C=C, gamma=gamma)
        scores = cross_val_score(model, X_train, y_train.ravel(), cv=5, scoring = scoring, n_jobs=8)
        return np.mean(scores)


    p = PSO(svm_score, 2, x_min=[2**(-8), 2**(-8)], x_max=[20, 10], v_min=[-2, -1], v_max=[2, 1])
    
    C, gamma = p.solve(epoch=20, verbose=True)
    
    print("Best training Score: {}".format(svm_score(C, gamma)))
    print("Best training params: {}".format((C, gamma)))
    
    best_svc = SVC(C=C, gamma=gamma)
    best_svc.fit(X_train, y_train.ravel())
    y_pre = best_svc.predict(X_test)

    return y_pre

def XGBoost_train_prediction(X_train, X_test, y_train, binary = True, cv_mode = "Grid") :
    print("\nXGBoost estimator")
    
    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # XGB
    XGB = XGBClassifier(nthread = 8)
    p_grid_xgb = dict(
        max_depth = [4, 5, 6, 7],
        learning_rate = np.linspace(0.03, 0.3, 10),
        n_estimators = [100, 200]
    )

    if cv_mode == "Grid":
        grid_xgb = GridSearchCV(estimator=XGB, param_grid=p_grid_xgb, scoring=scoring, cv=5, n_jobs = 8)
    else:
        grid_xgb = RandomizedSearchCV(estimator=XGB, param_distributions=p_grid_xgb, scoring=scoring, cv=5, n_jobs = 8)

    grid_xgb.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_xgb.best_score_))
    print("Best params: {}".format(grid_xgb.best_params_))
    y_pre = grid_xgb.predict(X_test)
    return y_pre

# MLP
def MLP_train_prediction(X_train, X_test, y_train, binary = True, cv_mode = "Grid") :
    print("\nMLP estimator")
    
    if binary:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    # MLP
    MLP = MLPClassifier(activation='relu', alpha=1e-3, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 100), 
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       tol=0.0001, validation_fraction=0.1,
       warm_start=False, verbose=10)

    p_grid_mlp = {'solver':['adam', 'sgd'], 'learning_rate' : ['adaptive', 'constant']}

    if cv_mode == "Grid":
        grid_mlp = GridSearchCV(estimator=MLP, param_grid=p_grid_mlp, scoring=scoring, cv=5, n_jobs = 8)
    else:
        grid_mlp = RandomizedSearchCV(estimator=MLP, param_distributions=p_grid_mlp, scoring=scoring, cv=5, n_jobs = 8)

    grid_mlp.fit(X_train, y_train.ravel())

    print("Best Validation Score: {}".format(grid_mlp.best_score_))
    print("Best params: {}".format(grid_mlp.best_params_))
    y_pre = grid_mlp.predict(X_test)
    return y_pre

# AutoML
# https://github.com/mljar/mljar-supervised
def AutoML_train_prediction(X_train, X_test, y_train, binary=True, cv_mode = "Grid") :
    print("\nAutoML estimator")

    automl = AutoML(mode="Compete")
    automl.fit(X_train, y_train.ravel())

    y_pre = automl.predict(X_test)

    return y_pre