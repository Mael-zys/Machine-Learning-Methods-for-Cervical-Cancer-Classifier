# Machine Learning Methods for Cervical Cancer Classifier

## Introduction 

This is a [challenge](https://www.kaggle.com/c/ima205challenge2021/overview) for our course IMA205. There are 2 tasks: binary classification and multi-class classification.

- For classifier, I implement [None, "SVM", "RF", "Bagging", "Logistic", "KNN", "PSO_SVM", "XGBoost", "MLP", "AutoML"], None means use all classifiers

- For feature extraction, I implement [None, ["martin"](https://www.researchgate.net/publication/265873515_Pap-smear_Benchmark_Data_For_Pattern_Classification), ["marina"](https://ieeexplore.ieee.org/document/8451588), ["sift_kmeans"](https://liverungrow.medium.com/sift-bag-of-features-svm-for-classification-b5f775d8e55f), ["dong"](https://doi.org/10.1007/s12652-020-02256-9), "martin_marina_dong"], None means use every pixels as features

- feature selection, I implement [None, "pca", "kpca", "spca", "select_best", "RF", "ExtraTrees", "shap", "RFECV", "SFS", "permutation"]

## Recommended environment

>python 3.6 \
>opencv-contrib-python 3.4.2. \
>opencv-python         4.1.2.30 \
>xgboost               1.2.0 \
>shap                  0.36.0 \
>mljar-supervised      0.8.9

## Dataset [DownLoad](https://drive.google.com/file/d/1KAAGC6vucd3p0wOJ8RPF30jo8-zuDxbI/view?usp=sharing)

Change the data directory "train_data_dir", "train_gt_dir" and "test_data_dir" in the machine_learning_dataloader.py to your own data directory.

Now I don't have the test labels

## Train and Predict

- You can firstly have a look at the arguments of the machine_learning_main.py

```
    python machine_learning_main.py --help
```

- Example

```
    python -u machine_learning_main.py --classifier "SVM" --binary True --mask_mode True --extract_feature "martin" --cv_mode "Grid"
```

- You can also write all these codes in the machine_learning_main.sh and then run them all

```
    sh machine_learning_main.sh
```

## Result

Here are some results. Because of some personal reasons, I lost most of submission history, and these are the few submission entries left.

### Binary classification


| Classifier | feature_extraction | feature_selection | feature number | Public Score |
|  ----  | ----  | ---- | ---- | ---- |
| SVM | marina | None | None | 0.94339 |
| AutoML | marina | None | None | 0.94072 |



### Multi-class classification


| Classifier | feature_extraction | feature_selection | feature number | Public Score |
|  ----  | ----  | ---- | ---- | ---- |
| SVM | martin_marina_dong | RF | 50 | 0.77220 |
| XGBoost | martin_marina_dong | RF | 50 | 0.76109 |
| AutoML | martin_marina_dong | None | None | 0.76821 |


## Future work

1. For the method sift_kmeans. Maybe it's better to use the library of sklearn "KElbowVisualizer" to help to select the best number of clusters.

2. Check the implementation of the feature extraction functions.

3. Visualize some results such as the results of feature selections methods.

## Reference

To be continued
