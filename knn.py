from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import os
import cv2
import numpy as np
import scipy.io
import time


class KNN:
    def __init__(self, X ,y, is_pca=False):
        self.X_train = X
        self.y_train = y
        self.clf = None
        self.is_pca = is_pca
        self.pca = None

    def fitWithoutPca(self):
        print("开始拟合.......")
        time_start = time.time()
        knn_clf = KNeighborsClassifier()
        self.y_train = self.y_train.ravel()

        knn_clf.fit(self.X_train, self.y_train)
        self.clf = knn_clf
        self.is_pca = False
        # self.save_model(self.clf, './model/predict_model_without_pca.m')
        time_end = time.time()

        print("拟合完毕")
        print('totally cost', time_end - time_start)

    def fitWithPca(self, n_components = 0.9):
        knn_clf = KNeighborsClassifier()
        self.pca = PCA(n_components)
        self.pca.fit(self.X_train)
        X_train_reducion = self.pca.transform(self.X_train)
        knn_clf.fit(X_train_reducion, self.y_train)
        self.clf = knn_clf
        self.is_pca = True

        # self.save_model(self.clf, './model/predict_model_with_pca.m')
        # self.save_model(self.pca, './model/pca.m')


    # TODO
    # 保存模型
    def save_model(self, model, name):
        joblib.dump(model, name)

    # TODO
    # 加载模型
    def load_model(self, predict_model_name, pca_model_name):
        if self.is_pca:
            self.pca = joblib.load(pca_model_name)
        self.clf = joblib.load(predict_model_name)

    def predict(self, X):
        if self.is_pca:
            X = self.pca.transform(X)

        y_pre = self.clf.predict(X)
        print(y_pre.shape)
        return y_pre
