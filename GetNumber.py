import cv2
import numpy as np
import random
import knn


def show_img(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def contours_sort(ccc, method=0):
    ccc = sorted(ccc, key=lambda x: (cv2.boundingRect(x)[0], cv2.boundingRect(x)[1]))
    return ccc

def read_img(knn_clf):
    img = cv2.imread('draw.jpg')
    show_img(img, 'src')
    # 灰度图
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_img(ref, 'gray')
    # 高斯滤波
    ref = cv2.GaussianBlur(ref, (5, 5), 1)
    show_img(ref, 'filter')
    # 阈值处理
    ref = cv2.threshold(ref, 100, 255, cv2.THRESH_BINARY_INV)[1]
    show_img(ref, 'threshold')
    # 边缘检测
    ref__, contours, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("最外侧轮廓数量：" + str(len(contours)))
    if len(contours) < 1:
        return None
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    show_img(img, 'contours')
    # 排序
    contours = contours_sort(contours, method='0')
    sub_imgs = []
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        # # TODO 为了适应MNIST数据集，暂时这样处理
        sub_img = ref[y:y + h, x:x + w]
        # 为了适应MNIST数据集，边界填充
        sub_img = cv2.copyMakeBorder(sub_img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=0)
        show_img(sub_img, str(i))
        sub_img = cv2.resize(sub_img, (28, 28))
        show_img(sub_img, str(i))
        # (28,28) -> (784,1)
        sub_img = sub_img.reshape((784,1))
        # (784,1) -> (1,784)
        sub_img = sub_img.T
        sub_imgs.append(sub_img)
    t = tuple(sub_imgs)
    X = np.vstack(t)
    print(X.shape)
    y_predict = knn_clf.predict(X)
    return y_predict

    # TODO
    # knn_clf.load_model('./model/predict_model_with_pca.m', './model/pca.m')
    # knn_clf.load_model('./model/predict_model_without_pca.m', None)


def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)


class Data:
    def __init__(self):
        self.X = None
        self.y = None
        self.sub_imgs = []
        self.target = []

    def getNumber(self):
        img = cv2.imread('draw.jpg')
        # 灰度图
        ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯滤波
        ref = cv2.GaussianBlur(ref, (5, 5), 1)
        # 阈值处理
        ref = cv2.threshold(ref, 100, 255, cv2.THRESH_BINARY_INV)[1]
        # 边缘检测
        ref__, contours, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            return None

        contours = contours_sort(contours, method='0')

        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            sub_img = ref[y:y + h, x:x + w]
            sub_img = cv2.copyMakeBorder(sub_img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=0)

            sub_img = cv2.resize(sub_img, (28, 28))
            # cv2.imwrite(str(i)+".jpg", sub_img)

            # (28,28) -> (784,1)
            x = sub_img.reshape((784,1))
            # (784,1) -> (1,784)
            x = x.T

            self.sub_imgs.append(x)
            print("输入target:")
            number = input()
            self.target.append(number)

            # 构造数据
            for i in range(999):
                img = rnd_warp(sub_img)
                x = sub_img.reshape((784, 1))   # (28,28) -> (784,1)
                x = x.T                         # (784,1) -> (1,784)
                self.sub_imgs.append(x)
                self.target.append(number)
            print("finish")

    def get_X_y(self):
        self.X = np.vstack(tuple(self.sub_imgs))
        self.y = np.vstack(tuple(self.target))
