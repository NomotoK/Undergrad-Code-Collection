# -*- coding: utf-8 -*-
# Created by: Leemon7
# Created on: 2021/6/22
# Function: KNN分类

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


# 读取鸢尾花数据集
data = datasets.load_iris()
# print(data)
target = data.target
data = data.data


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


class KNN:
    """使用Python语言实现K近邻算法（实现分类)"""

    def __init__(self, k):
        """
        初始化方法
        :param k: int， 邻居的个数
        """
        self.k = k

    def fit(self, X, y):
        """
        训练方法
        :param X: 类数组类型， 形状为：[样本数量，特征数量]， 待训练的样本特征（属性）
        :param y: 类数组类型，形状为: [样本数量], 每个样本的目标值（标签）
        :return:
        """
        # 将X转换成ndarray类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        """
        预测方法（对传递进来的样本数据进行预测）
        :param X:类数组类型， 形状为：[样本数量，特征数量]， 待预测的样本特征（属性）
        :return: result, 数组类型，预测的结果
        """
        X = np.asarray(X)
        result = []
        for x in X:
            # x为一行数据
            # ndarray数据可以直接进行相减
            # 使用欧氏距离进行计算两点的距离
            distance = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = distance.argsort()  # 通过排序，返回原数组中的索引
            # 进行截断，只取距离最小的k个元素的索引
            ind = index[:self.k]
            # bincountf返回每个元素出现的次数，元素必须是非负整数
            count = np.bincount(self.y[ind])
            # 返回ndarray中最大值所对应的索引
            result.append(count.argmax())
        return np.asarray(result)

    def predict2(self, X):
        """
        预测方法（对传递进来的样本数据进行预测）, 考虑权重（使用距离的倒数）
        :param X:类数组类型， 形状为：[样本数量，特征数量]， 待预测的样本特征（属性）
        :return: result, 数组类型，预测的结果
        """
        X = np.asarray(X)
        result = []
        for x in X:
            # x为一行数据
            # ndarray数据可以直接进行相减
            # 使用欧氏距离进行计算两点的距离
            distance = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = distance.argsort()  # 通过排序，返回原数组中的索引
            # 进行截断，只取距离最小的k个元素的索引
            ind = index[:self.k]
            # bincountf返回每个元素出现的次数，元素必须是非负整数
            count = np.bincount(self.y[ind], weights=1/distance[ind])
            # 返回ndarray中最大值所对应的索引
            result.append(count.argmax())
        return np.asarray(result)

    def score(self, y_predict, y_test):
        # c = y_test - y_predict
        # r1 = (c.shape[0] - np.count_nonzero(c)) / c.shape[0]
        r2 = np.sum(y_test==y_predict)/len(y_predict)
        # print(r1, r2)
        return r2

knn = KNN(k=5)
knn.fit(X_train, y_train)
# y_predict = knn.predict(X_test)
y_predict = knn.predict2(X_test)
# print(y_test)
# print(y_predict)
score = knn.score(y_predict, y_test)
print(score)


# 可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
# 默认情况下，matplotlib不支持中文显示，设置一下
mpl.rcParams['font.family'] = 'SimHei'
# 设置在中文字体时，能够正常显示负号-
mpl.rcParams['axes.unicode_minus'] = False
# 可视化训练集的数据
plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train)

# 可视化测试集的数据
right = X_test[y_predict == y_test]
wrong = X_test[y_predict != y_test]
plt.scatter(x=right[:,0], y=right[:,1], color='c', marker='x', label='right')
plt.scatter(x=wrong[:,0], y=wrong[:,1], color='m', marker='>', label='wrong')
plt.xlabel("花萼长度")
plt.ylabel("花瓣长度")
plt.title("KNN分类结果显示")
plt.legend(loc='best')
plt.show()