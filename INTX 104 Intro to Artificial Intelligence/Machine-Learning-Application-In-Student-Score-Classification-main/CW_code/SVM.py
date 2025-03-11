import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
from numpy import nan
from sklearn.datasets._base import Bunch
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from task1 import read_CSV
from sklearn.model_selection import learning_curve


def plot_svc_decision_function(model ,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0] ,xlim[1] ,30)
    y = np.linspace(ylim[0] ,ylim[1] ,30)
    Y ,X = np.meshgrid(y ,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P ,colors="k" ,levels=[-1 ,0 ,1] ,alpha=0.5 ,linestyles=["--" ,"-" ,"--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def read_CSV(path):
    data = pd.read_csv(path, sep=',', header=0)
    # data.drop(labels="ID", axis=1, inplace=True) #remove ID column
    data.drop(data[data['Programme'].isnull()].index, inplace=True)
    # data.drop_duplicates(subset=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], keep=False, inplace=True, )
    print("read_file: " + path)
    return data

def show_accuracy(y_hat, y_train, param):
    pass


# nmfdata = np.array(getNMFhandwritting)
# print(combined)
combined = np.array(read_CSV('NMFfeatures.csv'))
x, y = np.split(combined, (2,), axis=1)
x = x[:, 0:2]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=26, test_size=13)
clf = svm.SVC(C=4, kernel='rbf', gamma=5, decision_function_shape='ovr')  # ovr one to many
clf.fit(x_train, y_train.ravel())

print('SVM-the accuracy of train dataset is:', clf.score(x_train, y_train))  # accuracy
y_hat = clf.predict(x_train)
show_accuracy(y_hat, y_train, '训练集')
print('SVM-the accuracy of test dataset is:', clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
show_accuracy(y_hat, y_test, '测试集')

print('\ndecision_function:\n', clf.decision_function(x_train))  # check the decision fiction
print('\npredict:\n', clf.predict(x_train))

y_predict = clf.predict(x_test)
print(y_predict)
print("直接比对真实值和预测值:\n", y_test == y_predict)

score = clf.score(x_test, y_test)
print("准确率为：\n", score)


# Learning curve 检视过拟合
train_sizes, train_loss, test_loss = learning_curve(
    svm.SVC(gamma=0.001), x, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

# 平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
plt.figure()
plt.suptitle('Mean variance of training and cross-validation')
plt.title("(use NMF result as input)")
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
         label="Cross-validation")
plt.legend(loc='best')
plt.show()


# plt.figure()
# # clf = svm.SVC(kernel = "linear").fit(x,y)
# plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap="rainbow")
# plot_svc_decision_function(clf)
# r = np.exp(-(x**2).sum(1))
# plt.show()
#

plt.scatter(x[: ,0] ,x[: ,1] ,c=y ,s=50 ,cmap="rainbow")
ax = plt.gca()
xlim = ax.get_xlim()
ylim=ax.get_ylim()
xx = np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
XX,YY=np.meshgrid(xx,yy)
xy=np.vstack([XX.ravel(),YY.ravel()]).T
plt.scatter(xy[:,0],xy[:,1],s=1,cmap='rainbow')
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(x,y)
plt.scatter(x[:,0],x[:,1],c=y)
plt.title('SVM classification result')


plt.show()

#
#
#
#
# data = pd.read_csv("./CW_Data.csv", sep=',', header=0)
# data.drop(data[data['Programme'].isnull()].index, inplace=True)
# data.drop(labels="ID", axis=1, inplace=True)
# data.drop(labels="Programme", axis=1, inplace=True)
# data.drop_duplicates()
# data.to_csv('./processed.csv', index=False, header=True)
#
# # 1.读取数据集
#
#
# csvFile = open("processed.csv")
# csv_data = csv.reader(csvFile)
# cancer = np.array([i for i in csv_data])
# attribute_names = cancer[0, :5]
# da = cancer[1:, :5]
# data = []
# for i in da:
#     temp = []
#     for j in i:
#         if j == '?':
#             temp.append(nan)
#         else:
#             temp.append(int(float(j)))
#     data.append(temp)
# attribute_names = cancer[0, :4]
# target = []
# for i in cancer[0:, 4]:
#     if i == '0':
#         target.append(0)
#     if i == '1':
#         target.append(1)
#     if i == '2':
#         target.append(2)
#     if i == '3':
#         target.append(3)
#     if i == '4':
#         target.append(4)
# target_names = ['0', '1', '2', '3', '4']
# real_data = Bunch(data=data, target=target, feature_names=attribute_names, target_names=target_names)
# x_train, y_train, x_test,  y_test = train_test_split(real_data.data, real_data.target, test_size=0.5)
#
# # 3.训练svm分类器
# classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr:一对多策略
# classifier.fit(x_train, y_train)  # ravel函数在降维时默认是行序优先
#
# # 4.计算svc分类器的准确率
# print("训练集：", classifier.score(x_train, y_train))
# print("测试集：", classifier.score(x_test, y_test))
#
# # 查看决策函数
# print('train_decision_function:\n', classifier.decision_function(x_train))
# print('predict_result:\n', classifier.predict(x_train))
#
# y_predict = classifier.predict(x_test)
# print(y_predict)
# print("直接比对真实值和预测值:\n", y_test == y_predict)
#
# score = classifier.score(x_test, y_test)
# print("准确率为：\n", score)
#
# # Learning curve 检视过拟合
# train_sizes, train_loss, test_loss = learning_curve(
#     svm.SVC(gamma=0.001), real_data.data, real_data.target, cv=10, scoring='neg_mean_squared_error',
#     train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
#
# # 平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
# plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
#          label="Training")
# plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
#          label="Cross-validation")
#
# plt.xlabel("Training examples")
# plt.ylabel("Loss")
# plt.legend(loc="best")
# plt.show()
#
# param_range = np.logspace(-6, -2.3, 5)
#
# # 使用validation_curve快速找出参数对模型的影响
# train_loss, test_loss = validation_curve(
#     svm.SVC(), x_train, y_train, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error')
#
# # 平均每一轮的平均方差
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
# # 可视化图形
# plt.plot(param_range, train_loss_mean, 'o-', color="r",
#          label="Training")
# plt.plot(param_range, test_loss_mean, 'o-', color="g",
#          label="Cross-validation")
#
# plt.xlabel("gamma")
# plt.ylabel("Loss")
# plt.legend(loc="best")
# plt.show()
