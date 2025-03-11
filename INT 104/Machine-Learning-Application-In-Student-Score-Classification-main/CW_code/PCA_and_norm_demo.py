'''
数据预处理和降维，包括去空，标准化步骤和PCA主成分分析，最后生成二维图表，
并将降维后的数据保存为新的csv文件存储在根目录中供分类器调用
数据集都在archive里面


created by Hailn Xie 2023/02/08
'''




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import statsmodels as sm
# import pylab as pl
# import sklearn as sk
# from boto import sns
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import preprocessing

def read_CSV(path):  #读取csv数据集
    data = pd.read_csv(path, sep=',', header=0)
    confirmation = input('Do you want to remove any columns? (YES/NO) ')
    if(confirmation == 'YES'):
        column = input('select column: ')
        data.drop(labels=column, axis=1, inplace=True)  # 去除无关的列
    else:
        pass
    print("read_file: " + path)  # 显示所读取的文件
    print(str(data.shape[1]) + ' columns, ' + str(data.shape[0]) + ' rows')
    return data


# def data_cleaning(data): #去除噪音
#     attribute = input('input attribute')
#     data.drop(data[data['Programme'] == 0].index, inplace=True)  # remove major 0
#     data.drop(data[data['Programme'].isnull()].index, inplace=True)  # remove empty rows
#     data.drop_duplicates() # remove noises
#
#     return data


def data_cleaning(data): #去除噪音
    confirmation2 = input('Do you want to remove any empty attribute? (YES/NO) ')
    if(confirmation2 == 'YES'):
        attribute = input('select attribute: ')
        data.drop(data[data[attribute].isnull()].index, inplace=True)  # remove empty rows
        data.drop_duplicates()  # remove noises
        return data
    else:
        return data


def checkNull(data): #检查是否有空的列
    if data[data.isnull().T.any()].empty:
        print("No null attributes detected")

    else:
        print("Null attributes detected")
        return data[data.isnull().T.any()]

# def count_eachAttribute(data):
#     # remember: count before data cleaning
#     data_0 = data.loc[data.Programme == 0, :].iloc[:, :-1]
#     data_1 = data.loc[data.Programme == 1, :].iloc[:, :-1]
#     data_2 = data.loc[data.Programme == 2, :].iloc[:, :-1]
#     data_3 = data.loc[data.Programme == 3, :].iloc[:, :-1]
#     data_4 = data.loc[data.Programme == 4, :].iloc[:, :-1]
#     print(data_0.values, data_1.values, data_2.values, data_3.values, data_4.values)
#     return data_0.values, data_1.values, data_2.values, data_3.values, data_4.values

def count_eachAttribute_1(data):
    attribute = []
    for i in range(0, data.shape[1]):
        exec('data_{} = {}'.format(i, data.loc[data.Programme == i, :].iloc[:, :-1]))
    names = locals()
    list1 = []
    for i in range(0, data.shape[1]):
        print(names.get('data_' + str(i)).values)
        # print(attribute[i].values)
        # return ('data_' + str(i)).values


#
#
#
# def get_PCA(Majors):
#     Major_1 = Majors[1]
#     Major_2 = Majors[2]
#     Major_3 = Majors[3]
#     Major_4 = Majors[4]
#     dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0)  # reunion the numpy array
#     scaler = MinMaxScaler()  # StandardScaler MinMaxScaler RobustScaler
#     dataset = scaler.fit_transform(dataset)
#     # pca1 = PCA(n_components=4)
#     pca2 = PCA(n_components=2)
#
#
#     dataset_reduced = pca2.fit_transform(dataset)
#     score = pca2.score_samples(dataset)
#     # plt.figure()
#     plt.title("score distribution")
#     plt.scatter(range(len(score)), score, c='salmon')
#     plt.show()
#
#
#     # plt.figure()
#     plt.title('PCA')
#     for i in range(len(Major_1)): #elements ahead belong to major 1 because data are unioned in line
#         plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='turquoise')
#     for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
#         plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='salmon')
#     for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
#         plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='gold')
#     for i in range(len(Major_1) + len(Major_2) + len(Major_3),
#                    len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
#         plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='violet')
#     plt.savefig('./PCA.jpg')  # Save the pic of the PCA
#     plt.show()
#
#
#     DF1 = pd.DataFrame(dataset_reduced[0:len(Major_1),:])
#     DF1['Programme'] = 1
#     DF2 = pd.DataFrame(dataset_reduced[len(Major_1):len(Major_1) + len(Major_2),:])
#     DF2['Programme'] = 2
#     DF3 = pd.DataFrame(dataset_reduced[len(Major_1) + len(Major_2):len(Major_1) + len(Major_2) + len(Major_3),:])
#     DF3['Programme'] = 3
#     DF4 = pd.DataFrame(dataset_reduced[len(Major_1) + len(Major_2) + len(Major_3):len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4),:])
#     DF4['Programme'] = 4
#     Dataset = pd.concat([DF1, DF2, DF3, DF4], axis=0, join='inner')
#     Dataset.to_csv('./PCAfeatures.csv', index=False, header=True)  # Ignore index and header
#
#
#
#
#
file = input('select file: ')
data = read_CSV('./archive/'+ file + '.csv')

# print(data.describe())
#
#
data = data_cleaning(data)
checkNull(data)
# count_eachAttribute(data)
count_eachAttribute_1(data)
#
#
# features = data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
# scaler = MinMaxScaler() # normalisation process
# features = scaler.fit_transform(features.values)
#
# Dataset = pd.concat([pd.DataFrame(features[:, 0]), pd.DataFrame(features[:, 1]), pd.DataFrame(features[:, 2]), pd.DataFrame(features[:, 3]), pd.DataFrame(features[:, 4])], axis=1, join='inner')
# Dataset.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
# print(Dataset.corr())
#
#
# get_PCA(count_eachAttribute(data))

