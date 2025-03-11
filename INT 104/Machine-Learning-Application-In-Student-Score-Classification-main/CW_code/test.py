import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


#
#
#
# df = pd.read_csv('./CW_Data.csv', sep=',', header= 0)
# df.drop(labels="ID", axis=1, inplace=True)
# df.drop(df[df['Programme'].isnull()].index,inplace=True)
# df.drop_duplicates(subset=['Q1','Q2','Q3','Q4','Q5'], keep= False , inplace= True,)
# data = df.values
# df.describe()
# Class_0 = []
# Class_1 = []
# Class_2 = []
# Class_3 = []
# Class_4 = []
#
#
# for i in range(len(data)):
#     if data[i,-1] == 0:
#         Class_0.append(data[i,1:-1])
#     elif data[i,-1] == 1:
#         Class_1.append(data[i,1:-1])
#     elif data[i,-1] == 2:
#         Class_2.append(data[i,1:-1])
#     elif data[i,-1] == 3:
#         Class_3.append(data[i,1:-1])
#     elif data[i,-1] == 4:
#         Class_4.append(data[i,1:-1])
#
#
# # programme_count = df['Programme'].value_counts()
# # plt.figure()
# # plt.title('Programme Count')
# # for i in range(0, 5):
# #     plt.bar(str(i),  programme_count[i])
# # plt.show()
#
#
# plt.figure()
# plt.title('PCA(finalized)')
# numpy_name = np.array(df)
# pca = PCA(n_components = 2)
#
# dataset = np.concatenate([Class_0,Class_1,Class_2,Class_3,Class_4],axis=0)
# scaler = StandardScaler()
# dataset = scaler.fit_transform(dataset)
# newpca = pca.fit_transform(dataset)
# pca_reduced = pca.fit_transform(numpy_name)
#
# for i in range(len(Class_0)):
#     plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='gold')
# for i in range(len(Class_0),len(Class_0)+len(Class_1),1):
#     plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='salmon')
# for i in range(len(Class_0)+len(Class_1),len(Class_0)+len(Class_1)+len(Class_2),1):
#     plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='turquoise')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),1):
#     plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='violet')
# for i in range(len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3),len(Class_0)+len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='cornflowerblue')
# plt.show()
#
#
#
# plt.figure()
# plt.title('LDA')
# lda =LDA(n_components=2)
#
#
# train_x=np.concatenate([Class_1,Class_2,Class_3,Class_4],axis=0)
# scaler = StandardScaler()
# train_x = scaler.fit_transform(train_x)
# Class_1_y=np.zeros(len(Class_1))+1
# Class_2_y=np.zeros(len(Class_2))+2
# Class_3_y=np.zeros(len(Class_3))+3
# Class_4_y=np.zeros(len(Class_4))+4
# train_y=np.concatenate([Class_1_y,Class_2_y,Class_3_y,Class_4_y],axis=0)
# afterlda=lda.fit_transform(train_x,train_y)
# for i in range(len(Class_1)):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='turquoise')
# for i in range(len(Class_1),len(Class_1)+len(Class_2),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='salmon')
# for i in range(len(Class_1)+len(Class_2),len(Class_1)+len(Class_2)+len(Class_3),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='gold')
# for i in range(len(Class_1)+len(Class_2)+len(Class_3),len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='violet')
# plt.show()



    # Major_1 = Majors[1]
    # Major_2 = Majors[2]
    # Major_3 = Majors[3]
    # Major_4 = Majors[4]
    # dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0)  # reunion the numpy array
    # label_1 = np.zeros(len(Major_1)) + 1  # to build a label matrix
    # label_2 = np.zeros(len(Major_2)) + 2
    # label_3 = np.zeros(len(Major_3)) + 3
    # label_4 = np.zeros(len(Major_4)) + 4
    # labels = np.concatenate([label_1, label_2, label_3, label_4], axis=0)
    #
    # # Standardize the feature
    # sc = StandardScaler()
    # data_std = sc.fit_transform(dataset)
    #
    # # Count mean vectors
    # np.set_printoptions(precision=4)
    # mean_vecs = []
    # for label in range(1, 4):
    #     mean_vecs.append(np.mean(data_std[labels == label], axis=0))
    #
    # # Count SW matrix
    # k = 5 # k is the original dimension
    # Sw = np.zeros((k, k))
    # for label, mv in zip(range(1, 4), mean_vecs):
    #     Si = np.cov(data_std[labels == label].T)
    #     Sw += Si
    #
    # # Count Sb matrix
    # mean_all = np.mean(data_std, axis=0)
    # Sb = np.zeros((k, k))
    # for i, col_mv in enumerate(mean_vecs):
    #     n = dataset[labels == i + 1, :].shape[0]
    #     col_mv = col_mv.reshape(k, 1)  # column mean vector
    #     mean_all = mean_all.reshape(k, 1)
    #     Sb += n * (col_mv - mean_all).dot((col_mv - mean_all).T)
    #
    # # Count Generalized Eigenvalues
    # eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    # eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    #
    # # Transform matrix
    # w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
    # afterlda = data_std.dot(w)
    # plt.title('LDA-planB')
    # for i in range(len(Major_1)):  #elements ahead belong to major 1 because data are unioned in line
    #     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='blue')
    # for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
    #     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='yellow')
    # for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
    #     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='black')
    # for i in range(len(Major_1) + len(Major_2) + len(Major_3),
    #                len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
    #     plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='red')
    # plt.show()



# plt.figure()
# plt.title('t-SNE')
# tsne =TSNE(n_components=2)
#
#
# dataset=np.concatenate([Class_1,Class_2,Class_3,Class_4],axis=0)
# scaler = StandardScaler()
# dataset = scaler.fit_transform(dataset)
# tsne.fit_transform(dataset)
# newdataset = tsne.embedding_
# for i in range(len(Class_1)):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='turquoise')
# for i in range(len(Class_1),len(Class_1)+len(Class_2),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='salmon')
# for i in range(len(Class_1)+len(Class_2),len(Class_1)+len(Class_2)+len(Class_3),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='gold')
# for i in range(len(Class_1)+len(Class_2)+len(Class_3),len(Class_1)+len(Class_2)+len(Class_3)+len(Class_4),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='violet')
# plt.show()



# list1 = []
# for i in range(5):
#      list1.append(exec('var{} = {}'.format(i, i)))
#
# for i in range(5):
#     print(list1[i])



for i in range(5):
    exec('var{} = {}'.format(i, i+1))
names = locals()
list1 = []
for i in range(5):
    print(names.get('var' + str(i)), end=' ')
    list1.append(names.get('var' + str(i)), end=' ')
for i in range(5):
    print(list1[i])


def knn(data, y):  # knn分类器
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fm = four_metrics(y_test, y_pred)
    pred = pd.concat([pd.DataFrame(y_test.values, columns=['actual']), pd.DataFrame(y_pred, columns=['predicted'])],
                     axis=1)
    return cm, fm, pred