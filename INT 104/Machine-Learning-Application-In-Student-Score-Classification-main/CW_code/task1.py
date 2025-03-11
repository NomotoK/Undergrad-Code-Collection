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

def read_CSV(path):
    data = pd.read_csv(path, sep=',', header=0)
    data.drop(labels="ID", axis=1, inplace=True) #remove ID column
    print("read_file: " + path)
    return data


def checkNull(data):
    if data[data.isnull().T.any()].empty:
        print("No null attributes detected")

    else:
        print("Null attributes detected")
        return data[data.isnull().T.any()]


def reduction(data):
    data_1 = data.loc[data.Programme == 1, :].iloc[:, :-1]
    data_1 = data_1.sample(n=26+13, replace=True, axis=0, random_state=8)
    data_2 = data.loc[data.Programme == 2, :].iloc[:, :-1]
    data_2 = data_2.sample(n=26+13, replace=True, axis=0, random_state=8)
    data_3 = data.loc[data.Programme == 3, :].iloc[:, :-1]
    data_3 = data_3.sample(n=28, replace=True, axis=0, random_state=8) #totally is 28
    data_4 = data.loc[data.Programme == 4, :].iloc[:, :-1]
    data_4 = data_4.sample(n=26+13, replace=True, axis=0, random_state=8)
    newdataset = pd.concat([data_1, data_2, data_3, data_4], ignore_index=True, axis=0)

    return newdataset


def data_cleaning(data):
    data.drop(data[data['Programme'] == 0].index, inplace=True)  # remove major 0
    data.drop(data[data['Programme'].isnull()].index, inplace=True)  # remove empty rows
    data.drop_duplicates()#keep=False, subset=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], ignore_index=True , inplace=True)  # remove noises

    return data


def count_programme(data):
    # count only programme 1 to 4
    programme_count = data['Programme'].value_counts()
    # plt.figure(figsize = (10, 8))
    plt.title('Programme Count')
    for i in range(1, 5):
        plt.bar(str(i), programme_count[i])
    plt.show()
    print("Count members of each major")
    print("There are " + str(len(data.loc[data.Programme == 1, :])) + "student in Programme 1")
    print("There are " + str(len(data.loc[data.Programme == 2, :])) + "student in Programme 2")
    print("There are " + str(len(data.loc[data.Programme == 3, :])) + "student in Programme 3")
    print("There are " + str(len(data.loc[data.Programme == 4, :])) + "student in Programme 4")


def count_eachMajor(data):
    # remember: count before data cleaning
    data_0 = data.loc[data.Programme == 0, :].iloc[:, :-1]
    data_1 = data.loc[data.Programme == 1, :].iloc[:, :-1]
    data_2 = data.loc[data.Programme == 2, :].iloc[:, :-1]
    data_3 = data.loc[data.Programme == 3, :].iloc[:, :-1]
    data_4 = data.loc[data.Programme == 4, :].iloc[:, :-1]
    return data_0.values, data_1.values, data_2.values, data_3.values, data_4.values


def count_normalValueCharacter(Majors):
    Major_0 = Majors[0]
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    plt.figure()
    plt.title('Mean Score of Each Major')
    plt.scatter(np.array(Major_0).mean(-1), np.zeros(len(Major_0)))
    plt.scatter(np.array(Major_1).mean(-1), np.zeros(len(Major_1)) + 1)
    plt.scatter(np.array(Major_2).mean(-1), np.zeros(len(Major_2)) + 2)
    plt.scatter(np.array(Major_3).mean(-1), np.zeros(len(Major_3)) + 3)
    plt.scatter(np.array(Major_4).mean(-1), np.zeros(len(Major_4)) + 4)
    plt.show()

    plt.figure()
    plt.title('Standard Score of Each Major')
    plt.scatter(np.array(Major_0).mean(-1), np.zeros(len(Major_0)))
    plt.scatter(np.array(Major_1).mean(-1), np.zeros(len(Major_1)) + 1)
    plt.scatter(np.array(Major_2).mean(-1), np.zeros(len(Major_2)) + 2)
    plt.scatter(np.array(Major_3).mean(-1), np.zeros(len(Major_3)) + 3)
    plt.scatter(np.array(Major_4).mean(-1), np.zeros(len(Major_4)) + 4)
    plt.show()

    plt.figure()
    plt.title('Max Score of Each Major')
    plt.scatter(np.array(Major_0).mean(-1), np.zeros(len(Major_0)))
    plt.scatter(np.array(Major_1).mean(-1), np.zeros(len(Major_1)) + 1)
    plt.scatter(np.array(Major_2).mean(-1), np.zeros(len(Major_2)) + 2)
    plt.scatter(np.array(Major_3).mean(-1), np.zeros(len(Major_3)) + 3)
    plt.scatter(np.array(Major_4).mean(-1), np.zeros(len(Major_4)) + 4)
    plt.show()

    plt.figure()
    plt.title('Min Score of Each Major')
    plt.scatter(np.array(Major_0).mean(-1), np.zeros(len(Major_0)))
    plt.scatter(np.array(Major_1).mean(-1), np.zeros(len(Major_1)) + 1)
    plt.scatter(np.array(Major_2).mean(-1), np.zeros(len(Major_2)) + 2)
    plt.scatter(np.array(Major_3).mean(-1), np.zeros(len(Major_3)) + 3)
    plt.scatter(np.array(Major_4).mean(-1), np.zeros(len(Major_4)) + 4)
    plt.show()


def get_TSNE(Majors):
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    plt.figure()
    plt.title('T-SNE')
    tsen = TSNE(n_components=2)
    dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0) #reunion the numpy array
    scaler = MinMaxScaler() #standardscaler MinMaxScaler RobustScaler
    dataset = scaler.fit_transform(dataset)
    tsen.fit_transform(dataset)
    dataset = tsen.embedding_  #the insert vector
    for i in range(len(Major_1)): #elements ahead belong to major 1 because data are unioned in line
        plt.scatter(dataset[i][0], dataset[i][1], alpha=0.5, c='turquoise')
    for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
        plt.scatter(dataset[i][0], dataset[i][1], alpha=0.5, c='salmon')
    for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
        plt.scatter(dataset[i][0], dataset[i][1], alpha=0.5, c='gold')
    for i in range(len(Major_1) + len(Major_2) + len(Major_3),
                   len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
        plt.scatter(dataset[i][0], dataset[i][1], alpha=0.5, c='violet')
    plt.show()


def getPCAHandwriting(Majors):
    # coodinate with method below
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    P1 = get_PCA2(Major_1)
    P2 = get_PCA2(Major_2)
    P3 = get_PCA2(Major_3)
    P4 = get_PCA2(Major_4)
    plt.figure()
    plt.title('PCA(no pre-processing)')
    for i in range(len(P1)):
        plt.scatter(P1[i][0], P1[i][1], alpha=0.5, c='turquoise')
    for i in range(len(P2)):
        plt.scatter(P2[i][0], P2[i][1], alpha=0.5, c='salmon')
    for i in range(len(P3)):
        plt.scatter(P3[i][0], P3[i][1], alpha=0.5, c='gold')
    for i in range(len(P4)):
        plt.scatter(P4[i][0], P4[i][1], alpha=0.5, c='violet')
    plt.show()


def get_PCA2(Majors):
    k=2 # k is the components
    # Count the mean
    n_samples, n_features = Majors.shape
    mean = np.array([np.mean(Majors[:, i]) for i in range(n_features)])
    # Normalization
    norm_X = Majors - mean
    # Get Scatter Matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # Sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # Select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # Get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data


def get_PCA(Majors):
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0)  # reunion the numpy array
    scaler = MinMaxScaler()  # StandardScaler MinMaxScaler RobustScaler
    dataset = scaler.fit_transform(dataset)
    # pca1 = PCA(n_components=4)
    pca2 = PCA(n_components=2)
    # X_pca = pca1.fit_transform(dataset)
    # var_ratio = pca1.explained_variance_ratio_
    # plt.figure()
    # plt.bar([1, 2, 3, 4], var_ratio)
    # plt.xticks([1, 2, 3, 4], ['PC1', 'PC2', 'PC3', 'PC4'])
    # plt.ylabel("variance ratio ")
    # plt.show()
    #
    # plt.figure()
    # plt.title("Score")
    # plt.scatter(new\_data[:, 0], new\_data[:, 1], marker = &  # 039;o&#039;,c=score>-100)
    # plt.show()

    dataset_reduced = pca2.fit_transform(dataset)
    score = pca2.score_samples(dataset)
    plt.figure()
    plt.title("score distribution")
    plt.scatter(range(len(score)), score, c='salmon')
    plt.show()


    # plt.figure()
    # plt.title('PCA(bias removed)')
    # plt.scatter(dataset_reduced[:, 0], dataset_reduced[:, 1], marker='o', c=score < -4 )
    # plt.show()


    plt.figure()
    plt.title('PCA')
    for i in range(len(Major_1)): #elements ahead belong to major 1 because data are unioned in line
        plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='turquoise')
    for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
        plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='salmon')
    for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
        plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='gold')
    for i in range(len(Major_1) + len(Major_2) + len(Major_3),
                   len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
        plt.scatter(dataset_reduced[i][0], dataset_reduced[i][1], alpha=0.7, c='violet')
    plt.savefig('./PCA.jpg')  # Save the pic of the PCA
    plt.show()


    DF1 = pd.DataFrame(dataset_reduced[0:len(Major_1),:])
    DF1['Programme'] = 1
    DF2 = pd.DataFrame(dataset_reduced[len(Major_1):len(Major_1) + len(Major_2),:])
    DF2['Programme'] = 2
    DF3 = pd.DataFrame(dataset_reduced[len(Major_1) + len(Major_2):len(Major_1) + len(Major_2) + len(Major_3),:])
    DF3['Programme'] = 3
    DF4 = pd.DataFrame(dataset_reduced[len(Major_1) + len(Major_2) + len(Major_3):len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4),:])
    DF4['Programme'] = 4
    Dataset = pd.concat([DF1, DF2, DF3, DF4], axis=0, join='inner')
    Dataset.to_csv('./PCAfeatures.csv', index=False, header=True)  # Ignore index and header


def getLDAhandwriting(Majors):

    # Split the data to data and labels
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0)  # reunion the numpy array
    label_1 = np.zeros(len(Major_1)) + 1  # to build a label matrix
    label_2 = np.zeros(len(Major_2)) + 2
    label_3 = np.zeros(len(Major_3)) + 3
    label_4 = np.zeros(len(Major_4)) + 4
    labels = np.concatenate([label_1, label_2, label_3, label_4], axis=0)

    # Standardize the feature
    sc = StandardScaler()
    data_std = sc.fit_transform(dataset)

    # Count mean vectors
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(data_std[labels == label], axis=0))

    # Count SW matrix
    k = 5 # k is the original dimension
    Sw = np.zeros((k, k))
    for label, mv in zip(range(1, 4), mean_vecs):
        Si = np.cov(data_std[labels == label].T)
        Sw += Si

    # Count Sb matrix
    mean_all = np.mean(data_std, axis=0)
    Sb = np.zeros((k, k))
    for i, col_mv in enumerate(mean_vecs):
        n = dataset[labels == i + 1, :].shape[0]
        col_mv = col_mv.reshape(k, 1)  # column mean vector
        mean_all = mean_all.reshape(k, 1)
        Sb += n * (col_mv - mean_all).dot((col_mv - mean_all).T)

    # Count Generalized Eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # Transform matrix
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
    afterlda = data_std.dot(w)
    plt.title('LDA-2')
    for i in range(len(Major_1)):  #elements ahead belong to major 1 because data are unioned in line
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='turquoise')
    for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='salmon')
    for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='gold')
    for i in range(len(Major_1) + len(Major_2) + len(Major_3),
                   len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='violet')
    plt.show()


def get_LDA(Majors):
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    plt.figure()
    plt.title('LDA-1')
    lda = LDA(n_components=2)
    dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0) #reunion the numpy array
    scaler = MinMaxScaler() #standardscaler MinMaxScaler RobustScaler
    dataset = scaler.fit_transform(dataset)
    Major_1_y = np.zeros(len(Major_1)) + 1 # to build a labels matrix
    Major_2_y = np.zeros(len(Major_2)) + 2
    Major_3_y = np.zeros(len(Major_3)) + 3
    Major_4_y = np.zeros(len(Major_4)) + 4
    normalset = np.concatenate([Major_1_y, Major_2_y, Major_3_y, Major_4_y], axis=0) #union

    # TF_IDF_vectorizer = TfidfVectorizer()
    # TF_IDF = TF_IDF_vectorizer.fit_transform(dataset.tolist())
    # feature_names = TF_IDF_vectorizer.get_feature_names()

    afterlda = lda.fit_transform(dataset, normalset) #LDA is a supervised method and requires the use of labels

    for i in range(len(Major_1)):  #elements ahead belong to major 1 because data are unioned in line
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='turquoise')
    for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='salmon')
    for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='gold')
    for i in range(len(Major_1) + len(Major_2) + len(Major_3),
                   len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
        plt.scatter(afterlda[i][0], afterlda[i][1], alpha=0.5, c='violet')
    plt.show()


def get_NMF(Majors):
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    plt.figure()
    plt.title('NMF-1')
    nmf = NMF(n_components=2)
    dataset = np.concatenate([Major_1, Major_2, Major_3, Major_4], axis=0)  # reunion the numpy array
    scaler = MinMaxScaler()  # standardscaler MinMaxScaler RobustScaler
    dataset = scaler.fit_transform(dataset)
    W = nmf.fit_transform(dataset)
    H = nmf.components_
    # print(W)
    # print(H)
    for i in range(len(Major_1)):  # elements ahead belong to major 1 because data are unioned in line
        plt.scatter(W[i][0], W[i][1], alpha=0.5, c='turquoise')
    for i in range(len(Major_1), len(Major_1) + len(Major_2), 1):
        plt.scatter(W[i][0], W[i][1], alpha=0.5, c='salmon')
    for i in range(len(Major_1) + len(Major_2), len(Major_1) + len(Major_2) + len(Major_3), 1):
        plt.scatter(W[i][0], W[i][1], alpha=0.5, c='gold')
    for i in range(len(Major_1) + len(Major_2) + len(Major_3),
                   len(Major_1) + len(Major_2) + len(Major_3) + len(Major_4), 1):
        plt.scatter(W[i][0], W[i][1], alpha=0.5, c='violet')
    plt.show()


def getNMFhandwritting(Majors):
    #coodinate with method below
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]

    P1 = get_NMF2(Major_1)
    DF1 = pd.DataFrame(P1)
    DF1['Programme'] = 1
    P2 = get_NMF2(Major_2)
    DF2 = pd.DataFrame(P2)
    DF2['Programme'] = 2
    P3 = get_NMF2(Major_3)
    DF3 = pd.DataFrame(P3)
    DF3['Programme'] = 3
    P4 = get_NMF2(Major_4)
    DF4 = pd.DataFrame(P4)
    DF4['Programme'] = 4

    #Save the features in a csv file
    Dataset = pd.concat([DF1, DF2, DF3, DF4], axis=0, join='inner')
    Dataset.to_csv('./NMFfeatures.csv', index=False, header=True)  # Ignore index and header
    nmfdata = Dataset


    plt.figure()
    plt.title('NMF')
    for i in range(len(P1)):
        plt.scatter(P1[i][0],P1[i][1],alpha=0.5,c='turquoise')
    for i in range(len(P2)):
        plt.scatter(P2[i][0],P2[i][1],alpha=0.5,c='salmon')
    for i in range(len(P3)):
        plt.scatter(P3[i][0],P3[i][1],alpha=0.5,c='gold')
    for i in range(len(P4)):
        plt.scatter(P4[i][0],P4[i][1],alpha=0.5,c='violet')
    plt.savefig('./NMF.jpg') # Save the pic of the NMF
    plt.show()
    return nmfdata

def get_NMF2(Majors, steps=2000, alpha=0.0002, beta=0.02):
    # a hand writting NMF method (No sklearn)
    #draw back: Run too slow, Time complexity is too high
    N = len(Majors)
    M = len(Majors[0])
    K = 2
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    Q = Q.T
    for step in range(steps):
        for i in range(len(Majors)):
            for j in range(len(Majors[i])):
                  if Majors[i][j] > 0:
                     eij = Majors[i][j] - np.dot(P[i, :], Q[:, j])
                     for k in range(K):
                         P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                         Majors[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        for i in range(len(Majors)):
            for j in range(len(Majors[i])):
                if Majors[i][j] > 0:
                    e = e + pow(Majors[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    # print(P)
    # print("----------------------------------------------------------------------------")
    # print(Q.T)
    return P


def get_MDS(Majors):
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    P1 = get_MDS2(Major_1)
    P2 = get_MDS2(Major_2)
    P3 = get_MDS2(Major_3)
    P4 = get_MDS2(Major_4)
    plt.figure()
    plt.title('MDS')
    for i in range(len(P1)):
        plt.scatter(P1[i][0], P1[i][1], alpha=0.5, c='turquoise')
    for i in range(len(P2)):
        plt.scatter(P2[i][0], P2[i][1], alpha=0.5, c='salmon')
    for i in range(len(P3)):
        plt.scatter(P3[i][0], P3[i][1], alpha=0.5, c='gold')
    for i in range(len(P4)):
        plt.scatter(P4[i][0], P4[i][1], alpha=0.5, c='violet')
    plt.show()


def get_MDS2(Majors):
    k=2
    m, n = Majors.shape
    dist = np.zeros((m, m))
    disti = np.zeros(m)
    distj = np.zeros(m)
    B = np.zeros((m, m))
    for i in range(m):
        dist[i] = np.sum(np.square(Majors[i] - Majors), axis=1).reshape(1, m)
    for i in range(m):
        disti[i] = np.mean(dist[i, :])
        distj[i] = np.mean(dist[:, i])
    distij = np.mean(dist)
    for i in range(m):
        for j in range(m):
            B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
    lamda, V = np.linalg.eigh(B)
    index = np.argsort(-lamda)[:k]
    diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:k]))
    V_selected = V[:, index]
    afterMDS = V_selected.dot(diag_lamda)
    return afterMDS
    # mds = MDS(n_components=2)
    # scaler = MinMaxScaler()  # standardscaler MinMaxScaler RobustScaler
    # dataset = scaler.fit_transform(Majors)
    # afterMDS = mds.fit_transform(dataset)
    # return afterMDS


def get_ISOmap(Majors):
    Major_1 = Majors[1]
    Major_2 = Majors[2]
    Major_3 = Majors[3]
    Major_4 = Majors[4]
    P1 = get_ISOmap2(Major_1)
    P2 = get_ISOmap2(Major_2)
    P3 = get_ISOmap2(Major_3)
    P4 = get_ISOmap2(Major_4)
    plt.figure()
    plt.title('ISOmap')
    for i in range(len(P1)):
        plt.scatter(P1[i][0], P1[i][1], alpha=0.5, c='turquoise')
    for i in range(len(P2)):
        plt.scatter(P2[i][0], P2[i][1], alpha=0.5, c='salmon')
    for i in range(len(P3)):
        plt.scatter(P3[i][0], P3[i][1], alpha=0.5, c='gold')
    for i in range(len(P4)):
        plt.scatter(P4[i][0], P4[i][1], alpha=0.5, c='violet')
    plt.show()


def get_ISOmap2(Majors):
    ISOmap = Isomap(n_components=2)
    scaler = MinMaxScaler() # standardscaler MinMaxScaler RobustScaler
    dataset = scaler.fit_transform(Majors)
    after = ISOmap.fit_transform(dataset)
    return after


# data normalisation process  ---  unnecessary
# def data_norm(df,*cols):
#     df_n=df.copy()
#     for col in cols:
#         ma= df_n[col].max()
#         mi= df_n[col].min()
#         df_n[col]=(df_n[col]-mi)/(ma-mi)
#     return (df_n)
# df=data_norm(df,'Q1','Q2','Q3','Q4','Q5')


data = read_CSV('./CW_DATA.csv')
print(data.describe())
# count_eachMajor(data)
# checkNull(data)
# count_normalValueCharacter(count_eachMajor(data))

data = data_cleaning(data)
# data.describe()

features = data[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
scaler = MinMaxScaler() # normalisation process
features = scaler.fit_transform(features.values)

Dataset = pd.concat([pd.DataFrame(features[:, 0]), pd.DataFrame(features[:, 1]), pd.DataFrame(features[:, 2]), pd.DataFrame(features[:, 3]), pd.DataFrame(features[:, 4])], axis=1, join='inner')
Dataset.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
print(Dataset.corr())

figure, ax = plt.subplots(figsize=(12, 12))
# sns.heatmap(features.corr(), square=True, annot=True, ax=ax)
count_programme(data)


count_normalValueCharacter(count_eachMajor(data))

get_PCA(count_eachMajor(data))
getPCAHandwriting(count_eachMajor(data))

get_MDS(count_eachMajor(data))
get_ISOmap(count_eachMajor(data))

get_TSNE(count_eachMajor(data))

get_LDA(count_eachMajor(data))
getLDAhandwriting(count_eachMajor(data))

get_NMF(count_eachMajor(data))
getNMFhandwritting(count_eachMajor(data)) #Nice outcome

