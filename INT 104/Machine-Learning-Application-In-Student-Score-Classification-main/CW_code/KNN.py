import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets._base import load_data

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.model_selection import train_test_split



def load_data(PATH):
    # Load data (need to rewrite because label encoder requires numpy data)
    data = pd.read_csv(PATH, sep=',', header=0)
    data.drop(data[data['Programme'].isnull()].index, inplace=True)
    # dataset = data_cleaning(data) # available when using row data
    dataset = data.values # data ---> dataset when using row data
    # Divide the dataset into features and labels
    features = dataset[:, 0:2] # 2--->5 when using row data
    labels = dataset[:, 2] # 2--->5 when using row data


    # Transform the label
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(labels)
    label_encoded_y = label_encoder.transform(labels)
    # print(data)

    # Divide the data into train set and test set
    seed = 7  #seed = 7 ---> XGBoost     seed = 2 ---> SVM
    train_x, test_x, train_y, test_y = train_test_split(features, label_encoded_y, train_size=400, test_size=10)#26,13
    scaler = MinMaxScaler() # normalisation process
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x) # Test set is not necessary to fit the scaler
    return train_x, test_x, train_y, test_y


def KNN():
    # print("\nK-Nearest-Neighbors Classifier")
    # Read file
    # PATH = './CW_Data.csv'
    PATH = './NMFfeatures.csv'
    # PATH = './PCAfeatures.csv'
    train_x, test_x, train_y, test_y = load_data(PATH)
    # print(train_x, test_x, train_y, test_y)

    # Fit model with training data
    # classifier = GridSearchCV(
    #     estimator=KNeighborsClassifier(weights='distance'),
    #     param_grid={
    #         'n_neighbors': range(2, 10),
    #         'p': range(2, 5)
    #     },
    #     cv=3, n_jobs=-1, refit=True
    # )
    classifier = KNeighborsClassifier(weights='distance')
    classifier.fit(train_x, train_y)


    classifier.fit(train_x, train_y.ravel())
    joblib.dump(classifier, "KNN.kpl")  # Save the model
    y_pred = classifier.predict(test_x)

    plt.scatter(x=train_x[:, 0], y=train_x[:, 1], c=train_y, alpha=0.3)
    right = test_x[y_pred == test_y]
    wrong = test_x[y_pred != test_y]
    plt.scatter(x=right[:, 0], y=right[:, 1], color='r', marker='x', label='correct prediction')
    plt.scatter(x=wrong[:, 0], y=wrong[:, 1], color='r', marker='>', label='incorrect prediction')
    plt.legend(loc='best')
    plt.title('Visualization of KNN prediction(PCA input)')
    plt.show()

    x_min, x_max = train_x[:, 0].min() , train_x[:, 0].max()
    y_min, y_max = train_x[:, 1].min() , train_x[:, 1].max()
    h = .01  # 网格中的步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # 给不同区域赋以颜色
    cmap_bold = ListedColormap(['#FF0000', '#003300', '#0000FF'])  # 给不同属性的点赋以颜色
    Z = Z.reshape(xx.shape)


    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # 也画出所有的训练集数据
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('KNN data distribution and clustering range(PCA input)')

    plt.show()

    # Test and show the accuracy
    def show_accuracy(y_hat, y_test, param):
        pass
    # print("Best parameter: ", classifier.best_params_)
    print("K-Nearest-Neighbors-Accuracy on train set is: %.2f%%" % (classifier.score(train_x, train_y) * 100))
    y_hat = classifier.predict(train_x)
    show_accuracy(y_hat, train_y, 'Train set')
    print("K-Nearest-Neighbors-Accuracy on test set is: %.2f%%" % (classifier.score(test_x, test_y) * 100))
    y_hat = classifier.predict(test_x)
    show_accuracy(y_hat, test_y, 'Test set')
    return classifier.score(test_x, test_y) * 100

# Create color maps


KNN()
# score = []
# runtimes = 20
# for i in range (1, runtimes):
#
#     KNN(i)
#     score.append(KNN(i))
#
# plt.figure()
# plt.title('accuracy of training(KNN)')
# x1 = np.arange(1,runtimes)
# plt.plot(x1,score,'turquoise')
# plt.show()


