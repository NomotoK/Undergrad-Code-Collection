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
    print("\nK-Nearest-Neighbors Classifier")
    # Read file
    # PATH = './CW_DATA.csv'
    PATH = './NMFfeatures.csv'
    # PATH = './PCAfeatures.csv'
    train_x, test_x, train_y, test_y = load_data(PATH)
    # print(train_x, test_x, train_y, test_y)

    # Fit model with training data
    classifier = GridSearchCV(
        estimator=KNeighborsClassifier(weights='distance'),
        param_grid={
            'n_neighbors': range(2, 10),
            'p': range(2, 5)
        },
        cv=3, n_jobs=-1, refit=True
    )
    classifier.fit(train_x, train_y.ravel())
    joblib.dump(classifier, "KNN.kpl")  # Save the model
    # Test and show the accuracy
    def show_accuracy(y_hat, y_test, param):
        pass
    print("Best parameter: ", classifier.best_params_)
    print("K-Nearest-Neighbors-Accuracy on train set is: %.2f%%" % (classifier.score(train_x, train_y) * 100))
    y_hat = classifier.predict(train_x)
    show_accuracy(y_hat, train_y, 'Train set')
    print("K-Nearest-Neighbors-Accuracy on test set is: %.2f%%" % (classifier.score(test_x, test_y) * 100))
    y_hat = classifier.predict(test_x)
    show_accuracy(y_hat, test_y, 'Test set')

KNN()