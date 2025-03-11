import matplotlib.pyplot
import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt


def cust_eval(preds, dtrain):
    labels = dtrain.get_label()
    return 'ACC', accuracy_score(labels, np.argmax(preds, axis=1))




def getxy():
    df = pd.read_csv('./CW_Data.csv' , sep=',', header=0)
    df.drop(df[df['Programme'].isnull()].index, inplace=True)
    data = df.values
    Major_0 = []
    Major_1 = []
    Major_2 = []
    Major_3 = []
    Major_4 = []
    for i in range(len(data)):
        if data[i, -1] == 0:
            Major_0.append(data[i, 1:-1])
        elif data[i, -1] == 1:
            Major_1.append(data[i, 1:-1])
        elif data[i, -1] == 2:
            Major_2.append(data[i, 1:-1])
        elif data[i, -1] == 3:
            Major_3.append(data[i, 1:-1])
        elif data[i, -1] == 4:
            Major_4.append(data[i, 1:-1])
    train_x = np.concatenate([Major_0,Major_1,Major_2,Major_3,Major_4], axis=0)
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    train_x = train_x.astype(np.float32)
    Major_0_y = np.zeros(len(Major_0),dtype=np.compat.long)
    Major_1_y = np.zeros(len(Major_1), dtype=np.compat.long) + 1
    Major_2_y = np.zeros(len(Major_2), dtype=np.compat.long) + 2
    Major_3_y = np.zeros(len(Major_3), dtype=np.compat.long) + 3
    Major_4_y = np.zeros(len(Major_4), dtype=np.compat.long) + 4
    train_y = np.concatenate([Major_0_y,Major_1_y,Major_2_y,Major_3_y,Major_4_y], axis=0)
    return train_x, train_y

X, Y = getxy()
accuracy = []
kf = KFold(n_splits=30, shuffle=True, random_state=100)
for train_index, test_index in kf.split(X):
    train_X = X[train_index]
    train_Y = Y[train_index]
    test_X = X[test_index]
    test_Y = Y[test_index]
    model = XGBClassifier(use_label_encoder=False)
    model.fit(train_X, train_Y, eval_metric=cust_eval, eval_set=[(test_X, test_Y)], verbose=False)
    estimator = xgboost.XGBClassifier(use_label_encoder=True,eval_metric='mlogloss')
    y_pred = model.predict(test_X)
    acc = accuracy_score(test_Y, y_pred)
    print('accuracy:',acc)
    accuracy.append(acc)

plt.figure()
plt.title('accuracy of training(XGBoost)')
x1 = np.arange(0,30)
plt.plot(x1,accuracy)
# plot_importance(model)
plt.show()




