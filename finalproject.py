# AUTHOR: Jonathan Nguyen & Austin Porter
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def main():
    label_encoder = preprocessing.LabelEncoder()

    column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']

    labal_encoder_columns = ['workclass', 'education', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'native-country', 'income']

    census_data = pd.read_csv('adult.data', sep=',')
    census_data.columns = column_labels

    for x in labal_encoder_columns:
        census_data[x] = label_encoder.fit_transform(census_data[x])
        census_data[x].unique()

    census_train = pd.read_csv('adult.test', sep=',', skiprows=1)
    census_train.columns = column_labels

    for x in labal_encoder_columns:
        census_train[x] = label_encoder.fit_transform(census_train[x])
        census_train[x].unique()

    census_array = census_data.to_numpy()
    census_data_array = census_train.to_numpy()

    X = census_array[:, :14]
    y = census_array[:, 14]
    X_train = census_array[:, :14]
    y_train = census_array[:, 14]
    """
    # did not work
    rand_state = 101
    lr = LinearRegression()
    data = np.zeros((rand_state, 3))
    for k in range(rand_state):
        lrX_train, lrX_test, lrY_train, lrY_test = train_test_split(X, y, test_size=0.20, random_state=k)
        lr.fit(lrX_train, lrY_train)

        test_score = lr.score(lrX_test, lrY_test)
        train_score = lr.score(lrX_train, lrY_train)

        data[k][0] = k
        data[k][1] = test_score
        data[k][2] = train_score
    print(data)
    plt.plot(data[:, 0], data[:, 1], label="test")
    plt.plot(data[:, 0], data[:, 2], label="train")
    plt.legend()
    plt.show()
    """
    scalerX = preprocessing.StandardScaler().fit(X)
    X = scalerX.transform(X)

    scaler_train = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler_train.transform(X_train)

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # above was used for testing on the data set not needed when testing dataset

    vals = list(range(90, 135))
    train = []
    test = []

    for n in vals:
        mlp1 = MLPClassifier(hidden_layer_sizes=n, max_iter=2000, random_state=1).fit(X_train, y_train)  # x_train
        train.append(mlp1.score(X_train, y_train))
        # change to (x_train, y_train) when using only .dataset and
        # change to (X_train, y_train when using the .test data

        test.append(mlp1.score(X, y))
        # change to (x_test, y_test when using only .data dataset and
        # change to (X, y) when using the .test data

    plt.plot(vals, train, label='train')
    plt.plot(vals, test, label='test')
    plt.legend()
    plt.show()

    alphas = np.linspace(.001, 0.01, 50)
    scores = []
    for a in alphas:
        mlp2 = MLPClassifier(hidden_layer_sizes=129, max_iter=3000, alpha=a, random_state=1).fit(X, y)
        temp = cross_val_score(mlp2, X, y)
        scores.append(temp.mean())
    plt.plot(alphas, scores)
    plt.show()

    """
        USING ONLY DATA SET
        set hidden layer size to 10
        optimal alpha value is 0.0398 with 85 percent accuracy

        USING TEST AND DATA SET
        hidden layer is 129
        optimal alpha value is 0.0449 85 percent accurate
    """


if __name__ == "__main__":
    main()
