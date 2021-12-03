# AUTHOR: Jonathan Nguyen & Austin Porter
"""
DELETE THIS BEFORE SUBMITTING
Define the objectives and design the structure of the machine learning system
    Finding predictors for if someone will make over or under 50k depending on the data given by the dataset

Perform necessary preprocessing on the data set
    Split the dataset
    preprocess label using preprocessing.LabelEncoder() on x

Apply at least three different machine learning models and at least two different
model selection methods to find a good model for the data set.

Using sklearn, perform the training of the machine learning system with
combinations of the algorithms, model selections, and parameters to determine the
optimal setting

    Models: - 2
        use linear regression because why not -1
        classifier for prediction -2

    Methods: -3
        sklearn.LinearRegression
        svm,SVM???
        idk what else

Write a report documenting the project including the data, objectives, design, training
process and results

    Do this afterwards when we are done

Submit the written report, data set, source code and other related materials to folio.
ok

8. Give a presentation on your project (~ 5 minutes) at the final exam time (Dec. 6,
7:30-9:30am)
    I can really do this almost solo if you want I am typically pretty good at presentations

testing the commit and push
    """
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

    # play with alphas and test_sizes
    scalerX = preprocessing.StandardScaler().fit(X)
    X = scalerX.transform(X)

    scaler_train = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler_train.transform(X_train)

    # x_tra1in, x_test, y_tra1in, y_test = train_t1est_split(X, y, test_size=0.33, random_state=1)
    # above was used for testing on the data set not needed when testing dataset

    vals = list(range(90, 135))
    train = []
    test = []
    """
    for n in vals:
        mlp1 = MLPClassifier(hidden_layer_sizes=n, max_iter=2000, random_state=1).fit(X_train, y_train) # x_train
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
    """
    # optimal HLS is 10
    """
        USING ONLY DATA SET
        set hidden layer size to 10
        find the optimal alpha value
        
        USING TEST AND DATA SET
        hidden layer is 129
    """

    alphas = np.linspace(.1, 20, 75)
    scores = []
    for a in alphas:
        mlp2 = MLPClassifier(hidden_layer_sizes=129, max_iter=3000, alpha=a, random_state=1).fit(X, y)
        temp = cross_val_score(mlp2, X, y)
        scores.append(temp.mean())
    plt.plot(alphas, scores)
    plt.show()
    # TEST ONLY optimal point is 0.0398 about



if __name__ == "__main__":
    main()
