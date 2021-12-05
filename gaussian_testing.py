import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

def main():

    # loo = LeaveOneOut()

    label_encoder = preprocessing.LabelEncoder()

    column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']

    label_encoder_columns = ['workclass', 'education', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'native-country', 'income']

    census_data = pd.read_csv('adult.data', sep=',')
    census_data.columns = column_labels

    census_test_data = pd.read_csv('adult.test', sep=',', skiprows=1)
    census_test_data.columns = column_labels

    # The following will use label_encoder only on columns that contain strings as values

    for x in label_encoder_columns:
        census_data[x] = label_encoder.fit_transform(census_data[x])
        census_test_data[x] = label_encoder.fit_transform(census_test_data[x])

    census_data = census_data.drop(
        columns=['age'],
        axis=1
    )
    census_test_data = census_test_data.drop(
        columns=['age'],
        axis=1
    )

    print(census_test_data)

    census_array = census_data.to_numpy()
    census_test_array = census_test_data.to_numpy()

    X = census_array[:, :13]
    y = census_array[:, 13]

    X_test_data = census_test_array[:, :13]
    y_test_data = census_test_array[:, 13]

    # scores = []
    # for x in np.linspace(0.0000001, 0.00002, 101):
    #     gnb = GaussianNB(var_smoothing=x)
    #     #cvs = cross_val_score(gnb, X, y, n_jobs=-1, cv=loo).mean()
    #     cvs = cross_val_score(gnb, X, y, n_jobs=-1, cv=10).mean()
    #     scores.append(cvs)

    # the optimal x value found in the code about was x = 0.00001085 with cv=10
    # the optimal x value found in the code about was x = 0.00001085 with cv=loo

    gnb = GaussianNB(var_smoothing = 0.00001085)

    train_scores = []
    test_scores = []
    test_set = []

    r_state = range(0, 100)
    for r in r_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=r)
        gnb = GaussianNB(var_smoothing = 0.00001085)
        gnb.fit(X_train, y_train)
        train_scores.append(gnb.score(X_train, y_train))
        test_scores.append(gnb.score(X_test, y_test))
        test_set.append(gnb.score(X_test_data, y_test_data))

    # var_smoothing_vals = np.linspace(0.0000001, 0.00002, 101)
    # plt.plot(var_smoothing_vals, scores)
    # plt.xlabel('var_smoothing_vals')
    # plt.ylabel('scores')
    # plt.show()

    '''
    no modifications, we found that:
    mean of train_scores: 0.8053637405454961
    mean of test_scores: 0.8057971149371795
    mean of test_set: 0.8127444717444718

    Without the native-country or race, we found that:
    mean of train_scores: 0.8053637405454961
    mean of test_scores: 0.8057971149371795
    mean of test_set: 0.8059269041769044

    Without native-country, race, workclass, education, we found that:
    mean of train_scores: 0.8053641989456795
    mean of test_scores: 0.8057971149371795
    mean of test_set: 0.8059275184275185

    Without age, we found that:
    mean of train_scores: 0.8053678661471463
    mean of test_scores: 0.8057961842717538
    mean of test_set: 0.8105062819916243
    '''

    print("The mean of means for train_scores:" + str(sum(train_scores) / len(train_scores)))
    print("The mean of means for test_scores:" + str(sum(test_scores) / len(test_scores)))
    print("The mean of means for test_data_scores:" + str(sum(test_set) / len(test_set)))

    plt.plot(r_state, train_scores, label="train_scores")
    plt.plot(r_state, test_scores, label="test_scores")
    plt.plot(r_state, test_set, label="test_set_scores")
    plt.xlabel('random states')
    plt.ylabel('scores')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
