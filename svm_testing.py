import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Got tired of processing times and
# found modules that support GPU acceleration
# from thundersvm import SVC
from sklearn import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def main():

    # kf = KFold(n_splits=10)

    label_encoder = preprocessing.LabelEncoder()
    standard_scaler = preprocessing.StandardScaler()

    column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']

    label_encoder_columns = ['workclass', 'education', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'native-country', 'income']

    # feature_scoring results suggest that we should exclude native-country first
    census_data = pd.read_csv('adult.data', sep=',')
    census_data.columns = column_labels

    census_test_data = pd.read_csv('adult.test', sep=',', skiprows=1)
    census_test_data.columns = column_labels


    # The following will use label_encoder only on columns that contain strings as values

    for x in label_encoder_columns:
        census_data[x] = label_encoder.fit_transform(census_data[x])
        census_data[x].unique()
        census_test_data[x] = label_encoder.fit_transform(census_test_data[x])
        census_test_data[x].unique()

    census_data = census_data.drop(
        columns=['age'],
        axis=1
    )
    census_test_data = census_test_data.drop(
        columns=['age'],
        axis=1
    )

    print(census_data)

    census_array = census_data.to_numpy()
    census_test_array = census_test_data.to_numpy()

    X = census_array[:, :13]
    y = census_array[:, 13]

    X_test_data = census_test_array[:, :13]
    y_test_data = census_test_array[:, 13]

    ss = standard_scaler.fit(X, y)

    X = ss.transform(X)
    X_test_data = ss.transform(X_test_data)

    # For linear kernel w/ StandardScaler:
    # max_iter=3,000; test_score=0.4902; convergence warning
    # max_iter=6,000; test_score=0.6107; convergence warning
    # max_iter=10,000; test_score=0.7619; convergence warning
    # max_iter=20,000; test_score=0.8274; convergence warning
    # max_iter=100,000; test_score=0.8192; convergence warning
    # max_iter=200,000; test_score=0.8166; convergence warning
    # max_iter=600,000; test_score=0.8164
    # max_iter=700,000; test_score=0.8164
    # max_iter=800,000; test_score=0.8164
    # max_iter=1,000,000; test_score=0.8164; convergence warning when running in cross_val_score
    # max_iter=10,000,000; test_score=0.8164
    # max_iter=20,000,000; test_score=0.8164

    # The following was used to find an adequate max_iter value
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # print("This is the score for the svm: " + str(clf.score(X_test, y_test)))

    # scores = []
    # c_values = np.linspace(0.01, 1.0, 101)
    # count=0

    '''
    As mentioned in sklearn documentation, 5 or 10 cross fold validation is
    preferred to leave one out.
    '''

    '''
    Ran into convergence warnings before removing certain features with
    max_iter set to 2,000,000.

    No convergence warnings after removing the age column.

    Removing age and workclass reduces accuracy to 0.467

    Depending on what the value of C is, it can be more or less prone
    to give convergence warnings.
    '''

    ''' The following was used to find the optimal c value which is 0.09 '''

    # for c in c_values:
    #     clf = SVC(kernel='linear', C=c)
    #     print("The current count: " + str(count))
    #     count += 1 # counter for current progress
    #     # scores.append(cross_val_score(clf, X, y, n_jobs= 14, cv=kf).mean())
    #     scores.append(cross_val_score(clf, X, y, n_jobs= 14, cv=5).mean())

    # plt.plot(c_values, scores)
    # plt.xlabel('c_values')
    # plt.ylabel('scores')
    # plt.show()

    train_scores = []
    test_scores = []
    test_set = []
    r_state = range(0, 100)
    for r in r_state:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=r)
        clf = SVC(kernel='linear', C=0.09)
        clf.fit(X_train, y_train) # This is all values in the adult.data dataset
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        test_set.append(clf.score(X_test_data, y_test_data))


    plt.plot(r_state, test_scores, label='test_scores')
    plt.plot(r_state, train_scores, label='train_scores')
    plt.plot(r_state, test_set, label='test_set_scores')
    plt.xlabel('random_state')
    plt.xlabel('scores')
    plt.title('Test Scoring with external Test Set')
    plt.legend()
    plt.show()

    '''
    no modifications, we found that:
    mean of train_scores: 0.8146362594545036
    mean of test_scores: 0.8150767798976267
    mean of test_set: 0.8059275184275185

    Without the native-country or race, we found that:
    mean of train_scores: 0.8145610818244323
    mean of test_scores: 0.8150311772917637
    mean of test_set: 0.8126947174447174

    Without native-country, race, workclass, education, we found that:
    mean of train_scores: 0.8145569562227827
    mean of test_scores: 0.8150190786412288
    mean of test_set: 0.8127414004914009

    Without age, we found that:
    mean of train_scores: 0.8100453816181522
    mean of test_scores: 0.8105062819916243
    mean of test_set: 0.8102518427518427
    '''

    print("The mean of means for train_scores:" + str(sum(train_scores) / len(train_scores)))
    print("The mean of means for test_scores:" + str(sum(test_scores) / len(test_scores)))
    print("The mean of means for test_data_scores:" + str(sum(test_set) / len(test_set)))
if __name__ == "__main__":
    main()
