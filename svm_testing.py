import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def main():

    '''
    As mentioned in sklearn documentation, 5 or 10 cross fold validation is
    preferred to leave one out.
    '''

    kf = KFold(n_splits=5)

    label_encoder = preprocessing.LabelEncoder()
    standard_scaler = preprocessing.StandardScaler()

    column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']

    label_encoder_columns = ['workclass', 'education', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'native-country', 'income']

    census_data = pd.read_csv('adult.data', sep=',')
    census_data.columns = column_labels

    # The following will use label_encoder only on columns that contain strings as values

    for x in label_encoder_columns:
        census_data[x] = label_encoder.fit_transform(census_data[x])
        census_data[x].unique()

    census_array = census_data.to_numpy()

    X = census_array[:, :14]
    y = census_array[:, 14]

    standard_scaler.fit(X, y)
    X = standard_scaler.transform(X)

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
    # clf = svm.SVC(kernel='linear', C=1, max_iter=600000).fit(X_train, y_train)
    # print("This is the score for the svm: " + str(clf.score(X_test, y_test)))
    scores = []
    c_values = np.linspace(4.0, 4.9, 51)
    count=0

    '''
    Ran into convergence warnings before removing certain features with
    max_iter set to 2,000,000.
    '''
    # for c in c_values:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    #     clf = svm.SVC(kernel='linear', C=c, max_iter=2000000)
    #     print("The current count: " + str(count))
    #     count += 1 # counter for current progress
    #     scores.append(cross_val_score(clf, X, y, n_jobs= 14, cv=kf).mean())

    plt.plot(c_values, scores)
    plt.xlabel('c_values')
    plt.ylabel('scores')
    plt.show()

if __name__ == "__main__":
    main()
