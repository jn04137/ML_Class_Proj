import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

def main():

    loo = LeaveOneOut()

    label_encoder = preprocessing.LabelEncoder()

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    scores = []
    for x in np.linspace(0.0000001, 0.00002, 101):
        gnb = GaussianNB(var_smoothing=x)
        #cvs = cross_val_score(gnb, X, y, n_jobs=-1, cv=loo).mean()
        cvs = cross_val_score(gnb, X, y, n_jobs=-1, cv=10).mean()
        scores.append(cvs)

    # the optimal x value found in the code about was x = 0.00001 with cv=10
    # the optimal x value found in the code about was x = 0.00001085 with cv=loo
    var_smoothing_vals = np.linspace(0.0000001, 0.00002, 101)
    plt.plot(var_smoothing_vals, scores)
    plt.xlabel('var_smoothing_vals')
    plt.ylabel('scores')
    plt.show()

if __name__ == "__main__":
    main()
