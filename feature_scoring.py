'''
We referenced Rahil Shaikh's Feature Selection Techniques in Machine Learning with Python
blog post. Link: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
'''

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def main():
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

    X = census_data.iloc[:, 0:14]
    y = census_data.iloc[:, 14]

    # using SelectKBest
    high_impact = SelectKBest(score_func=chi2, k=10)
    fit = high_impact.fit(X, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
    feature_scores.columns = ['Features', 'Score']

    print(feature_scores.nlargest(14, 'Score'))


if __name__ == "__main__":
    main()
