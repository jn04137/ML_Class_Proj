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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def main():

    label_encoder = preprocessing.LabelEncoder()

    column_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']

    census_data = pd.read_csv('adult.data', sep=',')
    census_data.columns = column_labels
    census_data['workclass'] = label_encoder.fit_transform(census_data['workclass'])
    census_data['workclass'].unique()
    census_data['education'] = label_encoder.fit_transform(census_data['education'])
    census_data['education'].unique()
    census_data['marital-status'] = label_encoder.fit_transform(census_data['marital-status'])
    census_data['marital-status'].unique()
    census_data['occupation'] = label_encoder.fit_transform(census_data['occupation'])
    census_data['occupation'].unique()
    census_data['relationship'] = label_encoder.fit_transform(census_data['relationship'])
    census_data['relationship'].unique()
    census_data['race'] = label_encoder.fit_transform(census_data['race'])
    census_data['race'].unique()
    census_data['sex'] = label_encoder.fit_transform(census_data['sex'])
    census_data['sex'].unique()
    census_data['native-country'] = label_encoder.fit_transform(census_data['native-country'])
    census_data['native-country'].unique()
    census_data['income'] = label_encoder.fit_transform(census_data['income'])
    census_data['income'].unique()

    census_array = census_data.to_numpy()

    X = census_array[:, :14]
    y = census_array[:, 14]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print("Number of mislabeled points out of a total %d points: %d"
    % (X_test.shape[0], (y_test != y_pred).sum()))
    print(census_array)
    print(label_encoder.inverse_transform([1]))



if __name__ == "__main__":
    main()
