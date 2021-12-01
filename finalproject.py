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
from sklearn import preprocessing


def main():
    heist_data = np.genfromtxt('adult.data', delimiter=',', skip_header=1)
    # print(heist_data)
    print("Hello world!")
    x = heist_data[:, 0:3]
    y = heist_data[:, 13]
    le = preprocessing.LabelEncoder()
    # le_X = preprocessing.LabelEncoder()
    # leY = preprocessing.LabelEncoder()
    # print(x)
    # print(y)
    print("Hello World again")
    test_list = ["Atlanta", "NYC", "Atlanta", "Chicago"]
    # le.fit(test_list)
    example_list = le.transform(test_list) # cannot get to work can try to figure out
    print(*example_list)

    # le.classes_

    # le_X.fit(x)
    # leY.fit(y)

    # print(x)


if __name__ == "__main__":
    main()
