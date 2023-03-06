# Hand-written digit 2 recognition in MNIST dataset
## _Applying Logistic Regression_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Hand-written digit 2 recognition is a supervised machine learning model which is employ to recognize the digit 2 written in mnist dataset and gives the result as true/false. It gives a plot curve which shows the precision-recall of this model.

 ✨ MACHINE LEARNING MODEL WHICH VERIFIES THE DIGITS IN MNIST DATASET ✨

## Features

- Random number is given to the model
- Recognizes the digits based on its learning and provides the result
- If the digit is 2 it returns true otherwise false.


This model takes random digit of mnist dataset provided and produces the result on boolean datatype.
As [MNIST dataset] writes on the [Wikipedia][df1]

> The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems The database is also widely used for training and testing in the field of machine learning.


## Tech

Hand-written digit 2 recognition model uses a number of libraries to work properly:

- [from sklearn.datasets import fetch_openml] - To get the MNIST dataset.
- [from sklearn.linear_model import LogisticRegression] - Applying Logistic Regression model to get boolean result
- [from sklearn.model_selection import cross_val_score] - By cross validation checking the accuracy 
- [import matplotlib] - To get the features of matplotlib library as imshow()
- [import matplotlib.pyplot as plt] - plot the curve to see the precision-recall curve.
- [import numpy as np] - handling the multi dimensional array in mnist dataset.
- [from sklearn.model_selection import cross_val_predict] - Evaluating the classifier
- [from sklearn.metrics import confusion_matrix] - Confusion matrix calculation
- [from sklearn.metrics import precision_score, recall_score, f1_score] - to get the precision, recall, f1 score of the classifier
- [from sklearn.metrics import precision_recall_curve] - to get the curve of precision-recall.
Hand-written digit 2 recognition itself is open source with a [public repository][hnd] on GitHub.


## Conclusion
Using scikit-learn library created a handwritten digit recognizer algorithm working on MNIST dataset with f1-score of approximately 0.89.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   
   [MNIST dataset]: <https://en.wikipedia.org/wiki/MNIST_database>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [hnd]: <https://github.com/kondapalli19/-Hand-written-digit-2-recognition-in-MNIST-dataset>
