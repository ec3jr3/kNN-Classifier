from numpy import *
import operator
from os import listdir
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import importlib
from sklearn.linear_model import LogisticRegression
import joblib

#funcion para el examen
def nearest():

    dataset = pd.read_csv('data.csv')#, delimiter='\t')
    
    #normalizing the data
    
    xdata = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    xdata = xdata.apply(lambda x: (x - x.min(axis=0) ) / (x.max(axis=0) - x.min(axis=0)))
    
    #adding the column of class with the date
    xdata.loc[:, 'Outcome'] = dataset['Outcome']
    
    #SKLEARN TEST
    
    #spliting the dataset into attributes and labels
    x = xdata.iloc[:, :-1].values
    y = xdata.iloc[:, 8].values

    #train test split
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.20)

    #normalizing
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)

    #training and predictions
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(xtrain, ytrain)

    ypred = classifier.predict(xtest)

    #printing the results
    print(classification_report(ytest, ypred))
    print(confusion_matrix(ytest, ypred))
    


