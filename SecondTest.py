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
    

def tree_decision():
    #asignando nombre a cada columna
    
   dataset = pd.read_csv('data.csv')
    
    #x
   inputs = dataset.drop('Outcome', axis='columns')
   #y
   target = dataset['Outcome']
    
   model = tree.DecisionTreeClassifier()
   model.fit(inputs, target)
   model.score(inputs, target)
    
   model = model.fit(inputs, target)
    
   plt.figure(figsize=(25,25))
    
   print (tree.plot_tree(model, filled=True))
   

def logRegression():
    
    diabetes = pd.read_csv('data.csv')
    print (diabetes.head())
    
    #DIVIDIMOS EL DATA SET EN 3 PARTES
    
    #Para el entrenamiento
    Train = diabetes[:650]
    
    #Para el Test
    Test = diabetes[650:750]
    
    #para probar el algoritmo
    Check = diabetes[750:]
    
    #separando labels y features del TEST Y TRAIN
    trainLabel = np.asarray(Train['Outcome'])
    trainData = np.asarray(Train.drop('Outcome',1))

    testLabel = np.asarray(Test['Outcome'])
    testData = np.asarray(Test.drop('Outcome',1))
    
    #Normalizando los datos
    means = np.mean(trainData, axis=0)
    stds = np.std(trainData, axis=0)
    
    trainData = (trainData - means)/stds
    testData = (testData - means)/stds

    #Training the model
    diabetesT = LogisticRegression()
    diabetesT.fit(trainData, trainLabel)
    
    #accuracy
    accuracy = diabetesT.score(testData, testLabel)
    print ("The accurary is of :", accuracy)
    
    #guardando el modelo entrenado
    joblib.dump([diabetesT, means, stds], 'diabeteseModel.pkl')
    
    #cargando el modelo
    diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
    accuracyModel = diabetesLoadedModel.score(testData, testLabel)
    
    #prediccion
    exerciseData = Check[:1]
    print (exerciseData)
    
    #quitando la coluna de outcome
    sampleDataFeatures = np.asarray(exerciseData.drop('Outcome',1))
    sampleDataFeatures = (sampleDataFeatures - means)/stds
    
    #resultado
    predictionProbability = diabetesT.predict_proba(sampleDataFeatures) #primer valor es la probabilidad de que sea 0 y el segundo de que sea 1
    
    #0 -> no diabetico
    #1 -> diabetico
    
    prediction = diabetesT.predict(sampleDataFeatures)
    
    print (predictionProbability)
    print (prediction)
    
    
#def vector():
