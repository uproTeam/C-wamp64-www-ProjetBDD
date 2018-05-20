import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
from numpy import *
import pickle #bilbio pour exporter modele
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
#https://www.kaggle.com/uciml/pima-indians-diabetes-database/data

# User INterface
root = tk.Tk()
root.geometry("500x400+350+450")
root.title("Diabete Analytics")
root['bg'] = 'blue'


def hello():
    #for i in range(0,8):
            #print(tabPregnancies[1])
            print("Hello world")
texte1 = Label(root,text='Veuillez remplir les informations')
texte1['bg'] = 'blue'
texte1.pack()

#tableau une ligne
rows=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age', 'Outcame']
#Tab=[[IntVar() for i in range(len(rows))]]
#saisieDonne = np.zeros((1), dtype='i')
#tabPregnancies = np.zeros((0), dtype='i')
#tabGlucose = np.zeros((1), dtype='i')
#tabBloodPressure = np.zeros((2), dtype='i')
#tabSkinThickness = np.zeros((3), dtype='i')

#tabInsulin = np.zeros((4), dtype='i')
#tabBMI = np.zeros((5), dtype='i')
#tabDiabetesPedigreeFunction = np.zeros((6), dtype='i')
#tabAge = np.zeros((7), dtype='i')
#tabOutcome = np.zeros((8), dtype='i')
#tableau_de_zero = np.zeros((2, 3), dtype='i')

#saisieDonne=[[IntVar() for i in range(len(rows))]]
saisieDonne = IntVar()
Pregnancies = Entry(root,textvariable=saisieDonne).pack()
Glucose = Entry(root,textvariable=saisieDonne).pack()
BloodPressure = Entry(root,textvariable=saisieDonne).pack()
SkinThickness = Entry(root,textvariable=saisieDonne).pack()

Insulin = Entry(root,textvariable=saisieDonne).pack()
BMI = Entry(root,textvariable=saisieDonne).pack()
DiabetesPedigreeFunction = Entry(root,textvariable=saisieDonne).pack()
Age = Entry(root,textvariable=saisieDonne).pack()
Outcome = Entry(root,textvariable=saisieDonne).pack()
bouton1= Button(root,text="predire",command=hello).place(x='225',y='350')


dataset = pd.read_csv("diabetes.csv")

dataset.head();

#columns_target = ["Outcome"]
#columns_train= ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age" ]

#X = dataset[columns_target]
#Y = dataset[columns_train]

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)




print(dataset.describe())
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]


zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']





for column in zero_not_accepted:
   X[column] = X[column].replace(0, np.NaN)
   mean = int(X[column].mean(skipna=True))
   X[column] = X[column].replace(np.NaN, mean)

#Split des donnees pour la construcction du modele et son test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)


y_test.isnull().sum()

classifier = SVC(random_state=0, kernel='rbf')
classifier.fit(X_train, y_train)
print("Score avec svm : ")
print(classifier.score(X_test,y_test))




#Avec : logistic Regression
print("Avec Logistic Regression : ")
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
print(classifier.score(X_test, y_test))


#Random Forest
print("Avec Random Forest : ")
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=2, random_state=0);
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
print("Classement features par ordre d'importance")
print(pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False))
print("Random Forest en gardant les meilleurs features")


test = [[1,189,60,23,846,30.1,0.39,57]]
print(classifier.predict(test))
