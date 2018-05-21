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
#Begin of Kaggle Script


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


#test = [[1,189,60,23,846,30.1,0.39,57]]
#print(classifier.predict(test))

#End of kaggle Script
# User INterface
root = tk.Tk()
root.geometry("500x400+350+450")
root.title("Diabete Analytics")
#root['bg'] = 'blue'

def predire():
    #for i in range(0,8):
            #print(tabPregnancies[1])
            test = [[xPregnancies.get(),xGlucose.get(),xBloodPressure.get(),xSkinThickness.get(),xInsulin.get(),xBMI.get(),xDiabetesPedigreeFunction.get(),xAge.get()]]
            result = classifier.predict(test)
            print(result)
            if(result == 1):
                print('Vous avez le diabete')
                affichage = Label(root,text='Vous avez le diabete')
            else:
                print("Go McDo")
                affichage = Label(root,text='Go McDo',font=("Helvetica",16))
            affichage.place(x='225',y='280')
            #test['bg'] = 'blue'
texte1 = Label(root,text='Veuillez remplir les informations',font=("Helvetica",12))
#texte1['bg'] = 'blue'
texte1.pack()


#tableau une ligne
rows=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age', 'Outcame']
#Tab=[[IntVar() for i in range(len(rows))]]
#saisieDonne = np.zeros((1), dtype='i')
#tabPregnancies = np.zeros((0), dtype='i')

#tableau_de_zero = np.zeros((2, 3), dtype='i')

#saisieDonne=[[IntVar() for i in range(len(rows))]]
xPregnancies = DoubleVar()
lPregnancies = Label(root,text='Pregnancies :')
lPregnancies.place(x='25',y='20')
lGlucose = Label(root, text='Glucose :')
lBloodPressure = Label(root,text='BloodPressure :')
lSkinThickness = Label(root,text='SkinThicknes :')

lInsulin = Label(root,text='Insulin :')
lBMI = Label(root,text='BMI :')
lDiabetesPedigreeFunction = Label(root,text='DiabetesPedigreeFunction :')
lAge = Label(root,text='Age :')

#Packing Label
lGlucose.place(x='25', y ='40')
lBloodPressure.place(x='25',y='60')
lSkinThickness.place(x='25',y='80')
lInsulin.place(x='25',y='100')
lBMI.place(x='25',y='120')
lDiabetesPedigreeFunction.place(x='25',y='140')
lAge.place(x='25',y='160')

#Variables des champs de saisies
xBloodPressure = DoubleVar()
xSkinThickness = DoubleVar()

xInsulin = DoubleVar()
xBMI = DoubleVar()
xDiabetesPedigreeFunction = DoubleVar()
xAge = DoubleVar()

Pregnancies = Entry(root,textvariable=xPregnancies)
Pregnancies.pack()
xGlucose = DoubleVar()
Glucose = Entry(root,textvariable=xGlucose).pack()
xBloodPressure = DoubleVar()
BloodPressure = Entry(root,textvariable=xBloodPressure).pack()
SkinThickness = Entry(root,textvariable=xSkinThickness).pack()

Insulin = Entry(root,textvariable=xInsulin).pack()
BMI = Entry(root,textvariable=xBMI).pack()
DiabetesPedigreeFunction = Entry(root,textvariable=xDiabetesPedigreeFunction).pack()
Age = Entry(root,textvariable=xAge).pack()
#


#Bouton du formulaire
bouton1= Button(root,text="predire",command=predire).place(x='225',y='350')

