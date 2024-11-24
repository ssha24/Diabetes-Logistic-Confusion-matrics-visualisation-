import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
a=LogisticRegression()
import matplotlib.pyplot as plt

# spli for data test and train
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
df=pd.read_csv("diabetes.csv")
x=df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y=df["Outcome"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
a.fit(x_train,y_train)
y_pred=a.predict(x_test)
print("x_test",y_pred)
print("y_test",y_test)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cnf_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()
