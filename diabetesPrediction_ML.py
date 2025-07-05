import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
import pickle
import streamlit as st

#Data collection and analysis : 

diabetes_dataset=pd.read_csv('diabetes.csv')
#print(diabetes_dataset.head())
#print(diabetes_dataset.shape)
#print(diabetes_dataset.describe())
#print(diabetes_dataset['Outcome'].value_counts())
# 0 => non-diabetic    1 => diabetic
#print(diabetes_dataset.groupby('Outcome').mean())

# separating data and labels :

x=diabetes_dataset.drop(columns='Outcome',axis=1)
y=diabetes_dataset['Outcome']
#print(y)

# Data Standardization :

scaler=StandardScaler()
standardized_data = scaler.fit_transform(x)
#print(standardized_data)
x=standardized_data
y=diabetes_dataset['Outcome']

# split :

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 ,stratify=y ,random_state=2)
#print(x.shape, x_train.shape, x_test.shape)

# Training the Model  :

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

# Model Evaluation :

x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
#print('Accuracy score of training data :',training_data_accuracy)  # 78.66%

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
#print('Accuracy score of test data :',test_data_accuracy)     #77.27%

# Making Prediction :
   
# Saving the trained model :

import pickle    
filename = 'trained_model'
pickle.dump(classifier,open(filename,'wb'))

# Loading the saved model :

loaded_model =pickle.load(open('trained_model','rb'))

# testing: 
input_data=(3,125,58,0,0,31.6,0.151,24)  # readings of a non diabetic patient => 0
input_array = np.array(input_data)
input_array=input_array.reshape(1,8)

std_data=scaler.transform(input_array) # standardizing input data
print("\n\n",std_data)

prediction = loaded_model.predict(std_data)

if(prediction==0):
    print("\n\nPerson is non-diabetic")
elif(prediction==1):
    print("\n\nPerson is diabetic")

# WebApp :

def diabetes_prediction(input_data):
    input_array = np.array(input_data)
    input_array=input_array.reshape(1,8)

    prediction = loaded_model.predict(std_data)

    if(prediction==0):
     return 'Person is non-diabetic'
    elif(prediction==1):
     return "Person is diabetic"
    
def main():
   st.title(' Diabetes Prediction WebApp 🩺 ')
   Pregnancies = st.text_input('Number of Pregnancies :')
   Glucose = st.text_input('Glucose level :')
   BloodPressure = st.text_input('Blood Pressure value :')
   SkinThickness = st.text_input('Skin thickness :')
   Insulin =  st.text_input('Insulin level :')
   BMI = st.text_input('BMI index :')
   DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value :')
   Age = st.text_input('Age of the person :')
   
   #code for prediction :
   diagnosis= ''
   if st.button('Diabetes test result ') :
      diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
      st.success(diagnosis)
      

if __name__== '__main__' :
   main()