# -*- coding: utf-8 -*-
"""
Created on Sat May  1 10:08:01 2021

@author: Fardin Ibrahimi
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg
import pickle

image1 = mpimg.imread('h1.png')     
image2 = mpimg.imread('h2.png')     
image3 = mpimg.imread('h3.png')     
image4 = mpimg.imread('h4.png')     
image5 = mpimg.imread('h5.png')     
image6 = mpimg.imread('h6.png')     
image7 = mpimg.imread('h7.png')     
image8 = mpimg.imread('h8.png')     
image9 = mpimg.imread('h9.png')     
image10 = mpimg.imread('h10.png')     
image11 = mpimg.imread('h11.png')     
image12 = mpimg.imread('h12.png')     
image13 = mpimg.imread('h13.png')     
image14 = mpimg.imread('h14.png')     
image15 = mpimg.imread('h15.png')     
image16 = mpimg.imread('h16.png')     
image17 = mpimg.imread('h17.png')     
image18 = mpimg.imread('h18.png')     
image19 = mpimg.imread('h19.png')     


st.set_page_config(page_title='Heart Disease Detection Machine Learning App',layout='wide')
imageha = mpimg.imread('ha.jpg') 
st.write("""
# Heart Disease Detection Application""")    
st.image(imageha)
st.write("""
In this implementation, various **Logistic Regression** algorithm is used in this app for building a **Classification Model** to **Detect Heart Disease**.
""")

st.write("'Age and Target' Countplot")
st.image(image19)
    
st.write("'Chest Pain' Countplot")
st.image(image3)

st.write("'Gender' Countplot")
st.image(image2)
    
st.write("'trestbps' Distplot")
st.image(image4)
    
st.write("'trestbps' Histogram")
st.image(image5)
    
st.write("'chol' Distplot")
st.image(image6)
    
st.write("'chol' Histogram")
st.image(image7)
    
st.write("'fbs' Countplot")
st.image(image8)
    
st.write("'restecg' Countplot")
st.image(image9)
    
st.write("'thalach' Distplot")
st.image(image10)
    
st.write("'thalach' Histogram")
st.image(image11)
    
st.write("'exang' Countplot")
st.image(image12)
    
st.write("'oldpeak' Distplot")
st.image(image13)
    
st.write("'oldpeak' Histogram")
st.image(image14)
    
st.write("'oldpeak' Histogram")
st.image(image15)
    
st.write("'slope' Countplot")
st.image(image16)
    
st.write("'ca' Countplot")
st.image(image17)
    
st.write("'thal' Countplot")
st.image(image18)
    

data = pd.read_csv('heart.csv')
X = data[['age', 'sex', 'cp', 'trestbps', 'chol',
       'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']]
y = data['target']


st.sidebar.write("Please Input the Information accurately!")

    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
st.markdown('**1.1. Glimpse of dataset**')
st.write(data.head(50))
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
     #st.sidebar.header('2. Set Parameters'):
age = st.sidebar.slider('AGE',1,150,1,1)
cp = st.sidebar.slider('CP : Chest Pain', 0, 3, 1, 1)
sex = st.sidebar.slider ("Gender", 0,1,0,1)
trestbps = st.sidebar.slider('TESTBPS : Resting blood pressure (in mm Hg on admission in hospital)', 94, 200, 80, 1)
chol = st.sidebar.slider('CHOL : Serum cholestrol in mg/dl', 126, 564, 246, 2)
fbs = st.sidebar.slider('FBS : (Fasting blood sugar > 120mg/dl) 1 = True, 0 = False', 0, 1, 0, 1)
restecg = st.sidebar.slider('RESTECG : Rest ECG', 0, 2, 1, 1)
exang = st.sidebar.slider('EXANG : Exercise induced angina', 0, 1, 0, 1)
oldpeak = st.sidebar.slider('OLDPEAK : ST depression induced by exercise related to rest', 0.0, 6.2, 3.2, 0.2)
slope= st.sidebar.slider('SLOPE : The slope of the peak exercise ST segment', 0, 2, 1, 1)
ca= st.sidebar.slider('CA : Number of major vessels (0-3) colored by flourosopy', 0, 4, 2, 1)
thal= st.sidebar.slider('THAL : (0-3) 3 = Normal; 6 = Fixed defect; 7 = Reversable defect', 0, 3, 1, 1)
thalach = st.sidebar.slider('THALAK',71,202,150,1)
    
X_test_sc = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    #logregs = LogisticRegression()
    #logregs.fit(X_train, y_train)
    #y_pred_st = logregs.predict(X_test_sc)
    
load_clf = pickle.load(open('heart_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(X_test_sc)
prediction_proba = load_clf.predict_proba(X_test_sc)
    
answer = prediction[0]
        
if answer == 0:
    st.warning("**The prediction is, that the Heart Disease was not Detected**")
   
else:
    st.warning("**The prediction is, that the Heart Disease was Detected**")
        
st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')

st.sidebar.title("Created By:")
st.sidebar.subheader("Fardin Ibrahimi")
