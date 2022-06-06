import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)




st.title('Penguin Species prediction')

bill_length_mm=st.sidebar.slider('Bill_length_mm',int(df['bill_length_mm'].min()),int(df['bill_length_mm'].max()),step=1)
bill_depth_mm=st.sidebar.slider('Bill_depth_mm',int(df['bill_depth_mm'].min()),int(df['bill_depth_mm'].max()),step=1)
flipper_length_mm=st.sidebar.slider('flipper_length_mm',int(df['flipper_length_mm'].min()),int(df['flipper_length_mm'].max()),step=1)
body_mass_g=st.sidebar.slider('body_mass_g',int(df['body_mass_g'].min()),int(df['body_mass_g'].max()),step=10)

sex=st.sidebar.selectbox('Sex',(df['sex'].unique()))
island=st.sidebar.selectbox('island',(df['island'].unique()))

model=st.sidebar.selectbox('Select Model',('Random Forest Classifier','Support Vector machines','Logistic Regression'))


# prediction functions--->>>>
def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
    if model=='Random Forest Classifier':
        prediction=rf_clf.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
        if prediction==0:
            st.write("Speices predicted : Adelie")
        elif prediction==1:
            st.write("Speices predicted : Chinstrap")
        elif prediction==2:
            st.write("Speices predicted : Gentoo")

    elif model=='Support Vector machines':
        prediction=svc_model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
        if prediction==0:
            st.write("Speices predicted : Adelie")
        elif prediction==1:
            st.write("Speices predicted : Chinstrap")
        elif prediction==2:
            st.write("Speices predicted : Gentoo")

    elif model=='Logistic Regression':
        prediction=log_reg.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
        if prediction==0:
            st.write("Speices predicted : Adelie")
        elif prediction==1:
            st.write("Speices predicted : Chinstrap")
        elif prediction==2:
            st.write("Speices predicted : Gentoo")



st.sidebar.button('Predict',on_click=prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex))


