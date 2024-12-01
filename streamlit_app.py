# Install required libraries
# pip install streamlit pandas scikit-learn matplotlib seaborn plotly

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# LOAD THE DATASET
# Ensure the `diabetes.csv` file is in the same directory as `streamlit_app.py`
df = pd.read_csv('diabetes.csv')

# REMOVE THE PREGNANCIES COLUMN FROM THE DATASET
df = df.drop('Pregnancies', axis=1)

# HEADINGS
st.title('Diabetech - Monitoring Diabetes')
st.sidebar.header('Patient Data')
st.subheader('Training Data Overview')
st.write(df.describe())

# SPLIT THE DATA INTO X AND Y
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION TO GET USER INPUT (WITHOUT PREGNANCIES)
def user_report():
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    # Create a DataFrame with user inputs
    user_report_data = {
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(user_report_data, index=[0])

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data Input')
st.write(user_data)

# TRAINING THE MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# PREDICTING THE RESULT
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualized Patient Report')

# Function to color based on prediction
color = 'blue' if user_result[0] == 0 else 'red'

# Visualization - Age vs Glucose
st.header('Glucose Value Comparison (Others vs Yours)')
fig1 = plt.figure()
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig1)

# Visualization - Age vs Blood Pressure
st.header('Blood Pressure Comparison (Others vs Yours)')
fig2 = plt.figure()
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='coolwarm')
sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig2)

# Visualization - Age vs BMI
st.header('BMI Comparison (Others vs Yours)')
fig3 = plt.figure()
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig3)

# OUTPUT
st.subheader('Prediction Result')
output = 'You are not diabetic.' if user_result[0] == 0 else 'You are diabetic.'
st.title(output)

# DISPLAY ACCURACY
st.subheader('Model Accuracy')
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.write(f"{accuracy:.2f}%")
