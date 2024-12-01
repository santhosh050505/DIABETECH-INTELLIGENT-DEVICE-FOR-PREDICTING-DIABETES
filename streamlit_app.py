# Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv(r'C:\Users\User\Downloads\hs\diabetes.csv')

# Preprocessing
df = df.drop('Pregnancies', axis=1)  # Drop 'Pregnancies' column

# Streamlit Headings
st.title('Diabetech - Monitoring Diabetes')
st.sidebar.header('Patient Data')
st.subheader('Training Data Summary')
st.write(df.describe())

# Split Dataset into Features and Target
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to Collect User Input
def user_report():
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    data = {
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

# User Input Data
user_data = user_report()
st.subheader('Patient Data Input')
st.write(user_data)

# Train Random Forest Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Predict the Outcome
user_result = rf.predict(user_data)

# Visualizations
st.title('Patient Report and Visualizations')

# Define Color Based on Prediction
color = 'blue' if user_result[0] == 0 else 'red'

# Visualize: Age vs Glucose
st.header('Glucose Levels (Others vs Yours)')
fig1 = plt.figure()
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
plt.scatter(user_data['Age'], user_data['Glucose'], color=color, s=150, label='You')
plt.title('Glucose Levels')
plt.legend()
st.pyplot(fig1)

# Visualize: Age vs Blood Pressure
st.header('Blood Pressure (Others vs Yours)')
fig2 = plt.figure()
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
plt.scatter(user_data['Age'], user_data['BloodPressure'], color=color, s=150, label='You')
plt.title('Blood Pressure Levels')
plt.legend()
st.pyplot(fig2)

# Visualize: Age vs Skin Thickness
st.header('Skin Thickness (Others vs Yours)')
fig3 = plt.figure()
sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
plt.scatter(user_data['Age'], user_data['SkinThickness'], color=color, s=150, label='You')
plt.title('Skin Thickness Levels')
plt.legend()
st.pyplot(fig3)

# Visualize: Age vs Insulin
st.header('Insulin Levels (Others vs Yours)')
fig4 = plt.figure()
sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
plt.scatter(user_data['Age'], user_data['Insulin'], color=color, s=150, label='You')
plt.title('Insulin Levels')
plt.legend()
st.pyplot(fig4)

# Visualize: Age vs BMI
st.header('BMI (Others vs Yours)')
fig5 = plt.figure()
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
plt.scatter(user_data['Age'], user_data['BMI'], color=color, s=150, label='You')
plt.title('BMI Levels')
plt.legend()
st.pyplot(fig5)

# Visualize: Age vs DPF
st.header('Diabetes Pedigree Function (Others vs Yours)')
fig6 = plt.figure()
sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
plt.scatter(user_data['Age'], user_data['DiabetesPedigreeFunction'], color=color, s=150, label='You')
plt.title('Diabetes Pedigree Function')
plt.legend()
st.pyplot(fig6)

# Display Prediction
st.subheader('Prediction Report')
st.write('Result: **You are Diabetic**' if user_result[0] == 1 else 'Result: **You are Not Diabetic**')

# Display Model Accuracy
st.subheader('Model Accuracy')
st.write(f'Accuracy: {accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%')
