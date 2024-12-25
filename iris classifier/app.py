import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Title and description
st.title("Iris Classification App")
st.write("This app performs classification on the Iris dataset using Logistic Regression or Decision Tree.")

# Load the Iris dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['species'])

# Preprocessing: Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

st.write("### Iris Dataset Preview:")
st.dataframe(pd.concat([x, y], axis=1).head())

# Model selection
model_option = st.selectbox("Select a Model:", ["Logistic Regression", "Decision Tree"])

# User input for model prediction
st.write("### Input Features:")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

# Model training and selection
if model_option == "Logistic Regression":
    model = LogisticRegression()
else:
    model = DecisionTreeClassifier()

model.fit(x_train, y_train.values.ravel())

# Predictions for user input
input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
predicted_class = data.target_names[prediction[0]]

# Display the prediction
st.write(f"### Predicted Class: {predicted_class}")

# F1 Score for the model
y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='micro')
st.write(f"### F1 Score of the Model: {f1:.2f}")

# Confusion Matrix
st.write("### Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=data.target_names)
display.plot(ax=ax, cmap='Blues')
st.pyplot(fig)
