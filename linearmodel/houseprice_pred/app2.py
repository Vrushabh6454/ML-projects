import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("House Price Prediction App")

# Load dataset from file
try:
    df = pd.read_csv(r'data.csv')
    # Select important columns and rename them
    df = df[['RM', 'LSTAT', 'PTRATIO', 'AGE','TAX','MEDV']]
    df.columns = ['Rooms', 'Low_Status_Pop', 'Pupil_Teacher_Ratio', 'age','Tax','Price']
    st.write("Dataset Preview:")
    st.dataframe(df.head(5))
except FileNotFoundError:
    st.error("Dataset file 'C:\\ML\\orinson\\task 3\\data.csv' not found. Please ensure it exists in the specified path.")

# Define features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar: Model Parameters
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth of Decision Tree", min_value=1, max_value=20, value=5, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)

# Train model
model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-Squared (R2): {r2:.2f}")

# User Input for Prediction
st.write("### Make a Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

user_df = pd.DataFrame([user_input])
prediction = model.predict(user_df)[0]

st.write(f"### Predicted Price: ${prediction:.2f}")
