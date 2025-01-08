import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Test_Score': [35, 50, 55, 70, 65, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(data)


st.title("Linear Regression App")
st.write("This app demonstrates a simple Linear Regression model.")

st.subheader("Dataset")
st.write(df)


X = df[['Hours_Studied']]
y = df['Test_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
st.subheader("Model Evaluation")
st.write(f"Mean Squared Error: {mse:.2f}")


st.subheader("Scatter Plot with Regression Line")


fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data Points')
line = model.coef_ * X + model.intercept_
ax.plot(X, line, color='red', label='Regression Line')
ax.set_title("Linear Regression: Hours vs Test Score")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Test Score")
ax.legend()


st.pyplot(fig)
