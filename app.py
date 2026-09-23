
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼"
)

# Title
st.title("Employee Salary Prediction")
st.write("Predict employee salary using Machine Learning")

# Load dataset
data = pd.read_csv("salary_data.csv")

# Input and output
X = data[["YearsExperience"]]
y = data["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Model accuracy
predictions = model.predict(X)
accuracy = r2_score(y, predictions)

# User input
st.subheader("Enter Employee Details")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=3.0,
    step=0.5
)

# Prediction button
if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "YearsExperience": [experience]
    })

    salary = model.predict(input_data)[0]

    st.success(
        f"Predicted Annual Salary: ₹{salary:,.2f}"
    )

    st.write(
        f"Monthly Salary: ₹{salary / 12:,.2f}"
    )

# Display dataset
st.subheader("Employee Salary Dataset")
st.dataframe(data)

# Display model information
st.subheader("Model Information")
st.write("Algorithm: Linear Regression")
st.write(f"R² Score: {accuracy:.2f}")
