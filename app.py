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

import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Prediction")
st.write(
    "Predict estimated employee salary using "
    "Machine Learning."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("salary_data.csv")

data = load_data()

# Features and target
features = [
    "Age",
    "EducationLevel",
    "JobTitle",
    "YearsExperience"
]

X = data[features]
y = data["Salary"]

# Convert categorical data into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "education",
            OneHotEncoder(handle_unknown="ignore"),
            ["EducationLevel"]
        ),
        (
            "job",
            OneHotEncoder(handle_unknown="ignore"),
            ["JobTitle"]
        )
    ],
    remainder="passthrough"
)

# Create machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X, y)

st.subheader("Enter Employee Details")

# User inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25,
    step=1
)

education = st.selectbox(
    "Education Level",
    sorted(data["EducationLevel"].unique())
)

job_title = st.selectbox(
    "Job Title",
    sorted(data["JobTitle"].unique())
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

# Prediction
if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "EducationLevel": education,
        "JobTitle": job_title,
        "YearsExperience": experience
    }])

    salary = model.predict(input_data)[0]

    st.success(
        f"Predicted Annual Salary: ₹{salary:,.0f}"
    )

    st.info(
        f"Estimated Monthly Salary: "
        f"₹{salary / 12:,.0f}"
    )

# Show dataset
with st.expander("View Salary Dataset"):
    st.dataframe(data, use_container_width=True)

st.caption(
    "Educational demo: predictions depend on "
    "the sample dataset and may not reflect "
    "actual salaries."
)
