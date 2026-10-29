
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load CSV, handling common encodings
try:
    df = pd.read_csv("data.csv", encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv("data.csv", encoding="cp1252")
    except UnicodeDecodeError:
        df = pd.read_csv("data.csv", encoding="latin1")

# Prepare data
X = df[["HoursStudied"]]
y = df["ExamScore"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# App interface
st.title("Exam Score Predictor")
st.write("Enter hours studied to predict the exam score.")

hours = st.number_input(
    "Hours studied:",
    min_value=0.0,
    step=0.1
)

if st.button("Predict Score"):
    prediction_data = pd.DataFrame({"HoursStudied": [hours]})
    predicted_score = model.predict(prediction_data)[0]
    st.success(f"Predicted score: {predicted_score:.2f}")

st.write("### Sample Training Data")
st.dataframe(df)
