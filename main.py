
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Training data
df = pd.DataFrame({
    "HoursStudied": [1.5, 3.2, 5, 7.1, 8.5, 2.5, 4.5, 5.5, 6.2, 3.5],
    "ExamScore": [30, 50, 75, 85, 95, 63, 78, 86, 80, 72]
})

# Prepare and train model
X = df[["HoursStudied"]]
y = df["ExamScore"]

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
    input_data = pd.DataFrame({"HoursStudied": [hours]})
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted score: {prediction:.2f}")

st.write("### Sample Training Data")
st.dataframe(df)
