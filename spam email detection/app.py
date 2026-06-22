import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Email Detector")

email = st.text_area("Enter Email Content")

if st.button("Detect"):
    
    transformed = vectorizer.transform([email])

    prediction = model.predict(transformed)

    if prediction[0] == "spam":
        st.error("⚠ Spam Email Detected")
    else:
        st.success("✅ Genuine Email (Ham)")