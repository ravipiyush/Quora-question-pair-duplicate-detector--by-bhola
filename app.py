import streamlit as st
import pickle
from preprocess import clean_text

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

st.title("🔍 Quora Question Pair Similarity")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check Similarity"):
    q1_clean = clean_text(q1)
    q2_clean = clean_text(q2)

    combined = q1_clean + " " + q2_clean
    vector = tfidf.transform([combined])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("✅ Questions are Similar (Duplicate)")
    else:
        st.error("❌ Questions are Not Similar")
