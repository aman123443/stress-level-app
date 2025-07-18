import streamlit as st
import joblib
import numpy as np
# Inject custom CSS
with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Stress Level Predictor", layout="centered")
st.title("🧠 Student Stress Level Predictor")
st.markdown("""
<div style='background-color:#eaf4ff;padding:15px;border-radius:10px;margin-bottom:20px'>
    <h2 style='color:#004085;'>🎓 Welcome to the Stress Level Predictor</h2>
    <p>This app helps students understand their mental well-being using AI.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("Enter values for the following parameters (1–10):")

# List of input features
features = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying"
]

# Collect user input
inputs = [st.slider(f"{feature.replace('_', ' ').capitalize()}", 1, 10, 5) for feature in features]

# Predict and display result
if st.button("🔮 Predict"):
    prediction = model.predict([inputs])[0]
    probabilities = model.predict_proba([inputs])[0]
    stress_levels = ["Low", "Medium", "High"]

    st.markdown(f"### 🔮 Predicted Stress Level: **{stress_levels[prediction]}**")
    st.markdown("### 📊 Prediction Probabilities:")
    st.write({
        "Low": round(probabilities[0] * 100, 2),
        "Medium": round(probabilities[1] * 100, 2),
        "High": round(probabilities[2] * 100, 2),
    })

    # Show recommendation
    st.markdown("### 📝 Recommendation:")
    if prediction == 0:
        st.success("🟢 Keep up your healthy habits and balanced lifestyle.")
    elif prediction == 1:
        st.warning("🟠 Improve sleep, talk to peers, and practice mindfulness.")
    else:
        st.error("🔴 High stress detected. Seek professional support and consider reducing academic pressure.")
