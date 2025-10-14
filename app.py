from flask import Flask, render_template, request, redirect, url_for, flash, session as flask_session
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import joblib
import numpy as np
import os
from chatbot_gemini import get_bot_response # Import your chatbot function

# --- APP & DATABASE CONFIGURATION ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- DATABASE SETUP (PostgreSQL) ---
# Replace with your actual PostgreSQL connection string
DATABASE_URL = "postgresql://postgres:8459@localhost:5432/stress_app_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    author = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- ML MODEL & FEATURE LISTS ---
try:
    model = joblib.load("stress_model.pkl")
except FileNotFoundError:
    model = None
    print("Error: stress_model.pkl not found. Prediction feature will be disabled.")

MODEL_FEATURE_ORDER = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying"
]

RECOMMENDATIONS_DB = {
    "anxiety_level": "Try practicing the 4-7-8 breathing technique. Also, consider mindfulness apps for guided meditation.",
    "self_esteem": "Practice positive self-talk. Each day, write down three things you did well, no matter how small.",
    "depression": "Engage in light physical activity, like a 15-minute walk outside. Sunlight and movement can have a significant positive impact on mood.",
    "sleep_quality": "Establish a consistent sleep schedule. Avoid screens for at least an hour before bed to improve your natural sleep cycle.",
    "academic_performance": "Break down large assignments into smaller, manageable tasks. Use a planner to schedule your work.",
    "social_support": "Schedule regular calls or meetups with friends and family. A strong social connection is a powerful buffer against stress.",
}


# --- ROUTING AND PAGE LOGIC ---

@app.route("/", methods=["GET", "POST"])
def login():
    if "username" in flask_session:
        return redirect(url_for('home'))
    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username")
        password = request.form.get("password")
        db = SessionLocal()
        if action == "signup":
            existing_user = db.query(User).filter(User.username == username).first()
            if existing_user:
                flash("Username already exists.", "error")
            else:
                new_user = User(username=username, password=password) # In a real app, hash the password!
                db.add(new_user)
                db.commit()
                flask_session["username"] = new_user.username
                flash(f"Welcome, {new_user.username}! Your account has been created.", "success")
                db.close()
                return redirect(url_for('home'))
        elif action == "login":
            user = db.query(User).filter(User.username == username).first()
            if user and user.password == password: # In a real app, verify hashed password
                flask_session["username"] = user.username
                db.close()
                return redirect(url_for('home'))
            else:
                flash("Invalid username or password.", "error")
        db.close()
    return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if "username" not in flask_session:
        return redirect(url_for('login'))
    db = SessionLocal()
    if request.method == "POST":
        review_content = request.form.get("review_content")
        if review_content:
            new_review = Review(author=flask_session["username"], content=review_content)
            db.add(new_review)
            db.commit()
            flash("Thank you for your review!", "success")
            db.close()
            return redirect(url_for('home'))
    reviews = db.query(Review).order_by(Review.created_at.desc()).limit(6).all()
    db.close()
    return render_template("home.html", username=flask_session.get("username"), reviews=reviews)

@app.route("/logout")
def logout():
    flask_session.pop("username", None)
    flask_session.pop("chat_history", None) # Clear chat history on logout
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    if "username" not in flask_session:
        return redirect(url_for('login'))
    
    prediction_result = None
    if request.method == "POST":
        if not model:
            flash("The prediction model is not loaded. Please contact the administrator.", "error")
        else:
            try:
                # Collect data from the form
                input_data = [int(request.form.get(feature, 5)) for feature in MODEL_FEATURE_ORDER]
                
                # Make prediction
                probabilities = model.predict_proba([input_data])[0]
                stress_labels = ["Low", "Medium", "High"]
                predicted_index = probabilities.argmax()
                stress_level = stress_labels[predicted_index]

                # Get recommendations for high-rated factors
                recommendations = []
                threshold = 7 # We'll recommend for factors rated 7 or higher
                for feature, value in zip(MODEL_FEATURE_ORDER, input_data):
                    if value >= threshold and feature in RECOMMENDATIONS_DB:
                        recommendations.append({"factor": feature, "tip": RECOMMENDATIONS_DB[feature]})

                prediction_result = {
                    "level": stress_level,
                    "probabilities": {
                        "Low": probabilities[0] * 100,
                        "Medium": probabilities[1] * 100,
                        "High": probabilities[2] * 100,
                    },
                    "recommendations": recommendations
                }
            except Exception as e:
                print(f"Prediction Error: {e}")
                flash("An error occurred during prediction. Please try again.", "error")

    return render_template("predictor.html", username=flask_session.get("username"), features=MODEL_FEATURE_ORDER, prediction=prediction_result)

@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    if "username" not in flask_session:
        return redirect(url_for('login'))

    # Initialize chat history in session if it doesn't exist
    if "chat_history" not in flask_session:
        flask_session["chat_history"] = []

    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            # Add user message to history
            flask_session["chat_history"].append({'role': 'user', 'parts': [user_input]})
            
            # Get bot response
            # Note: The format for history for the Gemini API is slightly different
            api_history = flask_session["chat_history"]
            bot_response_stream = get_bot_response(user_input, api_history)
            
            # Combine the streamed chunks into a single response
            full_bot_response = "".join([chunk for chunk in bot_response_stream])

            # Add bot response to history
            flask_session["chat_history"].append({'role': 'model', 'parts': [full_bot_response]})
            flask_session.modified = True # Ensure the session is saved

            return redirect(url_for('advisor'))

    return render_template("advisor.html", username=flask_session.get("username"), chat_history=flask_session["chat_history"])

# --- RUN THE APP ---
if __name__ == "__main__":
    app.run(debug=True)