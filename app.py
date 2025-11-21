# ============================
# Flask & System Imports
# ============================
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session as flask_session, send_file
)
from datetime import datetime
import os
import io
import joblib
import numpy as np

# ============================
# SQLAlchemy (Database)
# ============================
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# ============================
# ReportLab (PDF Generation)
# ============================
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, Frame, PageTemplate, KeepInFrame
)

from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# for matplotlib headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# If you have a chatbot import; keep or comment out if not present
try:
    from chatbot_gemini import get_bot_response
except Exception:
    def get_bot_response(user_input, history):
        return ["(chatbot not configured)"]

# --- APP & DATABASE CONFIGURATION ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(24))

# --- DATABASE SETUP (PostgreSQL) ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///stress_app.db"
)

engine = create_engine(DATABASE_URL, future=True)
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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stress_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None
    print("Warning: stress_model.pkl not found or failed to load. Prediction disabled.")

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
    # Add more feature-specific recommendations as needed
}

# Ensure static directory exists for chart
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)

# --- Cache control for development (prevents browser from loading stale CSS/JS) ---
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# --- ROUTES ---
@app.route("/", methods=["GET", "POST"])
def login():
    if "username" in flask_session:
        return redirect(url_for('home'))

    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please provide username and password.", "error")
            return render_template("login.html")

        db = SessionLocal()
        try:
            if action == "signup":
                existing_user = db.query(User).filter(User.username == username).first()
                if existing_user:
                    flash("Username already exists.", "error")
                else:
                    new_user = User(username=username, password=password)
                    db.add(new_user)
                    db.commit()
                    flask_session["username"] = new_user.username
                    flash(f"Welcome, {new_user.username}! Your account has been created.", "success")
                    return redirect(url_for('home'))

            elif action == "login":
                user = db.query(User).filter(User.username == username).first()
                if user and user.password == password:
                    flask_session["username"] = user.username
                    return redirect(url_for('home'))
                else:
                    flash("Invalid username or password.", "error")
        finally:
            db.close()

    return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if "username" not in flask_session:
        return redirect(url_for('login'))

    db = SessionLocal()
    try:
        if request.method == "POST":
            review_content = request.form.get("review_content", "").strip()
            if review_content:
                new_review = Review(author=flask_session["username"], content=review_content)
                db.add(new_review)
                db.commit()
                flash("Thank you for your review!", "success")
                return redirect(url_for('home'))

        reviews = db.query(Review).order_by(Review.created_at.desc()).limit(6).all()
    finally:
        db.close()

    return render_template("home.html", username=flask_session.get("username"), reviews=reviews)

@app.route("/logout")
def logout():
    flask_session.pop("username", None)
    flask_session.pop("chat_history", None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    if "username" not in flask_session:
        return redirect(url_for('login'))

    prediction_result = None
    symptoms_short = request.form.get('symptoms', '')
    symptoms_long = request.form.get('symptoms_long', '')

    if request.method == "POST":
        if not model:
            flash("The prediction model is not loaded. Please contact the administrator.", "error")
        else:
            try:
                # Collect integer inputs; fallback to 5 if not provided or invalid
                input_data = []
                feature_values = {}
                for feature in MODEL_FEATURE_ORDER:
                    try:
                        val = int(request.form.get(feature, 5))
                    except Exception:
                        val = 5
                    input_data.append(val)
                    feature_values[feature] = val

                probabilities = model.predict_proba([input_data])[0]
                stress_labels = ["Low", "Medium", "High"]
                predicted_index = int(np.argmax(probabilities))
                stress_level = stress_labels[predicted_index]

                # Determine affected (high) and maintain (low-risk) parameters
                high_threshold = 7
                low_threshold = 3

                affected = []
                maintain = []

                for feature, value in feature_values.items():
                    if value >= high_threshold:
                        tip = RECOMMENDATIONS_DB.get(feature,
                                                   "Consider consulting a counselor or adopting stress-reduction strategies relevant to this area.")
                        affected.append({"factor": feature, "value": value, "tip": tip})
                    elif value <= low_threshold:
                        if feature in RECOMMENDATIONS_DB:
                            maintain_tip = "Good — keep doing this. " + RECOMMENDATIONS_DB[feature]
                        else:
                            maintain_tip = "This parameter is in a low-risk range. Maintain your current habits that support this."
                        maintain.append({"factor": feature, "value": value, "tip": maintain_tip})

                # Build readable probabilities and pack recs for PDF
                prediction_result = {
                    "level": stress_level,
                    "probabilities": {
                        "Low": probabilities[0] * 100,
                        "Medium": probabilities[1] * 100,
                        "High": probabilities[2] * 100,
                    },
                    "recommendations": affected,  # items needing attention
                    "maintain": maintain,         # items to maintain / low-risk
                    "feature_values": feature_values
                }

                # Create a packed recommendations string (multi-line) used for the PDF hidden field
                rec_lines = []
                if affected:
                    rec_lines.append("Affected Parameters:")
                    for a in affected:
                        human = a["factor"].replace("_", " ").title()
                        rec_lines.append(f"{human} ({a['value']}): {a['tip']}")

                if maintain:
                    if rec_lines:
                        rec_lines.append("")  # blank line separator
                    rec_lines.append("Parameters to Maintain:")
                    for m in maintain:
                        human = m["factor"].replace("_", " ").title()
                        rec_lines.append(f"{human} ({m['value']}): {m['tip']}")

                if not rec_lines:
                    rec_lines.append("No specific high-risk parameters detected. Inputs look balanced.")

                recommendations_packed = "\n".join(rec_lines)
                prediction_result["packed_recommendations"] = recommendations_packed

                # generate small chart image (headless-safe)
                try:
                    labels = ["Low", "Medium", "High"]
                    values = [probabilities[0]*100, probabilities[1]*100, probabilities[2]*100]
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(5,2.5))
                    bars = ax.bar(labels, values)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("Probability (%)")
                    ax.set_title("Stress Level Probabilities")
                    for bar in bars:
                        h = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontsize=9)
                    chart_path = os.path.join(os.path.dirname(__file__), "static", "chart.png")
                    plt.tight_layout()
                    fig.savefig(chart_path, dpi=100)
                    plt.close(fig)
                except Exception as e:
                    print("Chart generation failed:", e)

            except Exception as e:
                print(f"Prediction Error: {e}")
                flash("An error occurred during prediction. Please try again.", "error")

    # When rendering template, pass recommendations_packed if available (else empty string)
    recommendations_packed_to_template = ""
    if prediction_result and "packed_recommendations" in prediction_result:
        recommendations_packed_to_template = prediction_result["packed_recommendations"]

    return render_template(
        "predictor.html",
        username=flask_session.get("username"),
        features=MODEL_FEATURE_ORDER,
        prediction=prediction_result,
        symptoms_short=symptoms_short,
        symptoms_long=symptoms_long,
        recommendations_packed=recommendations_packed_to_template
    )

@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    if "username" not in flask_session:
        return redirect(url_for('login'))

    if "chat_history" not in flask_session:
        flask_session["chat_history"] = []

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            chat_history = flask_session["chat_history"]
            chat_history.append({'role': 'user', 'parts': [user_input]})
            bot_response_stream = get_bot_response(user_input, chat_history)
            full_bot_response = "".join([chunk for chunk in bot_response_stream])
            chat_history.append({'role': 'model', 'parts': [full_bot_response]})
            flask_session["chat_history"] = chat_history
            flask_session.modified = True
            return redirect(url_for('advisor'))

    return render_template("advisor.html", username=flask_session.get("username"), chat_history=flask_session["chat_history"])

# =============================
# PDF DOWNLOAD ROUTE
# =============================
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    """
    Clean, simple, professional PDF report (no charts).
    Nice spacing, centered header, left-aligned details.
    """

    patient_name = flask_session.get("username", "Unknown")
    symptoms = request.form.get('symptoms_long') or request.form.get('symptoms') or "Not provided"
    prediction = request.form.get('prediction', 'N/A')
    recommendations = request.form.get('recommendations', '').strip()

    buffer = io.BytesIO()

    # Create doc
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Clean professional styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    small_center = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#0b486b"),
        spaceBefore=12,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=11,
        leading=15
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["BodyText"],
        fontSize=11,
        leftIndent=14,
        bulletIndent=8,
        leading=15
    )

    elements = []

    # ======================================================
    # HEADER
    # ======================================================
    elements.append(Paragraph("MindWell — Stress Analysis Report", title_style))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        small_center
    ))

    # ======================================================
    # PATIENT DETAILS
    # ======================================================
    elements.append(Paragraph("Patient Details", heading_style))
    elements.append(Paragraph(f"<b>Name:</b> {patient_name}", normal_style))
    elements.append(Paragraph(f"<b>Predicted Stress Level:</b> {prediction}", normal_style))
    elements.append(Spacer(1, 10))

    # ======================================================
    # SYMPTOMS
    # ======================================================
    elements.append(Paragraph("Reported Symptoms", heading_style))
    elements.append(Paragraph(symptoms.replace("\n", "<br/>"), normal_style))

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================
    elements.append(Paragraph("Recommendations", heading_style))

    if recommendations:
        for line in recommendations.splitlines():
            ln = line.strip()
            if not ln:
                continue
            if ln.startswith("- "):
                elements.append(Paragraph(ln[2:], bullet_style, bulletText="•"))
            else:
                elements.append(Paragraph(ln, normal_style))
    else:
        elements.append(Paragraph(
            "No specific recommendations were provided. Stress levels appear balanced.",
            normal_style
        ))

    # ======================================================
    # ACTION CHECKLIST
    # ======================================================
    elements.append(Paragraph("Action Checklist", heading_style))

    checklist = [
        "Practice 10 minutes of slow breathing / mindfulness",
        "Do 15–30 minutes of light physical activity",
        "Use 25-minute focused work blocks (Pomodoro)",
        "Talk to a friend / family member once a week"
    ]

    for item in checklist:
        elements.append(Paragraph(item, bullet_style, bulletText="◻"))

    elements.append(Spacer(1, 20))

    # ======================================================
    # FOOTER
    # ======================================================
    footer = Paragraph(
        "This report is for informational purposes only — not a medical diagnosis.",
        ParagraphStyle("Footer", fontSize=8, alignment=TA_CENTER, textColor="#666666")
    )
    elements.append(footer)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="mindwell_report_simple.pdf",
        mimetype="application/pdf"
    )
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)

