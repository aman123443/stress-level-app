from flask import Flask, request, jsonify
import google.generativeai as genai
import time

# --- API KEY CONFIGURATION ---
API_KEY = "AIzaSyCp2_o7k4mredJFEg8DyzqBfKABXt9f-eU"  # Replace with your Gemini API key
genai.configure(api_key=API_KEY)

# --- SYSTEM INSTRUCTION ---
SYSTEM_INSTRUCTION = (
    "You are a friendly and knowledgeable AI mental health advisor focused ONLY on student stress and well-being. "
    "You must provide practical advice, tips, and suggestions related to common areas of student stress like "
    "anxiety, self-esteem, depression, sleep, academic performance, social pressure, and future concerns. "
    "Your job is to help students cope with these stress factors through relaxation techniques, study planning, "
    "mental health tips, motivation, and encouragement. "
    "Keep your responses concise, empathetic, and easy to understand. "
    "If the user asks something unrelated to student stress or mental well-being, kindly and firmly respond with: "
    "'I apologize, but my purpose is to assist with student stress and mental well-being. "
    "How can I help you with that today?'"
)

# --- MODEL INITIALIZATION ---
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

# --- FLASK APP ---
app = Flask(__name__)

def get_bot_response(user_prompt, chat_history=None, max_retries=3):
    if chat_history is None:
        chat_history = []

    retries = 0
    while retries < max_retries:
        try:
            chat_session = model.start_chat(history=chat_history)
            response = chat_session.send_message(user_prompt)
            return response.text

        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")

            if "429" in error_msg or "quota" in error_msg.lower():
                wait_time = 12
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
                continue

            if "API key not valid" in error_msg:
                return "There seems to be an issue with the API key. Please verify it is correct."

            return "I'm sorry, I encountered an issue while trying to respond."

    return "I'm sorry, I'm still hitting limits. Please try again later."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_prompt = data.get("message", "")
    chat_history = data.get("history", [])
    bot_reply = get_bot_response(user_prompt, chat_history)
    return jsonify({"reply": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
