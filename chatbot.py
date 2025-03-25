import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual API key if needed

# ✅ Define GEMINI_API_URL before using it
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

def chat_with_gemini(user_message):
    if not user_message:
        return "Message is required"
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": user_message}]}]}

    # ✅ Use the defined GEMINI_API_URL
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        response_data = response.json()
        try:
            reply = response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            reply = "Sorry, I couldn't process the response."
        return reply
    else:
        return f"Error: {response.json()}"

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chat_with_gemini(user_input)
        print(f"Bot: {response}")
