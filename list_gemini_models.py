import os
from dotenv import load_dotenv
from chatbot.chatbot_gemini import ChatBotGemini

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(ChatBotGemini.list_available_models(api_key))