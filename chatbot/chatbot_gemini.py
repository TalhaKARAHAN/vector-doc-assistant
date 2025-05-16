import google.generativeai as genai

class ChatBotGemini:
    def __init__(self, api_key, system_instruction=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        self.system_instruction = system_instruction or "You are a helpful assistant."
        self.history = []

    def generate_answer(self, prompt, context):
        context_str = str(context)
        full_prompt = f"{self.system_instruction}\n\n{prompt}\n\nContext:\n{context_str}"
        response = self.model.generate_content(full_prompt)
        answer = response.text
        self.history.append({"prompt": prompt, "response": answer})
        return answer

    def history_chat(self):
        return self.history

