from flask import Flask, render_template, request, jsonify
from vector_store.manager import ChromaDBManager
from utils.file_uploader import FileUploader
from file_processing.file_processor import FileProcessor
from query_docs.retrieve_docs import ChromaDBRetriever
from query_docs.result_displayer import ChromaDBResultDisplayer
from chatbot.chatbot_ollama import ChatBotOllama
from chatbot.chatbot_gemini import ChatBotGemini
import os
import traceback
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB
chromaDB_path = "./ChromaDBData"
collection_name = "Papers"
model_name = "distiluse-base-multilingual-cased-v1"
ollama_model = "qwen3:0.6b"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize components
db_manager = ChromaDBManager(chromaDB_path)
chroma_client, chroma_collection = db_manager.create.create_client_and_collection(
    collection_name, model_name)
db_manager.initialize_add(chroma_client)

# Initialize chatbots
chatbot_ollama = ChatBotOllama(model_name=ollama_model)
chatbot_gemini = ChatBotGemini(api_key=GOOGLE_API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Process the file
            file_processor = FileProcessor(db_manager, model_name, chroma_collection)
            file_processor.process_files([filepath])
            
            return jsonify({'message': 'File processed successfully'})
            
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/query', methods=['POST', 'OPTIONS'])
def query():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query_text = data.get('query')
        if not query_text:
            return jsonify({'error': 'No query provided'}), 400
        
        # Retrieve documents
        query_executor = ChromaDBRetriever(chroma_collection)
        results = query_executor.retrieve_docs(query_text, n_results=5)
        
        # 1. Ollama'dan cevap al
        ollama_response = chatbot_ollama.generate_answer(prompt=query_text, context=results)
        
        # 2. Gemini'ye daha iyi yazması için gönder
        gemini_prompt = (
            "Aşağıda bir soru ve Ollama LLM tarafından verilen bir cevap var. "
            "Lütfen bu cevabı daha açıklayıcı, akıcı ve profesyonel bir şekilde tekrar yaz. "
            "Gerekiyorsa döküman bağlamını da kullanabilirsin.\n\n"
            f"Soru: {query_text}\n"
            f"Ollama'nın cevabı: {ollama_response}"
        )
        final_response = chatbot_gemini.generate_answer(prompt=gemini_prompt, context=results)
        
        # Sonucu dön
        result_displayer = ChromaDBResultDisplayer()
        formatted_results = result_displayer.show_results(results, return_only_docs=True)
        
        return jsonify({
            'response': final_response,
            'ollama_response': ollama_response,
            'documents': formatted_results
        })
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True) 