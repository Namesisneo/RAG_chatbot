from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os
from main import MultiCollegeRAGChatbot  

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure PDF directory exists
pdf_dir = 'pdfs'
os.makedirs(pdf_dir, exist_ok=True)

# Initialize the chatbot (only once)
try:
    chatbot = MultiCollegeRAGChatbot(pdf_dir)
    
    # Prepare vector store during initialization
    chatbot.create_vector_store()
except Exception as e:
    logging.error(f"Error initializing chatbot: {e}")

# Route to serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")

        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        try:
            # Use the chatbot to generate a response
            bot_reply = chatbot.get_response(user_message)
            logging.debug(f"Generated bot reply: {bot_reply}")
            return jsonify({"reply": bot_reply})
        except Exception as e:
            logging.error(f"Chatbot Error: {e}")
            return jsonify({"error": "Failed to process your message. Try again later."}), 500
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/login')
def login():
    return render_template('loginpage.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)