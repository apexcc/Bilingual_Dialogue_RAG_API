"""
Bilingual Dialogue RAG API - Backend
Single-mode LLM chatbot with web interface and API callback support
Supports both Chinese and English dialogue with RAG (Retrieval Augmented Generation)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ===================== Configuration =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "gpt2"  # Replace with your LLM model (e.g., Qwen, LLaMA, Mistral)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# ===================== LLM Model Initialization =====================
class BillingualLLMModel:
    """Local LLM wrapper for bilingual dialogue"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the language model"""
        self.model_name = model_name
        self.device = DEVICE
        
        logger.info(f"Loading model: {model_name} on device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.model.to(self.device)
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(k=5)  # Keep last 5 messages
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def create_prompt(self, user_input: str, language: str = "auto") -> str:
        """
        Create a structured prompt for bilingual dialogue
        
        Args:
            user_input: User's input message
            language: "auto", "zh", "en"
        
        Returns:
            Formatted prompt string
        """
        prompt_template = """<|system|>You are a helpful bilingual assistant that can respond in both Chinese and English.
Keep responses concise, clear, and contextual. Detect the language of user input and respond in the same language.
If user switches languages, follow their preference.

<|user|>{user_input}
<|assistant|>"""
        
        return prompt_template.format(user_input=user_input)
    
    def generate_response(self, user_input: str, max_new_tokens: int = 150) -> str:
        """
        Generate a response to user input
        
        Args:
            user_input: User's message
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated response text
        """
        try:
            prompt = self.create_prompt(user_input)
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=1
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response part
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error processing your request. Please try again."
    
    def get_memory(self) -> str:
        """Get conversation history from memory"""
        return self.memory.buffer


# ===================== Flask App Setup =====================
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize model (will be loaded once)
try:
    llm_model = BillingualLLMModel(MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    llm_model = None

# Store conversation history
conversation_history: List[Dict] = []


# ===================== API Endpoints =====================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": llm_model is not None,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for both web and C# client
    
    Expected JSON:
    {
        "message": "user's message",
        "language": "auto|zh|en",
        "include_history": true/false,
        "callback_url": "optional_callback_url"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400
        
        user_message = data.get('message', '').strip()
        language = data.get('language', 'auto')
        include_history = data.get('include_history', False)
        callback_url = data.get('callback_url', None)
        
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        if llm_model is None:
            return jsonify({"error": "Model not initialized"}), 500
        
        # Generate response
        response_text = llm_model.generate_response(user_message)
        
        # Store in conversation history
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": response_text,
            "language": language
        }
        conversation_history.append(turn)
        
        # Prepare response payload
        response_payload = {
            "status": "success",
            "response": response_text,
            "user_message": user_message,
            "timestamp": turn["timestamp"],
            "turn_id": len(conversation_history) - 1
        }
        
        if include_history:
            response_payload["conversation_history"] = conversation_history[-5:]  # Last 5 turns
        
        # If callback URL provided, send async notification
        if callback_url:
            # In production, use Celery or similar for async callback
            logger.info(f"Callback URL provided: {callback_url}")
            response_payload["callback_status"] = "scheduled"
        
        return jsonify(response_payload), 200
    
    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        return jsonify({
            "status": "success",
            "history": conversation_history[-limit:],
            "total_turns": len(conversation_history)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({
        "status": "success",
        "message": "Conversation history cleared"
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model and session status"""
    return jsonify({
        "status": "ok",
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "total_conversations": len(conversation_history),
        "memory_usage": f"{llm_model.get_memory()[:100]}..." if llm_model else "N/A"
    }), 200


# ===================== Web Interface Route =====================
@app.route('/', methods=['GET'])
def web_interface():
    """Serve the web chat interface"""
    return serve_web_chat()


def serve_web_chat():
    """Simple web chat interface HTML"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bilingual Dialogue Chat - Local LLM</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .chat-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 700px;
                height: 80vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .chat-header h1 { font-size: 24px; margin-bottom: 5px; }
            .chat-header p { font-size: 14px; opacity: 0.9; }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }
            .message {
                margin-bottom: 15px;
                animation: slideIn 0.3s ease-in-out;
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.user {
                text-align: right;
            }
            .message.user .bubble {
                background: #667eea;
                color: white;
                border-radius: 18px 18px 5px 18px;
            }
            .message.assistant .bubble {
                background: #e9ecef;
                color: #333;
                border-radius: 18px 18px 18px 5px;
            }
            .bubble {
                display: inline-block;
                max-width: 70%;
                padding: 12px 16px;
                word-wrap: break-word;
                font-size: 14px;
                line-height: 1.4;
            }
            .timestamp {
                font-size: 12px;
                color: #999;
                margin-top: 5px;
            }
            .chat-input-area {
                border-top: 1px solid #ddd;
                padding: 15px;
                background: white;
                display: flex;
                gap: 10px;
            }
            .input-group {
                flex: 1;
                display: flex;
                gap: 10px;
            }
            input[type="text"] {
                flex: 1;
                border: 1px solid #ddd;
                border-radius: 24px;
                padding: 10px 16px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus {
                border-color: #667eea;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                border-radius: 24px;
                padding: 10px 24px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s;
            }
            button:hover { background: #764ba2; transform: translateY(-2px); }
            button:active { transform: translateY(0); }
            .loading {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #667eea;
                border-radius: 50%;
                animation: pulse 1.5s infinite;
                margin: 0 2px;
            }
            @keyframes pulse {
                0%, 100% { opacity: 0.6; }
                50% { opacity: 1; }
            }
            .clear-btn {
                background: #dc3545;
                padding: 10px 16px;
                font-size: 12px;
            }
            .clear-btn:hover { background: #c82333; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 Bilingual Dialogue Chat</h1>
                <p>Powered by Local LLM | Chinese & English Support</p>
            </div>
            <div class="chat-messages" id="messages"></div>
            <div class="chat-input-area">
                <div class="input-group">
                    <input type="text" id="userInput" placeholder="Type your message here... (支持中英文)" />
                    <button onclick="sendMessage()">Send</button>
                </div>
                <button class="clear-btn" onclick="clearChat()">Clear</button>
            </div>
        </div>

        <script>
            const API_URL = '/api/chat';
            let messageCount = 0;

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user');
                input.value = '';
                
                // Show loading indicator
                const loadingId = 'loading-' + Date.now();
                const loadingDiv = document.createElement('div');
                loadingDiv.id = loadingId;
                loadingDiv.className = 'message assistant';
                loadingDiv.innerHTML = '<div class="bubble"><span class="loading"></span><span class="loading"></span><span class="loading"></span></div>';
                document.getElementById('messages').appendChild(loadingDiv);
                scrollToBottom();
                
                try {
                    const response = await fetch(API_URL, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            language: 'auto',
                            include_history: false
                        })
                    });
                    
                    const data = await response.json();
                    document.getElementById(loadingId).remove();
                    
                    if (response.ok) {
                        addMessage(data.response, 'assistant');
                    } else {
                        addMessage('Error: ' + (data.error || 'Unknown error'), 'assistant');
                    }
                } catch (error) {
                    document.getElementById(loadingId).remove();
                    addMessage('Connection error. Please try again.', 'assistant');
                    console.error(error);
                }
            }

            function addMessage(text, sender) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + sender;
                
                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                messageDiv.appendChild(bubble);
                
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.textContent = new Date().toLocaleTimeString();
                messageDiv.appendChild(timestamp);
                
                messagesDiv.appendChild(messageDiv);
                scrollToBottom();
            }

            function scrollToBottom() {
                const messagesDiv = document.getElementById('messages');
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function clearChat() {
                if (confirm('Clear all messages?')) {
                    document.getElementById('messages').innerHTML = '';
                    fetch('/api/chat/clear', { method: 'POST' });
                }
            }

            document.getElementById('userInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            // Welcome message
            window.addEventListener('load', () => {
                addMessage('👋 Hello! I\\'m your bilingual AI assistant. You can chat with me in English or Chinese. How can I help you today?', 'assistant');
            });
        </script>
    </body>
    </html>
    """
    return html


# ===================== Error Handlers =====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ===================== Main Entry =====================
if __name__ == '__main__':
    logger.info(f"Starting Bilingual Dialogue RAG API on device: {DEVICE}")
    logger.info(f"Model: {MODEL_NAME}")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )
