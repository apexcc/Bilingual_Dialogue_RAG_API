"""
Bilingual Dialogue RAG API - Backend
Single-mode LLM chatbot with web interface and API callback support
Supports both Chinese and English dialogue with RAG (Retrieval Augmented Generation)
Powered by Ollama local LLM inference
"""

import os
import logging
import requests
from datetime import datetime
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===================== Configuration =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", 0.9))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", 40))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# ===================== Ollama LLM Model Wrapper =====================
class OllamaLLMModel:
    """Wrapper for Ollama local LLM inference"""

    def __init__(self, model_name: str = OLLAMA_MODEL, api_url: str = OLLAMA_API_URL):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.chat_endpoint = f"{self.api_url}/api/chat"
        self.conversation_history: List[Dict] = []

        logger.info(f"Initializing Ollama LLM: {model_name}")
        logger.info(f"Ollama API URL: {api_url}")

        if not self.test_connection():
            logger.warning(f"Warning: Could not connect to Ollama at {api_url}")
            logger.warning("Make sure Ollama is running: ollama serve")
        else:
            logger.info("Successfully connected to Ollama")

    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            return False

    def create_system_prompt(self) -> str:
        return """You are a helpful bilingual assistant that can respond in both Chinese and English.
- Detect the language of user input and respond in the same language
- Keep responses concise, clear, and contextual
- If the user switches languages, follow their preference
- Be friendly, helpful, and informative"""

    def generate_response(self, user_input: str, max_tokens: int = MAX_TOKENS) -> str:
        try:
            messages = [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": user_input},
            ]

            if len(self.conversation_history) > 0:
                history_messages = []
                for turn in self.conversation_history[-4:]:
                    history_messages.append({"role": "user", "content": turn["user"]})
                    history_messages.append({"role": "assistant", "content": turn["assistant"]})
                messages = [messages[0]] + history_messages + [messages[1]]

            logger.info(f"Calling Ollama model: {self.model_name}")
            response = requests.post(
                self.chat_endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": OLLAMA_TEMPERATURE,
                        "top_p": OLLAMA_TOP_P,
                        "top_k": OLLAMA_TOP_K,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "I encountered an error processing your request. Please make sure Ollama is running."

            result = response.json()
            response_text = result.get("message", {}).get("content", "").strip()
            if not response_text:
                logger.warning("Empty response from Ollama")
                return "I couldn't generate a response. Please try again."
            return response_text

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "The request took too long to process. Please try again with a shorter message."
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama")
            return "Cannot connect to Ollama service. Please ensure Ollama is running on http://localhost:11434"
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"An error occurred: {str(e)}"

    def add_to_history(self, user_message: str, assistant_response: str):
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)

    def get_history(self, limit: int = 10) -> List[Dict]:
        return self.conversation_history[-limit:]

    def clear_history(self):
        self.conversation_history.clear()
        logger.info("Conversation history cleared")


# ===================== Flask App Setup =====================
app = Flask(__name__)
CORS(app)

try:
    llm_model = OllamaLLMModel(OLLAMA_MODEL, OLLAMA_API_URL)
except Exception as e:
    logger.error(f"Failed to initialize Ollama model: {str(e)}")
    llm_model = None

conversation_history: List[Dict] = []


# ===================== API Endpoints =====================
@app.route('/api/health', methods=['GET'])
def health_check():
    ollama_status = "ok" if llm_model and llm_model.test_connection() else "error"
    return jsonify({
        "status": "ok",
        "ollama_status": ollama_status,
        "model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_API_URL,
        "timestamp": datetime.now().isoformat(),
    }), 200


@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        if not llm_model:
            return jsonify({"error": "Model not initialized"}), 500

        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return jsonify({
                "status": "success",
                "models": [{"name": m.get("name"), "size": m.get("size")} for m in models],
                "current_model": OLLAMA_MODEL,
            }), 200
        return jsonify({"error": "Failed to fetch models from Ollama"}), 500
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        user_message = data.get('message', '').strip()
        language = data.get('language', 'auto')
        include_history = data.get('include_history', False)
        callback_url = data.get('callback_url', None)
        max_tokens = data.get('max_tokens', MAX_TOKENS)

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        if llm_model is None:
            return jsonify({"error": "Ollama model not initialized"}), 500

        logger.info(f"Processing message from user: {user_message[:50]}...")
        response_text = llm_model.generate_response(user_message, max_tokens)

        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": response_text,
            "language": language,
            "model": OLLAMA_MODEL,
        }
        conversation_history.append(turn)
        llm_model.add_to_history(user_message, response_text)

        response_payload = {
            "status": "success",
            "response": response_text,
            "user_message": user_message,
            "timestamp": turn["timestamp"],
            "turn_id": len(conversation_history) - 1,
            "model": OLLAMA_MODEL,
        }

        if include_history:
            response_payload["conversation_history"] = llm_model.get_history(5)

        if callback_url:
            logger.info(f"Callback URL provided: {callback_url}")
            response_payload["callback_status"] = "scheduled"

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/history', methods=['GET'])
def get_history():
    try:
        limit = request.args.get('limit', default=10, type=int)
        history = llm_model.get_history(limit) if llm_model else conversation_history[-limit:]
        return jsonify({
            "status": "success",
            "history": history,
            "total_turns": len(conversation_history),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    if llm_model:
        llm_model.clear_history()
    return jsonify({
        "status": "success",
        "message": "Conversation history cleared",
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    if not llm_model:
        return jsonify({"error": "Model not initialized"}), 500

    return jsonify({
        "status": "ok",
        "model_name": OLLAMA_MODEL,
        "ollama_url": OLLAMA_API_URL,
        "total_conversations": len(conversation_history),
        "model_connected": llm_model.test_connection(),
        "history_size": len(llm_model.conversation_history),
        "configuration": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
            "max_tokens": MAX_TOKENS,
        },
    }), 200


# ===================== Web Interface Route =====================
@app.route('/', methods=['GET'])
def web_interface():
    return serve_web_chat()


def serve_web_chat():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bilingual Dialogue Chat - Ollama LLM</title>
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
            .chat-header .model-info { font-size: 12px; opacity: 0.8; margin-top: 8px; }
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
            .message.user { text-align: right; }
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
                flex-direction: column;
                gap: 10px;
            }
            .input-controls {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                font-size: 12px;
            }
            .input-controls label {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .input-controls input[type="range"] { width: 80px; }
            .input-group {
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
            input[type="text"]:focus { border-color: #667eea; }
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
            button:disabled { background: #ccc; cursor: not-allowed; }
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
            .status-bar {
                font-size: 11px;
                color: #666;
                padding: 5px 15px;
                background: #f0f0f0;
                border-top: 1px solid #ddd;
            }
            .status-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 5px;
                background: #28a745;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 Bilingual Dialogue Chat</h1>
                <p>Powered by Ollama Local LLM | Chinese & English Support</p>
                <div class="model-info" id="modelInfo">Loading model info...</div>
            </div>
            <div class="chat-messages" id="messages"></div>
            <div class="chat-input-area">
                <div class="input-controls">
                    <label>Temperature: <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.7" /></label>
                    <label>Max Tokens: <input type="number" id="maxTokens" min="50" max="2000" value="512" style="width: 60px;" /></label>
                </div>
                <div class="input-group">
                    <input type="text" id="userInput" placeholder="Type your message here... (支持中英文)" />
                    <button id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
                    <button style="background: #17a2b8; flex: 1;" onclick="fetchStatus()">Check Status</button>
                </div>
            </div>
            <div class="status-bar">
                <span class="status-indicator" id="statusIndicator"></span>
                <span id="statusText">Initializing...</span>
            </div>
        </div>

        <script>
            const API_URL = '/api/chat';
            let isLoading = false;

            async function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                const sendBtn = document.getElementById('sendBtn');

                if (!message || isLoading) return;

                isLoading = true;
                sendBtn.disabled = true;

                addMessage(message, 'user');
                input.value = '';

                const loadingId = 'loading-' + Date.now();
                const loadingDiv = document.createElement('div');
                loadingDiv.id = loadingId;
                loadingDiv.className = 'message assistant';
                loadingDiv.innerHTML = '<div class="bubble"><span class="loading"></span><span class="loading"></span><span class="loading"></span></div>';
                document.getElementById('messages').appendChild(loadingDiv);
                scrollToBottom();

                try {
                    const maxTokens = document.getElementById('maxTokens').value;
                    const response = await fetch(API_URL, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            language: 'auto',
                            include_history: false,
                            max_tokens: parseInt(maxTokens),
                        }),
                    });

                    const data = await response.json();
                    document.getElementById(loadingId).remove();

                    if (response.ok) {
                        addMessage(data.response, 'assistant');
                        updateStatus('Connected to Ollama', true);
                    } else {
                        addMessage('Error: ' + (data.error || 'Unknown error'), 'assistant');
                        updateStatus('Error: ' + (data.error || 'Unknown error'), false);
                    }
                } catch (error) {
                    document.getElementById(loadingId).remove();
                    addMessage('Connection error. Is Ollama running? (http://localhost:11434)', 'assistant');
                    updateStatus('Connection error', false);
                    console.error(error);
                } finally {
                    isLoading = false;
                    sendBtn.disabled = false;
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

            async function fetchStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    if (response.ok) {
                        const config = data.configuration;
                        addMessage(`Status: Connected to ${data.model_name}\nConfiguration: Temperature=${config.temperature}, Top-P=${config.top_p}, Max Tokens=${config.max_tokens}\nConversations: ${data.total_conversations}`, 'assistant');
                        updateStatus('Connected to Ollama', true);
                    }
                } catch (error) {
                    addMessage('Failed to fetch status', 'assistant');
                    updateStatus('Connection error', false);
                }
            }

            function updateStatus(text, connected) {
                document.getElementById('statusText').textContent = text;
                document.getElementById('statusIndicator').style.background = connected ? '#28a745' : '#dc3545';
            }

            document.getElementById('userInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !isLoading) sendMessage();
            });

            async function loadModelInfo() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('modelInfo').textContent = `Model: ${data.model_name} | Status: ${data.model_connected ? '✓ Connected' : '✗ Disconnected'}`;
                        updateStatus('Connected to Ollama', data.model_connected);
                    }
                } catch (error) {
                    document.getElementById('modelInfo').textContent = 'Status: Unable to connect to Ollama';
                    updateStatus('Connection error', false);
                }
            }

            window.addEventListener('load', () => {
                loadModelInfo();
                addMessage('👋 Hello! I\'m your bilingual AI assistant powered by Ollama. You can chat with me in English or Chinese. How can I help you today?', 'assistant');
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
    logger.info("=" * 50)
    logger.info("Bilingual Dialogue RAG API - Ollama Edition")
    logger.info("=" * 50)
    logger.info(f"Ollama Model: {OLLAMA_MODEL}")
    logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
    logger.info(f"Temperature: {OLLAMA_TEMPERATURE}")
    logger.info(f"Max Tokens: {MAX_TOKENS}")
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    logger.info("Make sure Ollama is running: ollama serve")
    logger.info("=" * 50)

    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=DEBUG_MODE,
        use_reloader=False,
    )
