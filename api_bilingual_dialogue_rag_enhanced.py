"""
Bilingual Dialogue RAG API - Enhanced Backend
Single-mode LLM chatbot with web interface and API callback support
Supports both Chinese and English dialogue with RAG (Retrieval Augmented Generation)
Works with Ollama for generation and embeddings
NOW INCLUDES: Google Search integration, dynamic model selection, language control, dialogue history
"""

import os
import re
import json
import logging
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===================== Configuration =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Gemma:latest")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")

API_HOST = os.getenv("API_HOST", "203.64.95.228")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

DATA_DIR = os.getenv("DATA_DIR", "./knowledge_base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
GOOGLE_SEARCH_TOP_K = int(os.getenv("GOOGLE_SEARCH_TOP_K", "3"))

# ===================== Ollama Client =====================
class OllamaClient:
    def __init__(self, api_url: str, model_name: str, embedding_model: str):
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chat_endpoint = f"{self.api_url}/api/chat"
        self.embed_endpoint = f"{self.api_url}/api/embed"
        self.legacy_embed_endpoint = f"{self.api_url}/api/embeddings"
        self.tags_endpoint = f"{self.api_url}/api/tags"

    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(self.tags_endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "models" in data:
                    return [model.get("name", "") for model in data["models"] if model.get("name")]
            return []
        except Exception as e:
            logger.warning(f"Failed to fetch available models: {e}")
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from Ollama.
        Tries /api/embed first, then /api/embeddings fallback.
        """
        if not texts:
            return []

        payload = {
            "model": self.embedding_model,
            "input": texts if isinstance(texts, list) else [texts]
        }

        endpoints = [self.embed_endpoint, self.legacy_embed_endpoint]
        last_error = None

        for endpoint in endpoints:
            try:
                response = requests.post(endpoint, json=payload, timeout=120)
                if response.status_code != 200:
                    last_error = f"{response.status_code}: {response.text}"
                    continue

                data = response.json()
                # Ollama embed API shape can vary
                if "embeddings" in data:
                    return data["embeddings"]
                if "embedding" in data:
                    return [data["embedding"]]
                if "data" in data and isinstance(data["data"], list):
                    if data["data"] and "embedding" in data["data"][0]:
                        return [item["embedding"] for item in data["data"]]
                if "data" in data and isinstance(data["data"], list):
                    if data["data"] and isinstance(data["data"][0], list):
                        return data["data"]

                last_error = f"Unexpected embed response format: {data}"
            except Exception as e:
                last_error = str(e)

        raise RuntimeError(f"Could not embed text using Ollama. Last error: {last_error}")

    def generate(self, messages: List[Dict], model: str = None, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Generate response using specified model (or default if not provided)"""
        model_to_use = model or self.model_name
        try:
            response = requests.post(
                self.chat_endpoint,
                json={
                    "model": model_to_use,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "num_predict": max_tokens
                    }
                },
                timeout=180
            )

            if response.status_code != 200:
                logger.error(f"Ollama generation failed: {response.status_code} - {response.text}")
                return "I encountered an error generating a response."

            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Error in Ollama generation: {str(e)}")
            return "I encountered an error generating a response."

# ===================== Google Search Client =====================
class GoogleSearchClient:
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.enabled = bool(api_key and search_engine_id)
        self.endpoint = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Search Google Custom Search API
        Returns list of dicts with 'title', 'link', 'snippet'
        """
        if not self.enabled:
            logger.warning("Google Search not configured (missing GOOGLE_API_KEY or GOOGLE_SEARCH_ENGINE_ID)")
            return []

        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(num_results, 10)
            }
            response = requests.get(self.endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                return results
            else:
                logger.warning(f"Google Search API error: {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Google Search failed: {e}")
            return []

# ===================== Text Chunking =====================
def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1 and last_space > chunk_size * 0.6:
                chunk = chunk[:last_space]
                end = start + last_space
        chunks.append(chunk.strip())
        start = max(0, end - overlap)
        if len(chunks) > 0 and start == max(0, chunks[-1] and len(chunks[-1]) or 0):
            # safety fallback to avoid infinite loop
            start += 1

    return [c for c in chunks if c]

# ===================== Simple RAG Store =====================
class RAGKnowledgeBase:
    def __init__(self, ollama_client: OllamaClient, data_dir: str = DATA_DIR):
        self.ollama_client = ollama_client
        self.data_dir = Path(data_dir)
        self.documents: List[Dict] = []
        self.embeddings: List[List[float]] = []
        self.texts: List[str] = []

    def load_documents(self):
        self.documents = []
        if not self.data_dir.exists():
            logger.warning(f"Knowledge base directory does not exist: {self.data_dir}")
            return

        for file_path in sorted(self.data_dir.rglob("*")):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in {".txt", ".md", ".json", ".csv"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
                if file_path.suffix.lower() == ".json":
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            text = "\n\n".join(json.dumps(item, ensure_ascii=False) for item in parsed)
                        elif isinstance(parsed, dict):
                            text = json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        pass

                self.documents.append({
                    "title": file_path.name,
                    "path": str(file_path),
                    "content": text
                })
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")

        self.rebuild_index()

    def rebuild_index(self):
        chunks: List[str] = []
        chunk_meta: List[Dict] = []

        for doc in self.documents:
            text = doc["content"]
            for chunk in split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(chunk)
                chunk_meta.append({"title": doc["title"], "path": doc["path"], "content": chunk})

        self.texts = chunks
        if not self.texts:
            self.embeddings = []
            return

        try:
            logger.info(f"Generating embeddings for {len(self.texts)} chunks using {self.ollama_client.embedding_model}")
            self.embeddings = self.ollama_client.embed_texts(self.texts)
        except Exception as e:
            logger.warning(f"Embedding generation failed, falling back to keyword search: {e}")
            self.embeddings = []

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self.texts:
            return []

        if self.embeddings:
            try:
                query_embedding = self.ollama_client.embed_texts([query])[0]
                scored = []

                for idx, doc_embedding in enumerate(self.embeddings):
                    similarity = cosine_similarity(query_embedding, doc_embedding)
                    scored.append({
                        "index": idx,
                        "score": similarity,
                        "text": self.texts[idx]
                    })

                scored.sort(key=lambda x: x["score"], reverse=True)
                results = []
                for item in scored[:top_k]:
                    results.append({
                        "text": item["text"],
                        "score": item["score"]
                    })
                return results
            except Exception as e:
                logger.warning(f"Semantic retrieval failed: {e}")

        # Fallback keyword search
        query_terms = extract_terms(query)
        if not query_terms:
            return [{"text": self.texts[0], "score": 0.0}]

        scored = []
        for idx, text in enumerate(self.texts):
            score = 0
            for term in query_terms:
                if term.lower() in text.lower():
                    score += 1
            if score > 0:
                scored.append({"index": idx, "score": score, "text": text})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [{"text": item["text"], "score": item["score"]} for item in scored[:top_k]]

    def get_context_for_query(self, query: str, top_k: int = TOP_K) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""
        return "\n\n---\n\n".join(item["text"] for item in results)

# ===================== Utility Functions =====================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def extract_terms(text: str) -> List[str]:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    stop_words = {
        "the","a","an","and","or","is","are","to","of","in","for","with","on","at","by",
        "as","be","this","that","these","those","it","its","from","your","you","we","they",
        "what","when","where","why","how","who","which","can","could","should","would",
        "please","hello","hi","thanks","thank","i","me","my","mine"
    }
    filtered = []
    for term in terms:
        if len(term) > 1 and term not in stop_words:
            filtered.append(term)
    return filtered

def detect_language(text: str) -> str:
    """
    Detect if text is primarily Chinese or English
    Returns 'chinese' or 'english'
    """
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    
    if chinese_chars > english_chars:
        return "chinese"
    return "english"

def format_dialogue_history(history: List[Dict]) -> str:
    """Format conversation history as a readable context string"""
    if not history:
        return ""
    
    formatted = "Previous conversation context:\n"
    for i, entry in enumerate(history[-10:], 1):  # Last 10 exchanges
        formatted += f"\n[Turn {i}]\n"
        formatted += f"User: {entry.get('user', '')}\n"
        formatted += f"Assistant: {entry.get('assistant', '')}\n"
    
    return formatted

# ===================== RAG LLM Wrapper =====================
class BilingualRAGLLM:
    def __init__(self, ollama_client: OllamaClient, knowledge_base: RAGKnowledgeBase, google_client: GoogleSearchClient):
        self.ollama_client = ollama_client
        self.knowledge_base = knowledge_base
        self.google_client = google_client
        self.conversation_history: List[Dict] = []

    def create_prompt(self, user_input: str, context: str = "", dialogue_history: str = "", target_language: str = None) -> List[Dict]:
        """
        Create prompt with:
        - Local knowledge base context
        - Google search results
        - Dialogue history
        - Target language instruction
        """
        # Detect language if not specified
        if target_language is None:
            target_language = detect_language(user_input)
        
        if target_language == "chinese":
            lang_instruction = "请使用中文回答。"
        else:
            lang_instruction = "Please answer in English."

        system_prompt = f"""
You are a helpful bilingual assistant that can answer questions in both Chinese and English.
{lang_instruction}

Instructions:
- Use the provided context from local knowledge base and latest web search results if available.
- If the answer is not found in the provided context, say so clearly.
- Consider the conversation history to maintain context and coherence.
- Keep the answer concise, practical, and easy to understand.
- If asked for code, provide clean, working code.
- Do not fabricate facts; rely on provided sources.
"""

        user_text = user_input.strip()
        
        # Build comprehensive context
        context_parts = []
        
        if context:
            context_parts.append(f"Local Knowledge Base Context:\n{context}")
        
        if dialogue_history:
            context_parts.append(dialogue_history)
        
        if context_parts:
            prompt_body = "\n\n---\n\n".join(context_parts) + f"\n\n---\n\nUser Question:\n{user_text}"
        else:
            prompt_body = user_text

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_body}
        ]

    def generate_response(
        self,
        user_input: str,
        model: str = None,
        top_k: int = TOP_K,
        temperature: float = 0.7,
        max_tokens: int = 512,
        enable_google_search: bool = True,
        target_language: str = None,
        include_dialogue_history: bool = True
    ) -> Dict:
        """
        Generate response with all enhancements
        Returns dict with response, sources, model used, language, etc.
        """
        
        # Retrieve local context
        local_context = self.knowledge_base.get_context_for_query(user_input, top_k=top_k)
        
        # Google search for fresh data
        google_results = []
        google_context = ""
        if enable_google_search:
            google_results = self.google_client.search(user_input, num_results=GOOGLE_SEARCH_TOP_K)
            if google_results:
                google_context_parts = []
                for i, result in enumerate(google_results, 1):
                    google_context_parts.append(
                        f"[Web Result {i}]\n"
                        f"Title: {result['title']}\n"
                        f"Link: {result['link']}\n"
                        f"Summary: {result['snippet']}"
                    )
                google_context = "\n\n".join(google_context_parts)
        
        # Combine contexts
        combined_context = ""
        if google_context and local_context:
            combined_context = f"Web Search Results (Latest):\n{google_context}\n\n---\n\nLocal Knowledge Base:\n{local_context}"
        elif google_context:
            combined_context = f"Web Search Results (Latest):\n{google_context}"
        elif local_context:
            combined_context = local_context
        
        # Include dialogue history
        dialogue_history_str = ""
        if include_dialogue_history and self.conversation_history:
            dialogue_history_str = format_dialogue_history(self.conversation_history)
        
        # Detect language if not specified
        if target_language is None:
            target_language = detect_language(user_input)
        
        # Build prompt
        messages = self.create_prompt(user_input, combined_context, dialogue_history_str, target_language)

        # Get response from Ollama
        reply = self.ollama_client.generate(messages, model=model, temperature=temperature, max_tokens=max_tokens)

        # Store in conversation memory
        self.conversation_history.append({
            "user": user_input,
            "assistant": reply,
            "timestamp": datetime.now().isoformat(),
            "context_used": bool(combined_context),
            "google_search_used": bool(google_results),
            "model_used": model or self.ollama_client.model_name,
            "language": target_language
        })

        if len(self.conversation_history) > 50:
            self.conversation_history.pop(0)

        return {
            "response": reply,
            "model_used": model or self.ollama_client.model_name,
            "language": target_language,
            "local_context_used": bool(local_context),
            "google_results": google_results,
            "dialogue_history_length": len(self.conversation_history)
        }

# ===================== Flask App =====================
app = Flask(__name__)
CORS(app)

ollama_client = OllamaClient(
    api_url=OLLAMA_API_URL,
    model_name=OLLAMA_MODEL,
    embedding_model=OLLAMA_EMBEDDING_MODEL
)

google_client = GoogleSearchClient(GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID)
knowledge_base = RAGKnowledgeBase(ollama_client, DATA_DIR)
rag_model = None

try:
    knowledge_base.load_documents()
    rag_model = BilingualRAGLLM(ollama_client, knowledge_base, google_client)
    logger.info("RAG model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG model: {e}")

# ===================== API Routes =====================
@app.route("/api/health", methods=["GET"])
def health_check():
    ollama_ok = ollama_client.test_connection()
    available_models = ollama_client.get_available_models() if ollama_ok else []
    
    return jsonify({
        "status": "ok",
        "ollama_connected": ollama_ok,
        "available_models": available_models,
        "default_model": OLLAMA_MODEL,
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "google_search_enabled": google_client.enabled,
        "knowledge_base_dir": DATA_DIR,
        "document_count": len(knowledge_base.documents),
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/api/models", methods=["GET"])
def get_models():
    """Get list of available models from Ollama"""
    try:
        available_models = ollama_client.get_available_models()
        return jsonify({
            "status": "success",
            "available_models": available_models,
            "default_model": OLLAMA_MODEL
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        if "message" not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Extract optional parameters
        model = data.get("model")  # Allow user to select model
        top_k = int(data.get("top_k", TOP_K))
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens", 512))
        enable_google_search = data.get("enable_google_search", True)
        target_language = data.get("target_language")  # 'english' or 'chinese'
        include_dialogue_history = data.get("include_dialogue_history", True)

        if rag_model is None:
            return jsonify({"error": "RAG model is not initialized"}), 500

        result = rag_model.generate_response(
            user_input=user_message,
            model=model,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_google_search=enable_google_search,
            target_language=target_language,
            include_dialogue_history=include_dialogue_history
        )

        return jsonify({
            "status": "success",
            "response": result["response"],
            "user_message": user_message,
            "model_used": result["model_used"],
            "language": result["language"],
            "local_context_used": result["local_context_used"],
            "google_results": result["google_results"],
            "dialogue_history_length": result["dialogue_history_length"],
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/index", methods=["POST"])
def reindex():
    try:
        knowledge_base.load_documents()
        return jsonify({
            "status": "success",
            "documents_loaded": len(knowledge_base.documents),
            "chunk_count": len(knowledge_base.texts)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "ollama_connected": ollama_client.test_connection(),
        "default_model": OLLAMA_MODEL,
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "google_search_enabled": google_client.enabled,
        "documents_loaded": len(knowledge_base.documents),
        "chunk_count": len(knowledge_base.texts),
        "top_k": TOP_K,
        "data_dir": DATA_DIR,
        "conversation_turns": len(rag_model.conversation_history) if rag_model else 0
    }), 200

@app.route("/api/history", methods=["GET"])
def history():
    hist = rag_model.conversation_history if rag_model else []
    limit = int(request.args.get("limit", 10))
    return jsonify({"status": "success", "history": hist[-limit:]}), 200

@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear conversation history"""
    if rag_model:
        rag_model.conversation_history = []
        return jsonify({"status": "success", "message": "Conversation history cleared"}), 200
    return jsonify({"error": "RAG model not initialized"}), 500

# ===================== Simple Web UI =====================
@app.route("/", methods=["GET"])
def web_interface():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Bilingual RAG Chat - Enhanced</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .chat-wrap {
                width: 100%;
                max-width: 960px;
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 32px 80px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #1d4ed8, #2563eb);
                color: white;
                padding: 20px 24px;
            }
            .header h1 { font-size: 24px; margin-bottom: 6px; }
            .header p { opacity: 0.9; font-size: 14px; }
            .controls {
                background: #f0f4f8;
                padding: 16px;
                border-bottom: 1px solid #e2e8f0;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 12px;
            }
            .control-group {
                display: flex;
                flex-direction: column;
            }
            .control-group label {
                font-size: 12px;
                font-weight: 600;
                color: #475569;
                margin-bottom: 4px;
            }
            .control-group select,
            .control-group input {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
                outline: none;
            }
            .control-group input[type="checkbox"] {
                width: 16px;
                height: 16px;
                cursor: pointer;
            }
            .messages {
                height: 50vh;
                overflow-y: auto;
                padding: 20px;
                background: #f8fafc;
            }
            .message {
                margin-bottom: 16px;
                display: flex;
                flex-direction: column;
            }
            .message.user { align-items: flex-end; }
            .message.assistant { align-items: flex-start; }
            .bubble {
                max-width: 75%;
                padding: 12px 14px;
                border-radius: 14px;
                line-height: 1.5;
                font-size: 15px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .message.user .bubble {
                background: #2563eb;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .message.assistant .bubble {
                background: #e2e8f0;
                color: #0f172a;
                border-bottom-left-radius: 4px;
            }
            .meta {
                font-size: 12px;
                color: #64748b;
                margin-top: 6px;
                padding: 0 4px;
            }
            .source-indicator {
                font-size: 11px;
                color: #7c3aed;
                margin-top: 4px;
            }
            .composer {
                display: flex;
                gap: 10px;
                padding: 16px 18px;
                background: white;
                border-top: 1px solid #e2e8f0;
            }
            input[type="text"] {
                flex: 1;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 12px 14px;
                font-size: 15px;
                outline: none;
            }
            button {
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 15px;
                font-weight: 600;
                background: #2563eb;
                color: white;
                cursor: pointer;
            }
            button:hover { background: #1d4ed8; }
            button.secondary {
                background: #64748b;
                padding: 8px 12px;
                font-size: 12px;
            }
            button.secondary:hover { background: #475569; }
            .status {
                padding: 10px 18px 16px;
                color: #475569;
                font-size: 12px;
                background: #fff;
                border-top: 1px solid #e2e8f0;
            }
        </style>
    </head>
    <body>
        <div class="chat-wrap">
            <div class="header">
                <h1>🤖 Bilingual RAG Chat - Enhanced</h1>
                <p>Powered by Ollama + Local KB + Google Search + Dialogue Memory</p>
            </div>

            <div class="controls">
                <div class="control-group">
                    <label for="modelSelect">Model:</label>
                    <select id="modelSelect">
                        <option value="">Default (Gemma:latest)</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="languageSelect">Target Language:</label>
                    <select id="languageSelect">
                        <option value="">Auto-detect</option>
                        <option value="english">English</option>
                        <option value="chinese">Chinese</option>
                    </select>
                </div>

                <div class="control-group">
                    <label for="googleSearch">
                        <input type="checkbox" id="googleSearch" checked /> Google Search
                    </label>
                </div>

                <div class="control-group">
                    <label for="dialogueHistory">
                        <input type="checkbox" id="dialogueHistory" checked /> Use History
                    </label>
                </div>

                <div class="control-group">
                    <button class="secondary" onclick="clearHistory()">Clear History</button>
                </div>
            </div>

            <div id="messages" class="messages"></div>

            <div class="composer">
                <input id="messageInput" type="text" placeholder="Type your message... (输入中英文都可)" />
                <button onclick="sendMessage()">Send</button>
            </div>

            <div class="status" id="statusBar">Ready</div>
        </div>

        <script>
            // Load available models on startup
            async function loadModels() {
                try {
                    const response = await fetch('/api/models');
                    const data = await response.json();
                    const select = document.getElementById('modelSelect');
                    if (data.available_models && data.available_models.length > 0) {
                        data.available_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            if (model === data.default_model) {
                                option.textContent += ' (default)';
                            }
                            select.appendChild(option);
                        });
                    }
                } catch (err) {
                    console.error('Failed to load models:', err);
                }
            }

            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const text = input.value.trim();
                if (!text) return;

                addMessage('user', text);
                input.value = '';

                const statusBar = document.getElementById('statusBar');
                statusBar.textContent = 'Thinking...';

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: text,
                            model: document.getElementById('modelSelect').value || null,
                            target_language: document.getElementById('languageSelect').value || null,
                            enable_google_search: document.getElementById('googleSearch').checked,
                            include_dialogue_history: document.getElementById('dialogueHistory').checked,
                            top_k: 4,
                            temperature: 0.7,
                            max_tokens: 512
                        })
                    });

                    const data = await response.json();
                    if (!response.ok) {
                        addMessage('assistant', 'Error: ' + (data.error || 'Unknown error'));
                        statusBar.textContent = 'Error';
                        return;
                    }

                    const sourceInfo = [];
                    if (data.google_results && data.google_results.length > 0) {
                        sourceInfo.push(`🌐 Google (${data.google_results.length} results)`);
                    }
                    if (data.local_context_used) {
                        sourceInfo.push('📚 Local KB');
                    }
                    
                    addMessage('assistant', data.response, {
                        model: data.model_used,
                        language: data.language,
                        sources: sourceInfo.join(' + ') || 'No external sources',
                        history_length: data.dialogue_history_length
                    });
                    
                    statusBar.textContent = `Ready • History: ${data.dialogue_history_length} turns • Lang: ${data.language}`;
                } catch (err) {
                    addMessage('assistant', 'Connection error. Please check Ollama is running.');
                    statusBar.textContent = 'Connection error';
                }
            }

            function addMessage(role, text, metadata = {}) {
                const messages = document.getElementById('messages');
                const wrapper = document.createElement('div');
                wrapper.className = 'message ' + role;

                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                wrapper.appendChild(bubble);

                if (role === 'assistant' && metadata) {
                    const meta = document.createElement('div');
                    meta.className = 'meta';
                    const details = [];
                    if (metadata.model) details.push(`Model: ${metadata.model}`);
                    if (metadata.language) details.push(`Lang: ${metadata.language}`);
                    if (metadata.sources) details.push(`Sources: ${metadata.sources}`);
                    if (metadata.history_length) details.push(`History: ${metadata.history_length} turns`);
                    meta.textContent = details.join(' • ');
                    wrapper.appendChild(meta);
                } else {
                    const meta = document.createElement('div');
                    meta.className = 'meta';
                    meta.textContent = 'You';
                    wrapper.appendChild(meta);
                }

                messages.appendChild(wrapper);
                messages.scrollTop = messages.scrollHeight;
            }

            async function clearHistory() {
                try {
                    const response = await fetch('/api/history', { method: 'DELETE' });
                    if (response.ok) {
                        addMessage('assistant', 'Conversation history has been cleared. Starting fresh!');
                    }
                } catch (err) {
                    console.error('Failed to clear history:', err);
                }
            }

            document.getElementById('messageInput').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') sendMessage();
            });

            // Initialize on load
            loadModels();
            addMessage('assistant', 'Hello! I am your enhanced bilingual AI assistant. I can now search Google for latest information, use dialogue memory, and support multiple models. Ask me anything in Chinese or English!');
        </script>
    </body>
    </html>
    """

# ===================== Error Handlers =====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ===================== Main Entry =====================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("ENHANCED Bilingual Dialogue RAG API")
    logger.info("Features: Google Search + Dynamic Model Selection + Language Control + Dialogue History")
    logger.info("=" * 70)
    logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
    logger.info(f"Chat model: {OLLAMA_MODEL}")
    logger.info(f"Embedding model: {OLLAMA_EMBEDDING_MODEL}")
    logger.info(f"Google Search: {'ENABLED' if google_client.enabled else 'DISABLED (set GOOGLE_API_KEY & GOOGLE_SEARCH_ENGINE_ID)'}")
    logger.info(f"Knowledge base directory: {DATA_DIR}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}, Top K: {TOP_K}")
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    logger.info("=" * 70)

    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=DEBUG_MODE,
        use_reloader=False
    )
