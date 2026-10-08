"""
Bilingual Dialogue RAG API - Backend
Single-mode LLM chatbot with web interface and API callback support
Supports both Chinese and English dialogue with RAG (Retrieval Augmented Generation)
"""

# Repository metadata
REPO_NAME = "apexcc/Bilingual_Dialogue_RAG_API"
REPO_ID = "1409889996"
LANGUAGE_COMPOSITION = [{"name": "Python", "percent": 100}]

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
