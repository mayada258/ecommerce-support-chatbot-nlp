"""
Deployment script — RAG-Based E-commerce Customer Support Chatbot

Combines the four pipeline stages (as required by the task):
1) Language Detection
2) Sentiment/Emotion Classification
3) Intent Classification
4) Q&A RAG

Guideline followed: complaint / negative-sentiment messages are routed
distinctly — an apology/acknowledgment is prepended before the
RAG-generated answer (rather than escalating to a human), because a
locally-deployed demo has no human agent to escalate to.

Run with:
    python app.py

Then send a POST request to http://localhost:5000/chat with JSON body:
    {"message": "your customer message here"}
"""

import os
import pickle

import joblib
import faiss
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Load all trained artifacts
# ---------------------------------------------------------------------------

# Language Detection (TF-IDF + Logistic Regression)
language_model = joblib.load("language_detection_model.pkl")
language_vectorizer = joblib.load("language_detection_vectorizer.pkl")

# Sentiment / Emotion (LSTM)
sentiment_model = load_model("sentiment_emotion_model.h5")
with open("sentiment_tokenizer.pkl", "rb") as f:
    sentiment_tokenizer = pickle.load(f)

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NEGATIVE_EMOTIONS = {"sadness", "anger", "fear"}
SENTIMENT_MAX_LEN = 50

# Intent Classifier (TF-IDF + Logistic Regression)
intent_model = joblib.load("intent_classifier_model.pkl")
intent_vectorizer = joblib.load("intent_classifier_vectorizer.pkl")

# Q&A RAG (FAISS + sentence-transformers + Groq)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rag_index = faiss.read_index("faiss_index.index")
with open("rag_responses.pkl", "rb") as f:
    rag_responses = pickle.load(f)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """System: "You are a helpful, professional customer support assistant
for an online retailer. Answer the customer's question using ONLY
the information in the retrieved support responses below. If the
customer sounds frustrated ({detected_sentiment}), acknowledge
that before answering. If the retrieved context does not cover
the question, say so honestly and offer to escalate to a human
agent rather than guessing."

Context (retrieved past support responses):
{retrieved_chunk_1}
{retrieved_chunk_2}
{retrieved_chunk_3}

Customer question: "{user_message}"
"""


# ---------------------------------------------------------------------------
# 2. Pipeline stage functions
# ---------------------------------------------------------------------------

def detect_language(message):
    vec = language_vectorizer.transform([message])
    return language_model.predict(vec)[0]


def detect_sentiment(message):
    seq = sentiment_tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=SENTIMENT_MAX_LEN)
    probs = sentiment_model.predict(padded, verbose=0)
    emotion = EMOTION_LABELS[int(np.argmax(probs))]
    return emotion


def classify_intent(message):
    vec = intent_vectorizer.transform([message])
    return intent_model.predict(vec)[0]


def retrieve_context(message, k=3):
    query_embedding = embedding_model.encode([message], convert_to_numpy=True).astype("float32")
    _, indices = rag_index.search(query_embedding, k)
    return [rag_responses[i] for i in indices[0]]


def generate_rag_answer(message, emotion):
    retrieved_chunks = retrieve_context(message, k=3)

    prompt = PROMPT_TEMPLATE.format(
        detected_sentiment=emotion,
        retrieved_chunk_1=retrieved_chunks[0],
        retrieved_chunk_2=retrieved_chunks[1],
        retrieved_chunk_3=retrieved_chunks[2],
        user_message=message
    )

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# 3. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(message):
    language = detect_language(message)
    emotion = detect_sentiment(message)
    intent_category = classify_intent(message)

    is_negative = emotion in NEGATIVE_EMOTIONS
    is_complaint = intent_category == "complaint"

    if intent_category == "greeting":
        response = "Hello! How can I help you today?"

    elif is_complaint or is_negative:
        # Guideline: complaint/negative-sentiment messages are routed
        # distinctly — an apology is prepended before the RAG answer.
        rag_answer = generate_rag_answer(message, emotion)
        response = (
            "I'm sorry to hear about the trouble you're experiencing — "
            "let's get this sorted out for you.\n\n" + rag_answer
        )

    else:
        response = generate_rag_answer(message, emotion)

    return {
        "detected_language": language,
        "detected_emotion": emotion,
        "intent_category": intent_category,
        "response": response
    }



# ---------------------------------------------------------------------------
# 4. Flask route
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Request body must include a 'message' field."}), 400

    result = run_pipeline(data["message"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
