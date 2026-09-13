# RAG-Based E-commerce Customer Support Chatbot

NLP Final Project — a Retrieval-Augmented Generation (RAG) chatbot for e-commerce
customer support. The system routes every customer message through four
integrated NLP stages before producing a grounded, tone-appropriate response.

## Pipeline

1. **Language Detection** — TF-IDF/CountVectorizer classifier
   (dataset: `papluca/language-identification`)
2. **Sentiment / Emotion Classification** — RNN or Transformer classifier
   (dataset: `dair-ai/emotion`)
3. **Intent Classification** — supervised classifier on gold intent labels
   (dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`)
4. **Q&A RAG** — retrieval + generation grounded in the support knowledge base
   (embeddings: `sentence-transformers/all-MiniLM-L6-v2`,
   vector store: Qdrant / FAISS / Chroma, LLM: Groq `gpt-oss-120b` / `gpt-oss-20b`)

## Project Structure

```
chatbot-project/
├── notebooks/          # one notebook per module
│   ├── 01_language_detection.ipynb
│   ├── 02_sentiment_classifier.ipynb
│   ├── 03_intent_classifier.ipynb
│   └── 04_rag_pipeline.ipynb
├── src/                # shared/reusable python modules
├── deployment/         # Flask/FastAPI app + inference pipeline
├── data/               # raw/processed data (gitignored, see below)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file (not committed) with:

```
GROQ_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url        # only if using Qdrant cloud
QDRANT_API_KEY=your_qdrant_key    # only if using Qdrant cloud
```

## Running the API

```bash
cd deployment
python app.py
```

## Design Decisions

- **Intent categories**: the 27 fine-grained intents from the Bitext dataset
  are condensed into 7 routing categories: `greeting`, `order_status`,
  `order_management`, `billing_and_refunds`, `account_management`,
  `complaint`, `out_of_scope`.
- **Complaint / negative-sentiment handling**: _(document your choice here —
  e.g. prepend an apology before the RAG answer, or escalate to a human)_.

## Notes

This is an end-to-end integrated system — each module's output feeds the
next, rather than four isolated tasks.
