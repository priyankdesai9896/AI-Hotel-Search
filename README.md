# Semantic Search with Couchbase and LangChain

This project provides a framework to embed and store text documents (e.g., hotel data) into Couchbase using `LangChain`, `sentence-transformers`, and `Torch` for semantic search.

It includes utilities to:
- Generate or load embeddings
- Store them in a Couchbase database
- Query them using LangChain's vector store abstractions
- Interface with the system using Streamlit

---

## 📁 Project Structure

```
.
├── config.py               # Configuration for Couchbase connection and constants
├── embedding_loader.py     # Embedding loader and Couchbase integration logic
├── requirements.txt        # Python dependencies
├── hotel_search_app.py     # Main application
```

---

## 🧠 Features

- 🔗 Uses LangChain's vector store abstraction
- 🧠 Embeds documents using Sentence Transformers
- 🗃️ Stores and queries embeddings using Couchbase
- 🖥️ Supports interactive interface via Streamlit (optional)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

Make sure you’re using Python 3.8 or above.

```bash
pip install -r requirements.txt
```

### 3. Set Couchbase Configuration

Update `config.py` with your Couchbase Capella or local database configuration:

```python
COUCHBASE_HOST = "your-host-url"
COUCHBASE_BUCKET = "your-bucket-name"
COUCHBASE_USERNAME = "your-username"
COUCHBASE_PASSWORD = "your-password"
```

### 4. Run the Embedding Loader

```bash
python embedding_loader.py
```

This will:
- Load/upsert documents in place 
- Embed "description" field in the hotel collection using sentence transformer model (all-MiniLM-L6-v2) and store them in "description_embedding" field
- Store them in Couchbase as vector documents

### 5. Launch Streamlit App (Optional)

If your project includes a UI:

```bash
streamlit run hotel_search_app.py
```

*(Add `app.py` if needed.)*

---

## 🧪 Tech Stack

- **LangChain**: for vector store and retrieval abstraction
- **Sentence Transformers**: for generating semantic embeddings
- **Couchbase**: vector-enabled document database
- **Torch**: backend for transformers
- **Streamlit**: optional UI for interaction

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for a full list:

```txt
couchbase==4.4.0
sentence-transformers
langchain
rich
pandas
numpy
streamlit
torch==2.2.2
```

---

## 🔐 Security

Avoid checking sensitive credentials like Couchbase passwords into version control. Use `.env` or secrets manager in production.

---

## ✨ Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [SentenceTransformers](https://www.sbert.net/)
- [Couchbase Capella](https://www.couchbase.com/products/capella)