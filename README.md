# Smart-Librarian - Backend (FastAPI + Chroma + OpenAI)
## Description

A lightweight API that powers the Smart Librarian app.
It uses a local Chroma vector store with OpenAI embeddings to retrieve and recommend books based on semantic similarity, and serves those results via a FastAPI endpoint.

## 🚀 Getting Started
### Clone & Set Up the Environment
```commandline
git clone <repo-url>
cd <repo-url>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Set up a working OpenAI API key in a .env file:
```commandline
OPENAI_API_KEY= your-api-key
```
### Build the Vector Index (first run)
```
python ingest.py
```
This reads data/book_summaries.json, embeds all book summaries, and stores them in .chroma/ for semantic retrieval.
Whenever book_summaries.json is updated, re-run this command to keep the embeddings in sync.

### Run the API Server
```commandline
uvicorn main:app --reload --port 8000
```

### Rebuild Index When Dataset Changes
```
Remove-Item -Recurse -Force .chroma
python ingest.py
```

### Set up the config.py file with your local paths
```python
DATA_PATH = "your-data-path"
CHROMA_PATH = "./chroma" # This should be your chroma path if installed correctly
CHAT_MODEL = "gpt-4o-mini"; # You can choose any desired model, a cheaper one is recommended for cost efficiency
```




