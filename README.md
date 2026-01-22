# 🎬 Movie Plot RAG System

A lightweight **Retrieval-Augmented Generation (RAG)** system that answers questions about movie plots using a subset of the Wikipedia Movie Plots dataset.

This project demonstrates the complete RAG pipeline: **Data → Embeddings → Vector Store → Retrieval → LLM → Structured Output**.

---

## ✨ Features

- **Load & Preprocess** a subset (~300 rows) of the Wikipedia Movie Plots dataset
- **Chunk** long plot texts (~300 words per chunk) using LangChain text splitters
- **Embed & Store** chunks in ChromaDB vector store using HuggingFace sentence transformers
- **Retrieve** top-k relevant chunks given a user query
- **Generate** answers using Google Gemini LLM with retrieved context
- **Structured JSON Output** with `answer`, `contexts`, and `reasoning` fields

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.13+**
- **Docker** and **Docker Compose** (for ChromaDB)
- **uv** - Fast Python package installer and resolver

### Installing uv

If you don't have `uv` installed, follow the official installation guide:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

📖 **Full documentation:** [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)

---

## 🔑 Getting a Google Gemini API Key

This project uses Google's Gemini LLM for answer generation. You'll need a free API key:

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key

📖 **Gemini API documentation:** [https://ai.google.dev/docs](https://ai.google.dev/docs)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/hasithdd/movie-chatbot.git
cd movie-chatbot
```

### 2. Install Dependencies

Using `uv` to sync dependencies from the lockfile:

```bash
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies from `uv.lock`

### 3. Set Up Environment Variables

Copy the example environment file and add your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your actual API key:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 4. Start ChromaDB (Vector Store)

Start the ChromaDB Docker container:

```bash
docker-compose up -d
```

Verify it's running:

```bash
docker-compose ps
```

You should see the `chroma` service running on port `8000`.

### 5. Ingest Data

Load the movie plots, chunk them, generate embeddings, and store in ChromaDB:

```bash
uv run python main.py --ingest
```

This will:
- Load ~300 movie plots from the dataset
- Chunk the plots into ~593 text segments
- Generate embeddings using `all-MiniLM-L6-v2`
- Store embeddings in ChromaDB

### 6. Start Querying

Run the interactive query loop:

```bash
uv run python main.py
```

Ask questions about movies:

```
Ask a question about a movie plot: Tell me about the Ringmaster movie

Thinking...
{
  "answer": "The movie \"Ringmaster\" begins with Prince (Dileep), a dog trainer...",
  "contexts": ["The movie starts with Prince (Dileep)..."],
  "reasoning": "I extracted all plot details related to the movie 'Ringmaster'..."
}
```

Type `exit` or `quit` to stop.

---

## 📁 Project Structure

```
movie-chatbot/
├── main.py                      # Main entry point (ingest & query loop)
├── pyproject.toml               # Project configuration & dependencies
├── uv.lock                      # Locked dependencies
├── docker-compose.yml           # ChromaDB Docker configuration
├── .env.example                 # Environment variables template
├── .env                         # Your environment variables (not in git)
├── movie_dataset/               # Dataset files
│   ├── wiki_movie_plots_deduped.csv
│   └── processed_movie_plots.csv
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── chroma_data/                 # ChromaDB persistent storage
└── src/
    ├── data/
    │   ├── loader.py            # Load & preprocess movie data
    │   ├── chunker.py           # Chunk plot texts
    │   ├── embeddings/
    │   │   └── sentence_transformer.py  # HuggingFace embeddings
    │   ├── vectorstore/
    │   │   └── chroma.py        # ChromaDB integration
    │   └── rag/
    │       └── rag.py           # RAG pipeline (retrieve + generate)
    └── llm/
        └── llm.py               # Google Gemini LLM setup
```

---

## 🛠️ Configuration

### Adjustable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `n_rows` | `loader.py` | 300 | Number of movie plots to load |
| `chunk_size` | `chunker.py` | 1800 | Characters per chunk (~300 words) |
| `chunk_overlap` | `chunker.py` | 200 | Overlap between chunks |
| `k` | `rag.py` | 5 | Number of chunks to retrieve |
| `model` | `llm.py` | `gemini-2.5-flash` | Gemini model to use |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `polars` | Fast DataFrame operations for data loading |
| `langchain-text-splitters` | Text chunking |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `sentence-transformers` | `all-MiniLM-L6-v2` embedding model |
| `chromadb` | Vector database |
| `langchain-chroma` | LangChain ChromaDB integration |
| `langchain-google-genai` | Google Gemini LLM integration |
| `pydantic` | Structured output validation |

---

## 🐳 Docker Commands

```bash
# Start ChromaDB
docker-compose up -d

# Stop ChromaDB
docker-compose down

# View logs
docker-compose logs -f chroma

# Reset ChromaDB data (requires sudo on Linux)
docker-compose down
sudo rm -rf chroma_data/*
docker-compose up -d
```

---

## 📊 Example Output

```json
{
  "answer": "The movie *2001: A Space Odyssey* features an artificial intelligence system called HAL 9000.",
  "contexts": [
    "2001: A Space Odyssey ... The HAL 9000 computer becomes antagonistic ..."
  ],
  "reasoning": "The question asked about AI. I searched the plots, found '2001: A Space Odyssey' with HAL 9000, and used it to form the answer."
}
```

---

## 🧪 Development

### Pre-commit Hooks

This project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

### Linting & Formatting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
uv run ty check
```

---

## 🔧 Troubleshooting

### ChromaDB Connection Error

```
Could not connect to Vector Store. Make sure Docker is running.
```

**Solution:** Ensure Docker is running and ChromaDB is started:
```bash
docker-compose up -d
```

### Permission Denied When Deleting chroma_data

```
rm: cannot remove 'chroma_data/...': Permission denied
```

**Solution:** Use sudo (files are owned by Docker):
```bash
sudo rm -rf chroma_data/*
```

### Module Not Found Error

**Solution:** Ensure you're using the virtual environment:
```bash
uv run python main.py --ingest
```

Or activate it manually:
```bash
source .venv/bin/activate
python main.py --ingest
```

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- [Wikipedia Movie Plots Dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Google Gemini](https://ai.google.dev/)
- [Astral uv](https://docs.astral.sh/uv/)
