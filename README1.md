# 🎓 University Academic Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application for answering university student and academic questions using a pre-built knowledge base.

The application uses:

* Streamlit for the web interface
* FAISS for vector similarity search
* Sentence Transformers for embeddings
* Groq `openai/gpt-oss-120b` for answer generation
* JSON metadata for source traceability

## Architecture

```text
University PDF Documents
        │
        ▼
Google Drive
        │
        ▼
One-Time Colab Ingestion
        │
        ├── PDF text extraction
        ├── Text chunking
        ├── Embedding generation
        ├── FAISS index creation
        └── Metadata creation
        │
        ▼
GitHub Repository
        │
        ├── faiss.index
        ├── metadata.json
        └── config.json
        │
        ▼
Streamlit Cloud
        │
        ▼
Student Question
        │
        ▼
Question Embedding
        │
        ▼
FAISS Similarity Search
        │
        ▼
Relevant Chunks
        │
        ▼
Metadata / Source Traceability
        │
        ▼
Groq GPT-OSS-120B
        │
        ▼
Answer + Sources
```

## Project Structure

```text
university-academic-rag/
│
├── app.py
├── requirements.txt
├── README.md
│
├── rag_knowledge_base/
│   ├── faiss.index
│   ├── metadata.json
│   └── config.json
│
└── .gitignore
```

## Knowledge Base

The original PDF documents are NOT included in this repository.

The PDFs are processed separately during the one-time knowledge-base creation stage.

The generated knowledge base contains:

### `faiss.index`

Contains the vector representations of the document chunks.

The application uses this file for similarity search.

### `metadata.json`

Contains the document chunks and their source information.

Each chunk contains information such as:

* source filename
* page number
* chunk number
* chunk ID
* chunk text
* file hash
* source label
* document type

This allows the application to trace retrieved information back to its original document and page.

### `config.json`

Stores the configuration used to create the vector database, including:

* embedding model
* embedding dimension
* chunk size
* chunk overlap
* FAISS index
* similarity method

## Embedding Model

The knowledge base uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The same embedding model must be used when creating query embeddings during application runtime.

The application does NOT recreate embeddings for the stored documents.

It only creates an embedding for the user's query and searches the existing FAISS index.

## Vector Search

The application uses FAISS with normalized embeddings.

The FAISS index uses inner-product similarity, which corresponds to cosine similarity when vectors are normalized.

The application retrieves the top relevant chunks from the knowledge base.

## Source Traceability

Source traceability is maintained through the relationship between FAISS vector positions and `metadata.json`.

For example:

```text
FAISS vector ID: 127
        │
        ▼
metadata.json → chunks[127]
        │
        ├── source_file: Academic_Policies.pdf
        ├── page_number: 4
        ├── chunk_number: 2
        └── text: ...
```

The retrieved source information is displayed in the Streamlit interface.

## Groq

The application uses Groq for answer generation.

Model:

```text
openai/gpt-oss-120b
```

The Groq API key is stored as an environment variable.

```text
GROQ_API_KEY
```

For Streamlit Cloud, add the key through the application's Secrets settings.

Do not hard-code the API key in `app.py`.

## Local Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project:

```bash
cd university-academic-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Groq API key.

Linux/macOS:

```bash
export GROQ_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Run the application:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application.
4. Select the GitHub repository.
5. Set the main file to:

```text
app.py
```

6. Deploy the application.
7. Open the application's Secrets settings.
8. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

9. Save the secret.
10. Restart/redeploy the application.

## Important

The original university PDFs are not required by the deployed application.

Only the generated RAG knowledge base is required:

```text
rag_knowledge_base/
├── faiss.index
├── metadata.json
└── config.json
```

If the original documents are updated, the knowledge base should be rebuilt so that the FAISS vectors and metadata correspond to the new document contents.

## Security

Never commit API keys to GitHub.

Do not put the following directly into `app.py`:

```python
GROQ_API_KEY = "..."
```

Use an environment variable instead:

```python
import os

api_key = os.environ.get("GROQ_API_KEY")
```

## Technologies

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Groq
* GPT-OSS-120B
* Retrieval-Augmented Generation (RAG)

## License

This project is intended for educational and academic use.
