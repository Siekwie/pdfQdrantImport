# PDF Qdrant Import Tool

This tool is used to vectorize PDF files and import them into Qdrant.

## Features

- Extracts text and layout from PDF files using `pymupdf4llm`
- Converts PDF content to Markdown to eliminate artifacts and preserve structure
- Uses `pymupdf.layout` for improved detection of multi-column text, headers, and footers
- Splits text into chunks while preserving page number metadata
- Generates embeddings using sentence transformers
- Imports vectors and metadata into Qdrant vector database

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Configure Qdrant connection:
   - Copy `config.example.env` to `.env`
   - Update with your Qdrant URL and API key if needed

## Usage

### Basic Usage

Place your PDF files in the `inputFiles` directory and run:

```bash
python pdf_import.py
```

### Command Line Options

```bash
python pdf_import.py [OPTIONS]
```

Options:
- `--input-dir`: Directory containing PDF files (default: `inputFiles`)
- `--qdrant-url`: Qdrant server URL (default: `http://localhost:6333` or `QDRANT_URL` env var)
- `--qdrant-api-key`: Qdrant API key (or use `QDRANT_API_KEY` env var)
- `--collection`: Qdrant collection name (default: `pdf_documents`)
- `--model`: Sentence transformer model name (default: `all-MiniLM-L6-v2`)
- `--chunk-size`: Text chunk size in characters (default: 500)
- `--chunk-overlap`: Text chunk overlap in characters (default: 50)

### Examples

Process PDFs with custom collection name:
```bash
python pdf_import.py --collection my_documents
```

Use a different embedding model:
```bash
python pdf_import.py --model all-mpnet-base-v2
```

Connect to Qdrant Cloud:
```bash
python pdf_import.py --qdrant-url https://your-cluster.qdrant.io --qdrant-api-key your-api-key
```

## How It Works

1. **PDF Markdown Extraction**: Uses `pymupdf4llm` and `pymupdf.layout` to extract clean Markdown text from each page, preserving structure and eliminating artifacts.
2. **Text Chunking**: Splits Markdown text into overlapping chunks, maintaining page-level metadata.
3. **Embedding Generation**: Uses sentence transformers to generate vector embeddings for each chunk.
4. **Qdrant Import**: Uploads vectors along with rich metadata (source, page number, chunk index) to Qdrant.

## Output

Each PDF chunk is stored in Qdrant with:
- Vector embedding
- Original text chunk
- Source file name and path
- Chunk index and total chunks
- Import timestamp

## Requirements

- Python 3.8+
- Qdrant server (local or cloud)
- PDF files to process

## Notes

- The default embedding model (`all-MiniLM-L6-v2`) is fast and efficient
- For better quality embeddings, consider using `all-mpnet-base-v2` (also better at coding results)
- Chunk size and overlap can be adjusted based on your document characteristics