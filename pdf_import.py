"""
PDF Vectorization and Qdrant Import Tool

This script processes PDF files, extracts text, generates embeddings,
and imports them into a Qdrant vector database.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import pymupdf.layout  # Activates improved layout detection
    import pymupdf4llm
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


class PDFQdrantImporter:
    """Handles PDF processing, vectorization, and Qdrant import."""
    
    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "pdf_documents",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the PDF Qdrant Importer.
        
        Args:
            qdrant_url: Qdrant server URL (default: http://localhost:6333)
            qdrant_api_key: Qdrant API key (optional)
            collection_name: Name of the Qdrant collection
            model_name: Sentence transformer model name
            chunk_size: Number of characters per text chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Qdrant client
        if self.qdrant_api_key:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        else:
            self.client = QdrantClient(url=self.qdrant_url)
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create or get collection
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
            else:
                print(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise
    
    def extract_markdown_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract markdown from a PDF file using pymupdf4llm.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        try:
            # page_chunks=True returns a list of dictionaries, one per page
            # Each dictionary has 'text', 'page_number', etc.
            md_pages = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                show_progress=False
            )
            return md_pages
        except Exception as e:
            print(f"Error extracting markdown from {pdf_path}: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF file: extract markdown, chunk it, and generate embeddings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing chunk data and metadata
        """
        print(f"Processing PDF: {pdf_path.name}")
        
        # Extract markdown pages
        md_pages = self.extract_markdown_from_pdf(pdf_path)
        if not md_pages:
            print(f"Warning: No text extracted from {pdf_path.name}")
            return []
        
        all_points = []
        global_chunk_idx = 0
        
        for page_data in md_pages:
            text = page_data.get("text", "")
            page_num = page_data.get("metadata", {}).get("page", 0) + 1 # 1-based indexing
            
            if not text.strip():
                continue
                
            # Chunk the page text
            chunks = self.chunk_text(text)
            
            if not chunks:
                continue
                
            # Generate embeddings for the chunks of this page
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            
            # Prepare points for Qdrant
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = {
                    "id": str(uuid.uuid4()),
                    "vector": embedding.tolist(),
                    "payload": {
                        "text": chunk,
                        "source_file": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page_number": page_num,
                        "chunk_index": idx,
                        "global_chunk_index": global_chunk_idx,
                        "imported_at": datetime.now().isoformat()
                    }
                }
                all_points.append(point)
                global_chunk_idx += 1
        
        print(f"Created {len(all_points)} total chunks from {pdf_path.name}")
        return all_points
    
    def import_to_qdrant(self, points: List[Dict[str, Any]]):
        """
        Import points to Qdrant collection.
        
        Args:
            points: List of point dictionaries
        """
        if not points:
            print("No points to import")
            return
        
        print(f"Importing {len(points)} points to Qdrant...")
        
        # Convert to PointStruct objects
        qdrant_points = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"]
            )
            for point in points
        ]
        
        # Batch upload (Qdrant recommends batches of 100)
        batch_size = 100
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Uploaded batch {i // batch_size + 1}/{(len(qdrant_points) + batch_size - 1) // batch_size}")
        
        print(f"Successfully imported {len(points)} points to collection '{self.collection_name}'")
    
    def process_directory(self, directory_path: Path):
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
        """
        pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF file(s) to process")
        
        all_points = []
        for pdf_file in pdf_files:
            try:
                points = self.process_pdf(pdf_file)
                all_points.extend(points)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
        
        if all_points:
            self.import_to_qdrant(all_points)
        else:
            print("No points generated from PDF files")


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vectorize PDF files and import them into Qdrant"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputFiles",
        help="Directory containing PDF files (default: inputFiles)"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant server URL (default: http://localhost:6333 or QDRANT_URL env var)"
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (or use QDRANT_API_KEY env var)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="pdf_documents",
        help="Qdrant collection name (default: pdf_documents)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Text chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Text chunk overlap in characters (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Initialize importer
    importer = PDFQdrantImporter(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        collection_name=args.collection,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process PDFs
    importer.process_directory(input_dir)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
