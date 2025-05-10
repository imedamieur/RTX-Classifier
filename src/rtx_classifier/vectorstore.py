"""
VectorStore setup module for RTX Classifier.

This module handles loading PDF documents, extracting text,
creating embeddings, and setting up a local ChromaDB vector store.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Constants for document chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Get configuration from environment variables
DEFAULT_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aaoifi_standards")
DEFAULT_PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./vectorstore")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def extract_standard_info(file_path: Path) -> Tuple[str, str]:
    """
    Extract the standard name and type from the PDF filename.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of (standard_name, standard_type)
    """
    filename = file_path.stem
    
    # Extract standard name (FAS4, FAS10, etc.)
    standard_match = re.search(r'(FAS\d+|SS\d+)', filename, re.IGNORECASE)
    standard_name = standard_match.group(1).upper() if standard_match else "UNKNOWN"
    
    # Determine if it's a Financial Accounting Standard or Shari'ah Standard
    if standard_name.startswith('FAS'):
        standard_type = "Financial Accounting Standard"
    elif standard_name.startswith('SS'):
        standard_type = "Shari'ah Standard"
    else:
        standard_type = "Unknown Standard"
        
    return standard_name, standard_type


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extract text from a PDF file and split into chunks with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of text chunks with metadata
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Extract standard info from filename
    path_obj = Path(pdf_path)
    standard_name, standard_type = extract_standard_info(path_obj)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Process chunks to extract paragraph numbers and add metadata
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Try to identify paragraph numbers using regex
        para_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', chunk.page_content)
        paragraph = para_match.group(1) if para_match else f"section-{i+1}"
        
        processed_chunks.append({
            "text": chunk.page_content,
            "metadata": {
                "standard": standard_name,
                "standard_type": standard_type,
                "paragraph": paragraph,
                "page": chunk.metadata.get("page", 0) + 1  # 1-indexed page numbers
            }
        })
    
    return processed_chunks


def setup_chromadb(
    persist_directory: Optional[str] = None,
    embedding_function_name: str = "openai"
) -> chromadb.Client:
    """
    Set up and return a ChromaDB client.
    
    Args:
        persist_directory: Directory to persist the database (None for in-memory)
        embedding_function_name: Name of the embedding function to use
        
    Returns:
        ChromaDB client
    """
    # Use environment variable if persist_directory is True
    if persist_directory is True:
        persist_directory = DEFAULT_PERSIST_DIRECTORY
    
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        # Use PersistentClient when a persistence directory is specified
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        # Use in-memory client when no persistence is needed
        client = chromadb.Client()
    
    return client


def get_embedding_function(name: str = "openai", **kwargs):
    """
    Get the specified embedding function.
    
    Args:
        name: Name of the embedding function
        **kwargs: Additional arguments for the embedding function
        
    Returns:
        Embedding function
    """
    if name == "openai":
        # Get API key from environment or kwargs
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        
        # Get model name from environment or default
        model_name = kwargs.get("model_name") or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        
        # Return ChromaDB's OpenAI embedding function
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name
        )
    elif name == "default":
        # Use ChromaDB's default embedding function
        return embedding_functions.DefaultEmbeddingFunction()
    else:
        raise ValueError(f"Unsupported embedding function: {name}")


def load_docs_into_vectorstore(
    docs_dir: str,
    collection_name: str = None,
    persist_directory: Optional[str] = None,
    embedding_function_name: str = "openai",
    **kwargs
) -> Tuple[chromadb.Client, chromadb.Collection]:
    """
    Load PDF documents into a ChromaDB vector store.
    
    Args:
        docs_dir: Directory containing PDF documents
        collection_name: Name of the collection to create
        persist_directory: Directory to persist the database
        embedding_function_name: Name of the embedding function to use
        **kwargs: Additional arguments for the embedding function
        
    Returns:
        Tuple of (chromadb_client, collection)
    """
    # Use environment variables if not specified
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_NAME
        
    # Setup ChromaDB
    client = setup_chromadb(persist_directory)
    
    # Get embedding function
    embedding_func = get_embedding_function(embedding_function_name, **kwargs)
    
    # Create or get collection
    collection = None
    try:
        # Try to get existing collection
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        print(f"Retrieved existing collection: {collection_name}")
    except (ValueError, chromadb.errors.NotFoundError):
        # Collection doesn't exist, create a new one
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        print(f"Created new collection: {collection_name}")
    
    # Check if collection already has documents
    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} documents")
        return client, collection
    
    # Process PDF documents
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0
    
    # Get all PDF files from docs directory
    pdf_paths = list(Path(docs_dir).glob("*.pdf")) + list(Path(docs_dir).glob("*.PDF"))
    
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name}...")
        chunks = extract_text_from_pdf(str(pdf_path))
        
        for chunk in chunks:
            all_chunks.append(chunk["text"])
            all_metadatas.append(chunk["metadata"])
            all_ids.append(f"chunk-{chunk_id}")
            chunk_id += 1
    
    # Add documents to collection in batches
    if all_chunks:
        total_chunks = len(all_chunks)
        print(f"Adding {total_chunks} chunks to collection in batches...")
        
        # Define batch size (approximately 50,000 tokens per batch)
        # Each chunk is about CHUNK_SIZE characters, which is roughly 250-300 tokens
        # So we'll use a batch size of ~150 chunks to stay well under the limit
        batch_size = 150
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            print(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({i} to {end_idx-1})...")
            
            batch_chunks = all_chunks[i:end_idx]
            batch_metadatas = all_metadatas[i:end_idx]
            batch_ids = all_ids[i:end_idx]
            
            try:
                collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                # If this was an OpenAI error related to token limits, try with smaller batch
                if "max_tokens" in str(e).lower():
                    print("Token limit exceeded. Trying with smaller batch size...")
                    # Try with half the batch size
                    half_batch = len(batch_chunks) // 2
                    if half_batch > 0:
                        try:
                            print(f"Adding first half ({half_batch} documents)...")
                            collection.add(
                                documents=batch_chunks[:half_batch],
                                metadatas=batch_metadatas[:half_batch],
                                ids=batch_ids[:half_batch]
                            )
                            print(f"Adding second half ({len(batch_chunks) - half_batch} documents)...")
                            collection.add(
                                documents=batch_chunks[half_batch:],
                                metadatas=batch_metadatas[half_batch:],
                                ids=batch_ids[half_batch:]
                            )
                        except Exception as e2:
                            print(f"Error with half batches: {str(e2)}")
                            # In case of persistent errors, skip this batch
                            print("Skipping problematic batch...")
        
        print("Documents added successfully")
    else:
        print("No documents found to add")
        
    return client, collection


def get_or_create_vectorstore(
    docs_dir: Optional[str] = None,
    collection_name: str = None,
    persist_directory: Optional[str] = None,
    embedding_function_name: str = "openai",
    **kwargs
) -> Tuple[chromadb.Client, chromadb.Collection]:
    """
    Get or create a vector store from PDF documents.
    
    Args:
        docs_dir: Directory containing PDF documents (None to use default)
        collection_name: Name of the collection to create or retrieve
        persist_directory: Directory to persist the database
        embedding_function_name: Name of the embedding function to use
        **kwargs: Additional arguments for the embedding function
        
    Returns:
        Tuple of (chromadb_client, collection)
    """
    # Use default docs directory if not provided
    if docs_dir is None:
        docs_dir = str(Path(__file__).parent.parent.parent / "docs")
    
    # Use environment variable values if not explicitly provided
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_NAME
        
    # Use default persist directory if not provided and persistence is desired
    if persist_directory is True:
        persist_directory = DEFAULT_PERSIST_DIRECTORY
    
    return load_docs_into_vectorstore(
        docs_dir=docs_dir,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function_name=embedding_function_name,
        **kwargs
    )


if __name__ == "__main__":
    """
    Script entry point for creating the vectorstore from command line.
    """
    import argparse
    
    # Make sure environment variables are loaded
    dotenv.load_dotenv()
    
    parser = argparse.ArgumentParser(description="Load PDF documents into a vector store")
    parser.add_argument("--docs-dir", help="Directory containing PDF documents")
    parser.add_argument("--persist-dir", help=f"Directory to persist the database (default: {DEFAULT_PERSIST_DIRECTORY})")
    parser.add_argument("--collection", help=f"Collection name (default: {DEFAULT_COLLECTION_NAME})")
    parser.add_argument("--embeddings", default="openai", help="Embedding function (openai or default)")
    
    args = parser.parse_args()
    
    client, collection = get_or_create_vectorstore(
        docs_dir=args.docs_dir,
        collection_name=args.collection,
        persist_directory=args.persist_dir or True,  # Default to using persistence
        embedding_function_name=args.embeddings
    )
    
    print(f"Collection {collection.name} created with {collection.count()} documents")