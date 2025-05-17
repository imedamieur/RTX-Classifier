"""
RetrieveNode for the RTX Classifier.

Performs semantic search over pre-embedded AAOIFI FAS 4/7/10/28/32 corpus using ChromaDB.
"""
from typing import Dict, List, Any, Optional
import os
from dataclasses import dataclass
import dotenv

# Load environment variables
dotenv.load_dotenv()

import chromadb
import numpy as np
import cohere
from langchain_openai import OpenAIEmbeddings

from rtx_classifier.vectorstore import get_or_create_vectorstore, get_embedding_function

# Get configuration from environment variables
DEFAULT_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "aaoifi_standards")
DEFAULT_PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./vectorstore")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "rerank-english-v3.0")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector database."""
    text: str
    standard: str  # e.g., "FAS4", "FAS7", etc.
    paragraph: str  # e.g., "3.1.2"
    score: float


class RetrieveNode:
    """
    Retrieval node that performs semantic search over AAOIFI standards.
    Uses ChromaDB as the vector database and optionally Cohere's reranker
    for improved results quality.
    """
    
    def __init__(
        self, 
        docs_dir: Optional[str] = None,
        collection_name: str = None,
        persist_directory: Optional[str] = None,
        embedding_function_name: str = "openai",
        top_k: int = None,
        use_reranker: bool = None,
    ):
        """
        Initialize the RetrieveNode.
        
        Args:
            docs_dir: Directory containing PDF documents
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
            embedding_function_name: Name of the embedding function to use
            top_k: Number of documents to retrieve
            use_reranker: Whether to use Cohere's reranker for improved results
        """
        self.docs_dir = docs_dir
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        
        # Handle persist_directory=True to use the environment variable
        if persist_directory is True:
            self.persist_directory = DEFAULT_PERSIST_DIRECTORY
        else:
            self.persist_directory = persist_directory
            
        self.embedding_function_name = embedding_function_name
        self.top_k = top_k or TOP_K_RETRIEVAL
        self.use_reranker = use_reranker if use_reranker is not None else USE_RERANKER
        self._client = None
        self._collection = None
        self._embeddings = None
        self._langchain_embeddings = None
        self._cohere_client = None
    
    def _get_collection(self):
        """Initialize and return the ChromaDB collection."""
        if self._collection is None:
            # Initialize the vector store (creates or loads existing)
            self._client, self._collection = get_or_create_vectorstore(
                docs_dir=self.docs_dir,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function_name=self.embedding_function_name
            )
                
        return self._collection
    
    def _get_embeddings(self):
        """Get the embedding function for queries."""
        if self._embeddings is None:
            # Use the same embedding function as the collection
            self._embeddings = get_embedding_function(self.embedding_function_name)
                
        return self._embeddings
    
    def _get_langchain_embeddings(self):
        """Get LangChain embeddings for embed_query."""
        if self._langchain_embeddings is None:
            # Use LangChain's OpenAI embeddings
            model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            self._langchain_embeddings = OpenAIEmbeddings(model=model_name)
        return self._langchain_embeddings
    
    def _get_cohere_client(self):
        """Initialize and return the Cohere client."""
        if self._cohere_client is None:
            if not COHERE_API_KEY:
                print("Warning: COHERE_API_KEY environment variable is not set.")
                print("Using a fake client for demonstration purposes only.")
                # Create a mock client for demonstration
                self._cohere_client = None
            else:
                self._cohere_client = cohere.Client(COHERE_API_KEY)
        return self._cohere_client
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Embed the query for semantic search.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector for the query
        """
        try:
            # First try LangChain embeddings which are guaranteed to have embed_query method
            langchain_embeddings = self._get_langchain_embeddings()
            return langchain_embeddings.embed_query(query)
        except Exception as e:
            # If that fails, try different approaches with the ChromaDB embedding function
            embeddings = self._get_embeddings()
            
            try:
                # Try ChromaDB OpenAIEmbeddingFunction style - it uses a direct function call
                if callable(embeddings):
                    # ChromaDB embedding functions are callable directly with a list of strings
                    return embeddings([query])[0]
                # Try embed_documents method (some embedding functions have this)
                elif hasattr(embeddings, 'embed_documents'):
                    return embeddings.embed_documents([query])[0]
                # Final fallback - try direct embedding with __call__
                else:
                    return embeddings([query])[0]
            except Exception as nested_e:
                raise Exception(f"All embedding methods failed. Primary error: {str(e)}. Fallback error: {str(nested_e)}")
    
    def _format_entries_for_search(self, state: Dict[str, Any]) -> str:
        """
        Format the transaction entries for semantic search.
        
        Args:
            state: Current state with transaction entries
            
        Returns:
            Formatted query string
        """
        entries = state.get("entries", [])
        formatted_entries = []
        
        for entry in entries:
            account = entry.get("account", "")
            debit = entry.get("debit", 0.0)
            credit = entry.get("credit", 0.0)
            
            if debit > 0:
                formatted_entries.append(f"Dr. {account}: {debit}")
            if credit > 0:
                formatted_entries.append(f"Cr. {account}: {credit}")
        
        query = "Transaction entries:\n" + "\n".join(formatted_entries)
        
        # Add optional context if available
        context = state.get("context")
        if context:
            query += f"\n\nContext: {context}"
            
        adjustments = state.get("adjustments")
        if adjustments:
            query += f"\n\nAdjustments: {adjustments}"
            
        treatment = state.get("accounting_treatment")
        if treatment:
            query += f"\n\nAccounting treatment: {treatment}"
            
        return query
    
    def _query_vectorstore(self, query_text: str) -> List[RetrievedDocument]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: Query text to search for
            
        Returns:
            List of retrieved documents with metadata
        """
        collection = self._get_collection()
        
        # Get query embedding using LangChain embeddings
        query_embedding = self._embed_query(query_text)
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        for i in range(len(results["documents"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity score (1 - normalized distance)
            score = 1.0 - min(distance, 1.0)
            
            doc = RetrievedDocument(
                text=text,
                standard=metadata.get("standard", ""),
                paragraph=metadata.get("paragraph", ""),
                score=score
            )
            documents.append(doc)
        
        # Apply Cohere's reranking if enabled
        if self.use_reranker:
            documents = self._apply_reranking(query_text, documents)
            
            # Sort documents by score in descending order after reranking
            documents.sort(key=lambda x: x.score, reverse=True)
            
        return documents
    
    def _apply_reranking(self, query_text: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Apply Cohere's reranking to improve document relevance.
        
        Args:
            query_text: The query text
            documents: List of retrieved documents
            
        Returns:
            Reranked list of documents
        """
        if not self.use_reranker or not documents:
            return documents
            
        try:
            co = self._get_cohere_client()
            
            # Extract document texts for reranking
            texts = [doc.text for doc in documents]
            
            # Use Cohere's reranker
            rerank_results = co.rerank(
                query=query_text,
                documents=texts,
                model=RERANKER_MODEL,
                top_n=len(texts)  # Get all results reranked
            )
            
            # Create a new list of documents based on reranking
            reranked_docs = []
            for result in rerank_results.results:
                # Find the original document matching this reranked result
                orig_doc_idx = result.index
                orig_doc = documents[orig_doc_idx]
                
                # Create a new document with the updated score
                reranked_doc = RetrievedDocument(
                    text=orig_doc.text,
                    standard=orig_doc.standard,
                    paragraph=orig_doc.paragraph,
                    score=result.relevance_score  # Use Cohere's relevance score
                )
                reranked_docs.append(reranked_doc)
                
            return reranked_docs
            
        except Exception as e:
            # Log the error but continue with the original documents
            print(f"Reranker error: {str(e)}")
            return documents
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and retrieve relevant documents.
        
        Args:
            state: Input state with transaction entries
            
        Returns:
            Updated state with retrieved documents
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)
        
        # Initialize default output fields.
        # These will be populated if retrieval is successful,
        # or remain empty if there's an error, insufficient input, or upstream failure.
        new_state["retrieved_texts"] = []
        new_state["sources"] = []

        # Check if a previous node already marked the state as invalid.
        # If "valid" is not in state or is False, this node should not proceed.
        # The "valid" status and any "error_message" from upstream are preserved.
        if not state.get("valid", False):
            new_state["valid"] = state.get("valid", False) # Ensure 'valid' is explicitly carried over
            # error_message should ideally be set by the upstream node that invalidated the state.
            return new_state
        
        # If we reach here, state["valid"] was True from upstream.
        # Set 'valid' to True for this node's operations by default.
        # It will be set to False if an issue occurs specifically within this node.
        new_state["valid"] = True

        try:
            entries = new_state.get("entries", [])
            context = new_state.get("context")
            adjustments = new_state.get("adjustments")
            treatment = new_state.get("accounting_treatment")

            # Check if entries contain any actual debit or credit information.
            # An entry is considered meaningful if it's a dictionary and has a positive debit or credit.
            has_meaningful_entries = any(
                isinstance(entry, dict) and (entry.get("debit", 0.0) > 0 or entry.get("credit", 0.0) > 0)
                for entry in entries
            )

            # If there are no meaningful entries AND no other contextual information,
            # then retrieval is unlikely to be effective.
            if not has_meaningful_entries and not context and not adjustments and not treatment:
                new_state["error_message"] = (
                    "Insufficient information for retrieval: Please provide meaningful transaction entries, "
                    "context, adjustments, or accounting treatment."
                )
                new_state["valid"] = False
                return new_state # retrieved_texts and sources remain empty

            # Format the query
            query = self._format_entries_for_search(new_state)
            
            # Query the vector store
            documents = self._query_vectorstore(query)
            
            reranking_status = "with reranking" if self.use_reranker else "without reranking"
            print(f"Retrieved documents ({reranking_status}): {len(documents)} results")
            
            # Add debug information about top results
            if documents:
                print(f"Top result: {documents[0].standard} ¶{documents[0].paragraph} (score: {documents[0].score:.4f})")
                if len(documents) > 1:
                    print(f"Second result: {documents[1].standard} ¶{documents[1].paragraph} (score: {documents[1].score:.4f})")
            
            # Extract texts and add to state
            new_state["retrieved_texts"] = [doc.text for doc in documents]
            
            # Also store the document metadata for citing sources
            new_state["sources"] = [ # Changed "retrieved_sources" to "sources"
                f"{doc.standard} ¶{doc.paragraph}" for doc in documents
            ]
            
            # Store scores for evaluation/debugging
            new_state["retrieval_scores"] = [doc.score for doc in documents]
            
            # If documents list is empty, retrieved_texts and sources will correctly be empty.
            # This is a valid outcome (no results found) and not an error in itself.
            # The 'valid' flag remains True.
            
        except Exception as e:
            new_state["error_message"] = f"Retrieval error: {str(e)}"
            new_state["valid"] = False
            # retrieved_texts and sources are already initialized to empty lists due to the setup at the method's start.
            
        return new_state