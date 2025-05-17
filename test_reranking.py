"""
Test script for the Cohere reranker integration.

This script instantiates a RetrieveNode and tests it with a sample query.
"""
import os
import json
import dotenv
from rtx_classifier.nodes.retrieve import RetrieveNode

# Load environment variables from .env file
dotenv.load_dotenv()

# Check if COHERE_API_KEY is set
if not os.environ.get("COHERE_API_KEY"):
    print("Warning: COHERE_API_KEY is not set in environment variables.")
    print("The reranker will not be used. Set it in your .env file.")

# Create a test state with some transaction entries
test_state = {
    "entries": [
        {"account": "Murabaha Receivables", "debit": 100000.00, "credit": 0.00},
        {"account": "Unearned Profit", "debit": 0.00, "credit": 20000.00},
        {"account": "Cash", "debit": 0.00, "credit": 80000.00}
    ],
    "context": "Islamic bank providing Murabaha financing for a customer to purchase inventory.",
    "accounting_treatment": "The bank purchased goods and sold them to the customer at cost plus markup, with deferred payment terms.",
    "valid": True
}

# Test with reranker enabled
print("\n===== Testing with reranker enabled =====")
retrieve_node_with_reranker = RetrieveNode(use_reranker=True)
result_with_reranker = retrieve_node_with_reranker(test_state)

# Test with reranker disabled
print("\n===== Testing with reranker disabled =====")
retrieve_node_without_reranker = RetrieveNode(use_reranker=False)
result_without_reranker = retrieve_node_without_reranker(test_state)

# Compare top results from both approaches
print("\n===== Results comparison =====")
print(f"Number of results with reranker: {len(result_with_reranker['retrieved_texts'])}")
print(f"Number of results without reranker: {len(result_without_reranker['retrieved_texts'])}")

print("\nTop sources with reranker:")
for i, source in enumerate(result_with_reranker['sources'][:3]):
    print(f"  {i+1}. {source} (score: {result_with_reranker['retrieval_scores'][i]:.4f})")

print("\nTop sources without reranker:")
for i, source in enumerate(result_without_reranker['sources'][:3]):
    print(f"  {i+1}. {source} (score: {result_without_reranker['retrieval_scores'][i]:.4f})")

print("\nDone!")
