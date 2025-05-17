# Reverse-Transaction Classifier for AAOIFI Standards

A Python package that classifies accounting transactions according to AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) Financial Accounting Standards (FAS).

## Overview

This package ingests a JSON payload containing transaction entries and returns a classification across five AAOIFI standards (FAS 4, 7, 10, 28, and 32), including probability scores, rationale, and source references.

The classification pipeline is implemented using LangGraph and leverages advanced AI (GPT-4o) to match transactions to the appropriate accounting standard.

## Features

- **Schema and balance validation** of transaction entries
- **Semantic search** over AAOIFI standards corpus using ChromaDB
- **Multi-LLM classification** with Chain-of-Thought prompting
- **Majority voting** and **logits calibration**
- **Verification** of rationales against sources
- **Detailed output** with probabilities and citations

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key  # Optional, for improved retrieval with reranking
```

Additional optional environment variables:
```
USE_RERANKER=true                  # Enable/disable Cohere's reranking (default: true)
RERANKER_MODEL=rerank-english-v3.0 # Cohere reranker model to use
```

## Usage

### Command-Line Interface

```bash
# Build the vector database first (only needed once)
python -m rtx_classifier.vectorstore

# Run the classifier
python -m rtx_classifier.run sample.json
```

Optional arguments for the classifier:
- `--output/-o`: Path to write JSON output (defaults to stdout)
- `--model/-m`: OpenAI model to use (defaults to gpt-4o)
- `--api-key/-k`: OpenAI API key (defaults to OPENAI_API_KEY env var)

### Python API

```python
from rtx_classifier.pipeline import run_classifier

# Input JSON with transaction entries
json_data = {
    "entries": [
        {"account": "Murabaha Receivables", "debit": 100000.00, "credit": 0.00},
        {"account": "Unearned Profit", "debit": 0.00, "credit": 20000.00},
        {"account": "Cash", "debit": 0.00, "credit": 80000.00}
    ],
    "context": "Islamic bank providing Murabaha financing for a customer to purchase inventory.",
    "accounting_treatment": "The bank purchased goods and sold them to the customer at cost plus markup, with deferred payment terms."
}

# Run the classifier
result = run_classifier(json_data)

# Output: {"prob_vector": [...], "rationale": "...", "sources": [...]}
print(result)
```

## Input Format

```json
{
  "entries": [
    {"account": "Account Name", "debit": 100.0, "credit": 0.0},
    {"account": "Another Account", "debit": 0.0, "credit": 100.0}
  ],
  "context": "Optional context about the transaction",
  "adjustments": "Optional information about adjustments",
  "accounting_treatment": "Optional accounting treatment details"
}
```

## Output Format

```json
{
  "prob_vector": [0.01, 0.02, 0.05, 0.90, 0.02],
  "rationale": "This transaction is classified under FAS 28 (Murabaha and Other Deferred Payment Sales) because...",
  "sources": ["FAS28 ¶3.1", "FAS28 ¶4.2"]
}
```

## Architecture

The classification pipeline consists of these LangGraph nodes:

1. **PreprocessNode** - Schema/balance/forbidden-account checks
2. **RetrieveNode** - Semantic search over AAOIFI standards corpus using local ChromaDB
3. **ClassifyFanOutNode** - Parallel LLM calls with Chain-of-Thought
4. **MajorityVoteNode** - Aggregates logits and selects provisional label
5. **CalibrateNode** - Temperature-scales logits to probabilities
6. **VerifyNode** - Verifies rationale cites only retrieved text
7. **ComposeNode** - Returns final JSON output

## Vector Database

The system builds a local ChromaDB vector database from the PDF documents in the `docs` folder, including:
- FAS 4: Foreign Currency Transactions and Foreign Operations
- FAS 7: Disclosure of Bases for Profit Allocation 
- FAS 10: Istisna'a and Parallel Istisna'a
- FAS 28: Murabaha and Other Deferred Payment Sales
- FAS 32: Financing by Qard
- Various Shari'ah Standards (SS8-SS12)

## Testing

Run the tests:

```bash
pytest tests/
```

## License

MIT