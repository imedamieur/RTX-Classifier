# RTX Classifier: Reverse-Transaction Classifier for AAOIFI Standards

A Python-based machine learning system that classifies accounting transactions according to AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) Financial Accounting Standards (FAS).

## Overview

The RTX Classifier analyzes transaction entries and contextual information to determine the most applicable AAOIFI standard among FAS 4, 7, 10, 28, and 32. It provides classification results with rationale, probability scores, and source references from the standards.

Built with a modular LangGraph architecture, the system combines semantic search, large language models, and verification steps to ensure accurate and reliable classifications for Islamic financial transactions.

## Key Features

- **Multi-Standard Classification**: Identifies the most relevant AAOIFI standard with probability scores
- **Interactive Web Interface**: User-friendly Streamlit app for both regular and debug modes
- **Step-by-Step Debugging**: Visualize each stage of the classification pipeline
- **Raw API Results Display**: View underlying API responses in debug mode
- **Model Provider Options**: Support for OpenAI and Google generative AI models
- **Alternative Standards**: Suggests other potentially applicable standards with reasoning

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key  # Optional, for Google models
COHERE_API_KEY=your_cohere_api_key  # Optional, for improved retrieval with reranking

# Optional configuration
USE_RERANKER=true                   # Enable/disable Cohere's reranking (default: true)
RERANKER_MODEL=rerank-english-v3.0  # Cohere reranker model to use
DEFAULT_MODEL=gpt-4o                # Default model to use
DEFAULT_MODEL_PROVIDER=openai       # Default provider (openai or google)
TEMPERATURE=0.7                     # Default temperature for classification
FALLBACK_TEMPERATURE=0.3            # Temperature for retry attempts
CALIBRATION_TEMP=0.7                # Temperature scaling for calibration
TOP_K_RETRIEVAL=10                  # Number of documents to retrieve
```

## Usage

### Web Interface

Run the Streamlit app:

```bash
# Standard interface
streamlit run src/rtx_classifier/app.py

# Debug interface with step-by-step pipeline visualization
streamlit run src/rtx_classifier/app_debug.py
```

### Command-Line Interface

```bash
# Build the vector database (only needed once)
python -m rtx_classifier.vectorstore

# Run the classifier on a sample JSON file
python -m rtx_classifier.run sample.json

# Output to a file
python -m rtx_classifier.run sample.json --output result.json

# Specify model and provider
python -m rtx_classifier.run sample.json --model-provider google --model-name gemini-2.0-flash
```

### Python API

```python
from rtx_classifier.pipeline import run_classifier

# Input JSON with transaction entries and context
json_data = {
    "entries": [
        {"account": "Murabaha Receivables", "debit": 100000.00, "credit": 0.00},
        {"account": "Unearned Profit", "debit": 0.00, "credit": 20000.00},
        {"account": "Cash", "debit": 0.00, "credit": 80000.00}
    ],
    "context": "Islamic bank providing Murabaha financing for a customer to purchase inventory.",
    "adjustments": "The profit rate is 5% per annum for 2 years.",
    "accounting_treatment": "The bank purchased goods and sold them to the customer at cost plus markup, with deferred payment terms."
}

# Run the classifier
result = run_classifier(json_data)
print(result)

# Using a specific model
result = run_classifier(
    json_data, 
    model_provider="openai", 
    model_name="gpt-4o"
)

# Debug mode to see intermediate steps
from rtx_classifier.pipeline_debug import run_classifier_debug
debug_results = run_classifier_debug(json_data)
```

## Input Format

The system accepts JSON input with the following structure:

```json
{
  "entries": [
    {"account": "Account Name", "debit": 100.0, "credit": 0.0},
    {"account": "Another Account", "debit": 0.0, "credit": 100.0}
  ],
  "context": "Context about the transaction",
  "adjustments": "Information about adjustments",
  "accounting_treatment": "Description of accounting treatment"
}
```

## Output Format

The classifier returns a JSON object with these fields:

```json
{
  "label": "FAS28",
  "prob_vector": [0.01, 0.02, 0.05, 0.90, 0.02],
  "rationale": "This transaction is classified under FAS 28 (Murabaha and Other Deferred Payment Sales) because...",
  "sources": ["FAS28 ¶3.1", "FAS28 ¶4.2"],
  "alternative_standards": [
    {"standard": "FAS10", "reason": "Could be considered if..."},
    {"standard": "FAS32", "reason": "Might apply if..."}
  ]
}
```

The `prob_vector` corresponds to probabilities for [FAS4, FAS7, FAS10, FAS28, FAS32].

## Classification Pipeline Architecture

The RTX Classifier uses a modular LangGraph pipeline with these nodes:

1. **Preprocess**: Validates input data (schema, balance checks, forbidden accounts)
2. **Retrieve**: Performs semantic search over AAOIFI standards using ChromaDB
3. **Classify**: Executes multiple parallel LLM calls to generate classifications
4. **Majority Vote**: Aggregates results from classification runs
5. **Calibrate**: Applies temperature scaling to logits for final probabilities
6. **Verify**: Ensures the classification reasoning is valid and references retrieved content
7. **Compose**: Constructs the final output with probabilities, rationale and sources

## Supported Standards

The system classifies transactions across these AAOIFI Financial Accounting Standards:

- **FAS 4**: Foreign Currency Transactions and Foreign Operations
- **FAS 7**: Disclosure of Bases for Profit Allocation
- **FAS 10**: Istisna'a and Parallel Istisna'a
- **FAS 28**: Murabaha and Other Deferred Payment Sales
- **FAS 32**: Financing by Qard

Additional Shari'ah Standards (SS8-SS12) are included in the knowledge base to provide context.

## Testing

Run the test suite:

```bash
pytest tests/
```

## Debug Mode Features

The debug version (app_debug.py) provides:

- Step-by-step visualization of each pipeline node
- Raw API retrieval results from the semantic search
- Detailed logits and probability scores at each stage
- Alternative standards considered during classification
- Access to the raw state data for each node

## License

MIT
