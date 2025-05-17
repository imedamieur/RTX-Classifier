"""
ClassifyFanOutNode for the RTX Classifier.

Runs parallel LLM calls with few-shot Chain-of-Thought for classification.
Supports both OpenAI and Google Gemini models.
"""
from typing import Dict, List, Any, Optional, Callable
import json
import os
from pathlib import Path
import random
import dotenv

# Load environment variables
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import  OutputParserException # Changed import

# Standard labels for classification
LABELS = ["FAS4", "FAS7", "FAS10", "FAS28", "FAS32"]

# Full names for standard citations
STANDARD_FULL_NAMES = {
    "FAS4": "FAS 4: Musharaka financing",
    "FAS7": "FAS 7: Salam and Parallel Salam",
    "FAS10": "FAS 10: Istisna'a and Parallel Istisna'a",
    "FAS28": "FAS 28: Murabaha and Other Deferred Payment Sales",
    "FAS32": "FAS 32: Ijara and Ijara Muntahia Bittamleek",
}

# Get configuration from environment variables
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
FALLBACK_TEMPERATURE = float(os.getenv("FALLBACK_TEMPERATURE", "0.3"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
GOOGLE_DEFAULT_MODEL = os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-pro") # Added for Google specific model


class ClassifyFanOutNode:
    """
    ClassifyFanOutNode that runs multiple parallel LLM calls for classification.
    Uses few-shot Chain-of-Thought prompting with JSON mode enabled.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        num_parallel: int = 3,
        template_path: Optional[str] = None,
        output_parser: Optional[Callable] = None,
    ):
        """
        Initialize the ClassifyFanOutNode.
        
        Args:
            llm: LangChain chat model instance
            num_parallel: Number of parallel LLM calls to make
            template_path: Path to the Jinja template for prompting
            output_parser: Custom output parser function
        """
        self.llm = llm
        self.num_parallel = num_parallel
        self.template_path = template_path
        self.output_parser = output_parser or JsonOutputParser()
        
        # Load prompt template if path is provided
        self._prompt_template = None
        if template_path:
            self._load_template(template_path)
    
    def _load_template(self, template_path: str):
        """Load template from the specified path."""
        try:
            template_dir = Path(__file__).parent.parent / "templates"
            # Handle both cases: full filename or just the base name
            if os.path.isabs(template_path):
                template_file = Path(template_path)
            else:
                template_file = template_dir / template_path
            
            # Add .jinja extension if not present
            if not template_file.exists() and not str(template_path).endswith(".jinja"):
                template_file = template_dir / f"{template_path}.jinja"
            
            if template_file.exists():
                template_content = template_file.read_text(encoding='utf-8')
                
                # Create proper template with jinja2 format
                try:
                    self._prompt_template = PromptTemplate.from_template(
                        template_content,
                        template_format="jinja2"
                    )
                    return
                except Exception as template_error:
                    print(f"Error creating template from file {template_file}: {template_error}")
            else:
                print(f"Template file not found: {template_file}")
        except Exception as e:
            print(f"Failed to load template {template_path}: {e}")
        
        # Fall back to default template if loading fails or file doesn't exist
        print("Using default template instead")
        self._prompt_template = self._get_default_template()

    def _get_default_template(self) -> PromptTemplate: # Corrected return type hint
        """Get default template for classification."""
        template = """
        You are a financial auditor specializing in AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) standards.

        Analyze the following accounting transaction and determine which AAOIFI FAS standard applies best.

        Transaction entries:
        {% for entry in entries %}
        {% if entry.debit %}Dr. {{ entry.account }}: {{ entry.debit }}{% endif %}
        {% if entry.credit %}Cr. {{ entry.account }}: {{ entry.credit }}{% endif %}
        {% endfor %}

        {% if context %}
        Context: {{ context }}
        {% endif %}

        {% if adjustments %}
        Adjustments: {{ adjustments }}
        {% endif %}

        {% if accounting_treatment %}
        Accounting treatment: {{ accounting_treatment }}
        {% endif %}

        The transaction should be classified according to one of these AAOIFI standards:
        - FAS 4: Musharaka financing
        - FAS 7: Salam and Parallel Salam
        - FAS 10: Istisna'a and Parallel Istisna'a
        - FAS 28: Murabaha and Other Deferred Payment Sales
        - FAS 32: Ijara and Ijara Muntahia Bittamleek

        {% if excluded_standards is defined and excluded_standards %}
        IMPORTANT: The following standards are to be EXCLUDED. Their corresponding confidence scores (logits) in the output MUST be extremely low (near zero):
        {% for standard in excluded_standards %}
        - {{ standard }} (This refers to {{ STANDARD_FULL_NAMES.get(standard, 'Unknown Standard') }})
        {% endfor %}
        Remember, the logits array corresponds to [FAS4, FAS7, FAS10, FAS28, FAS32].
        {% endif %}

        {% if api_query %}
        SEARCH QUERY USED: "{{ api_query }}"
        {% endif %}

        RELEVANCE SCORE EXPLANATION: The relevance scores below indicate how closely each chunk of text matches the transaction details. Scores range from 0 to 1, where higher scores (closer to 1) indicate stronger relevance to the transaction. Give more weight to chunks with higher relevance scores when determining classification.

        Based on the RAG (BM25 + VECTOR STORE) retrieved sources (sorted by relevance):
        {% for chunk in api_chunks|sort(attribute='score', reverse=true) %}
        --- Source from {{ chunk.document_name }} (Relevance Score: {{ chunk.score|round(3) }}) ---
        {{ chunk.chunk_text }}
        {% endfor %}

        Think through this step by step:
        1. Identify the accounts involved and their nature
        2. Determine the transaction type based on account names and flow
        3. Match the transaction characteristics to AAOIFI standards
        4. Prioritize evidence from chunks with higher relevance scores (>0.7 is highly relevant)
        5. If multiple standards could apply, use the highest relevance score chunks to make your final decision

        Finally, provide your classification and reasoning in this JSON format:
        {
        "logits": [float, float, float, float, float],  // Confidence scores for each standard in order: FAS4, FAS7, FAS10, FAS28, FAS32
        "rationale": "Your detailed reasoning for the classification. This reasoning MUST clearly state the determined AAOIFI standard."
        }

        Your logits should sum to 1.0 representing a probability distribution.
        """
        return PromptTemplate.from_template(
            template,
            template_format="jinja2",
            partial_variables={"STANDARD_FULL_NAMES": STANDARD_FULL_NAMES}
            # Removed explicit input_variables to let them be inferred by from_template
            # input_variables=["entries", "context", "adjustments", "accounting_treatment", "api_chunks"]
        )

    def _get_prompt_template(self):
        """Get the prompt template, loading default if needed."""
        if not self._prompt_template:
            if self.template_path:
                self._load_template(self.template_path)
            else:
                self._prompt_template = self._get_default_template()
        return self._prompt_template
    
    def _run_single_classification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single LLM classification."""
        if not self.llm:
            raise ValueError("LLM is not initialized")
        
        # Get the prompt template
        prompt = self._get_prompt_template()
        
        # Process api_retrieve data
        api_chunks = []
        api_query = ""
        api_retrieve = state.get("api_retrieve_response", None)
        if api_retrieve and isinstance(api_retrieve, dict):
            # Extract the FAS chunks and query from the API retrieve response
            api_chunks = api_retrieve.get("fas_chunks", [])
            api_query = api_retrieve.get("query", "")
            
            # Sort chunks by relevance score (highest first)
            if api_chunks:
                api_chunks = sorted(api_chunks, key=lambda x: float(x.get("score", 0)), reverse=True)
            

        
        # If no API chunks are available, set empty list to ensure proper handling
        if not api_chunks:
            api_chunks = []
        
        # Prepare variables for the template
        inputs = {
            "entries": state.get("entries", []),
            "context": state.get("context"),
            "adjustments": state.get("adjustments"),
            "accounting_treatment": state.get("accounting_treatment"),
            "excluded_standards": state.get("excluded_standards", []), # Added for exclusions
            "api_chunks": api_chunks,  # Use API retrieved chunks exclusively
            "api_query": api_query,  # Add the query that was used for retrieval
        }
        
        # Create and run the chain
        chain = prompt | self.llm | self.output_parser
        
        # Call the LLM and parse the output
        try:
            result = chain.invoke(inputs) # This can raise OutputParsingError
            
            # If chain.invoke() was successful, JsonOutputParser ensures 'result' is a dict.
            logits = result.get("logits", [])
            rationale = result.get("rationale", "")

            
            
            # Check logits
            if not isinstance(logits, list) or len(logits) != 5:
                raise ValueError(f"Expected 5 logits, got: {logits}")
            
            # Normalize logits if they don't sum to 1.0
            logits_sum = sum(logits)
            if abs(logits_sum - 1.0) > 0.01: # Using 0.01 tolerance
                if logits_sum == 0: # Avoid division by zero
                    # If sum is 0, cannot normalize, treat as an error or assign uniform distribution.
                    # For now, let's keep it as an issue that might lead to "N/A" classification.
                    print("Warning: Sum of logits is 0, cannot normalize. Classification might be N/A.")
                else:
                    logits = [v / logits_sum for v in logits]

            classification = "N/A"
            standard_citation = "N/A"
            if logits and len(logits) == len(LABELS): # Ensure logits is not empty and matches LABELS length
                try:
                    max_logit_index = logits.index(max(logits))
                    classification = LABELS[max_logit_index]
                    standard_citation = STANDARD_FULL_NAMES.get(classification, "Unknown standard name")
                except ValueError: # max() on empty sequence or other value errors
                    print(f"Error determining classification from logits: {logits}")
            else:
                print(f"Logits list is empty or length mismatch: {logits}")
                
            return {
                "logits": logits,
                "rationale": rationale,
                "classification": classification,
                "standard_citation": standard_citation
            }
            
        except  OutputParserException as ope:
            # Handle cases where the LLM output was not valid JSON
            error_message = str(ope)
            llm_output_snippet = "Not available"
            if hasattr(ope, 'llm_output') and ope.llm_output:
                llm_output_snippet = str(ope.llm_output)[:200] # Get first 200 chars

            print(f"Classification OutputParsingError: {error_message}. Raw LLM output snippet: '{llm_output_snippet}'")
            return {
                "logits": [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal probabilities
                "rationale": "Failed to parse LLM output. The response was not in the expected JSON format.",
                "classification": "Error",
                "standard_citation": "Error parsing LLM output"
            }
            
        except Exception as e: # Catches ValueErrors from logit checks, or other unexpected errors
            # Return default values on error
            print(f"Classification error: {e}")
            return {
                "logits": [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal probabilities
                "rationale": f"Failed to classify: {str(e)}", # Original rationale format for other errors
                "classification": "Error",
                "standard_citation": f"Error during classification: {str(e)}"
            }
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and run classification.
        
        Args:
            state: Input state with transaction entries and retrieved texts
            
        Returns:
            Updated state with classification results
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)

        api_retrieve = new_state.get("api_retrieve_response", None)
        
        # Check if API retrieve contains valid data and add relevant information to the state
        if api_retrieve and isinstance(api_retrieve, dict):
            query = api_retrieve.get("query", "")
            if query:
                new_state["api_query"] = query
        
        # Check if previous steps failed
        if not new_state.get("valid", False):
            return new_state
        
        # Apply temporary temperature if it's a retry
        temp_param = {}
        if new_state.get("retry_count", 0) > 0:
            temp_param = {"temperature": float(os.getenv("FALLBACK_TEMPERATURE", FALLBACK_TEMPERATURE))}
        
        # Make a copy of the LLM with potentially modified parameters
        current_llm = None
        if self.llm:
            if isinstance(self.llm, ChatOpenAI):
                # Handle OpenAI LLM
                llm_params = self.llm.model_dump()
                llm_params.update(temp_param)
                current_llm = ChatOpenAI(**llm_params)
            elif isinstance(self.llm, ChatGoogleGenerativeAI):
                # Handle Google Gemini LLM
                llm_params = self.llm.model_dump()
                if temp_param:
                    llm_params["temperature"] = temp_param["temperature"]
                current_llm = ChatGoogleGenerativeAI(**llm_params)
            else:
                # Default fallback
                current_llm = self.llm
        else:
            # Create a default LLM if none was provided
            model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").lower()
            if model_provider == "openai":
                current_llm = ChatOpenAI(
                    model_name=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE if not temp_param else temp_param["temperature"],
                )
            elif model_provider == "google":
                current_llm = ChatGoogleGenerativeAI(
                    model=GOOGLE_DEFAULT_MODEL,  # Use GOOGLE_DEFAULT_MODEL
                    temperature=DEFAULT_TEMPERATURE if not temp_param else temp_param["temperature"],
                )
        
        # Create a temporary node with the current LLM
        temp_node = ClassifyFanOutNode(
            llm=current_llm,
            num_parallel=self.num_parallel,
            template_path=self.template_path,
            output_parser=self.output_parser
        )
        
        # Run parallel classifications
        logits_list = []
        rationales = []
        classifications = [] # Added
        standard_citations = [] # Added
        
        for _ in range(self.num_parallel):
            result = temp_node._run_single_classification(new_state)
            logits_list.append(result["logits"])
            rationales.append(result["rationale"])
            classifications.append(result["classification"]) # Added
            standard_citations.append(result["standard_citation"]) # Added
        
        # Store results in state
        new_state["logits"] = logits_list
        new_state["rationales"] = rationales
        new_state["classifications"] = classifications # Added
        new_state["standard_citations"] = standard_citations # Added
        
        return new_state