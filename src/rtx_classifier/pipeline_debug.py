"""
Debug version of the LangGraph pipeline for the Reverse-Transaction Classifier.
Exposes intermediate results from each node.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
import os
import copy
import dotenv

# Load environment variables
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
import numpy as np

from rtx_classifier.nodes.preprocess import PreprocessNode
from rtx_classifier.nodes.retrieve import RetrieveNode
from rtx_classifier.nodes.classify import ClassifyFanOutNode
from rtx_classifier.nodes.voting import MajorityVoteNode
from rtx_classifier.nodes.calibrate import CalibrateNode
from rtx_classifier.nodes.verify import VerifyNode
from rtx_classifier.nodes.compose import ComposeNode

# Get configuration from environment variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
FALLBACK_TEMPERATURE = float(os.getenv("FALLBACK_TEMPERATURE", "0.3"))
CALIBRATION_TEMP = float(os.getenv("CALIBRATION_TEMP", "0.7"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))


@dataclass
class ClassifierState:
    """State for the Reverse-Transaction Classifier pipeline."""
    # Input data
    entries: List[Dict[str, Any]]
    context: Optional[str] = None
    adjustments: Optional[str] = None
    accounting_treatment: Optional[str] = None
    
    # Processing state
    valid: bool = False
    error_message: Optional[str] = None
    retrieved_texts: List[str] = field(default_factory=list)
    logits: List[List[float]] = field(default_factory=list) # From ClassifyFanOutNode
    rationales: List[str] = field(default_factory=list) # From ClassifyFanOutNode
    classifications: List[str] = field(default_factory=list) # From ClassifyFanOutNode
    standard_citations: List[str] = field(default_factory=list) # From ClassifyFanOutNode
    avg_logits: List[float] = field(default_factory=list) # From MajorityVoteNode
    provisional_label: Optional[int] = None # Retained as it might be used by calibration/voting before verification
    provisional_label_name: Optional[str] = None # From MajorityVoteNode
    provisional_rationale: Optional[str] = None # From MajorityVoteNode
    provisional_standard_citation: Optional[str] = None # From MajorityVoteNode
    final_classification_label: Optional[str] = None # Added by VerifyNode
    retry_count: int = 0
    
    # Final output focus
    prob_vector: List[float] = field(default_factory=list)
    label: Optional[str] = None  # Added for easier access to final label
    rationale: Optional[str] = None  # Added for easier access to final rationale
    alternative_standards: List[Dict[str, str]] = field(default_factory=list) # Added for potential alternative standards
    
    # Make the dataclass fully compatible with newer LangGraph versions
    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Make the state iterable for compatibility with LangGraph."""
        return iter(asdict(self).items())
    
    def __getitem__(self, key):
        """Implement __getitem__ for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    
    def get(self, key, default=None):
        """Implement get method for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default
        
    def keys(self):
        """Implement keys method for dict-like access."""
        return asdict(self).keys()
        
    def items(self):
        """Implement items method for dict-like access."""
        return asdict(self).items()
        
    def __setitem__(self, key, value):
        """Implement __setitem__ for dict-like update access."""
        setattr(self, key, value)
        
    def values(self):
        """Implement values method for dict-like access."""
        return asdict(self).values()
        
    def update(self, other=None, **kwargs):
        """Implement update method for compatibility with graph transitions."""
        if other is not None:
            for key, value in other.items() if hasattr(other, 'items') else other:
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
        
    def copy(self):
        """Implement copy method for graph transitions."""
        return self.__class__(**asdict(self))


def get_llm(
    model_provider: str = None,
    model_name: str = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    temperature: float = None,
) -> BaseChatModel:
    """
    Get an LLM instance of the requested type.
    
    Args:
        model_provider: Provider of the model ('openai' or 'google')
        model_name: Name of the model to use
        openai_api_key: OpenAI API key
        google_api_key: Google API key
        temperature: Temperature for sampling
        
    Returns:
        A BaseChatModel instance
    """
    # Use environment variables if not provided
    model_provider = model_provider or DEFAULT_MODEL_PROVIDER
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    if model_provider.lower() == "openai":
        # Handle OpenAI models
        if not model_name:
            model_name = DEFAULT_MODEL
        
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI models")
        
        print(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            api_key=openai_api_key,
            model_name=model_name,
            temperature=temperature,
        )
    
    elif model_provider.lower() == "google":
        # Handle Google Gemini models
        if not google_api_key:
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("Google API key is required for Gemini models")
        
        # Use gemini-2.0-flash as requested by the user for Google provider
        effective_model_name = "gemini-2.0-flash"
        print(f"Using Google model: {effective_model_name} (overrides provided model_name if any for Google provider)")
        
        return ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=effective_model_name,
            temperature=temperature,
        )
    
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


def build_classifier_graph(
    model_provider: str = None,
    model_name: str = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    temperature: float = None,
    fallback_temperature: float = None,
    calibration_temp: float = None,
) -> StateGraph:
    """
    Build the RTX Classifier LangGraph pipeline.
    
    Args:
        model_provider: Provider of the model ('openai' or 'google')
        model_name: Name of the model to use for classification
        openai_api_key: OpenAI API key
        google_api_key: Google API key
        temperature: Primary temperature for classification
        fallback_temperature: Temperature for retry attempts
        calibration_temp: Temperature scaling factor for calibration
        
    Returns:
        A LangGraph StateGraph for the RTX Classifier
    """
    # Use environment variables if not provided
    model_provider = model_provider or DEFAULT_MODEL_PROVIDER
    
    if fallback_temperature is None:
        fallback_temperature = FALLBACK_TEMPERATURE
        
    if calibration_temp is None:
        calibration_temp = CALIBRATION_TEMP
    
    # Initialize LLM
    llm = get_llm(
        model_provider=model_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        temperature=temperature,
    )
    
    # Initialize nodes
    preprocess_node = PreprocessNode()
    retrieve_node = RetrieveNode(
        top_k=TOP_K_RETRIEVAL,
        persist_directory=True,  # Use environment variable for persistence directory
    )
    classify_node = ClassifyFanOutNode(
        llm=llm,
        num_parallel=3,
    )
    majority_vote_node = MajorityVoteNode()
    calibrate_node = CalibrateNode(temperature=calibration_temp)
    verify_node = VerifyNode(llm=llm)
    compose_node = ComposeNode()
    
    # Build the graph
    workflow = StateGraph(ClassifierState)
    
    # Add nodes
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("majority_vote", majority_vote_node)
    workflow.add_node("calibrate", calibrate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("compose", compose_node)
    
    # Define edges
    workflow.add_edge("preprocess", "retrieve")
    workflow.add_edge("retrieve", "classify")
    workflow.add_edge("classify", "majority_vote")
    workflow.add_edge("majority_vote", "calibrate")
    workflow.add_edge("calibrate", "verify")
    
    # Define conditional edges for verification results
    workflow.add_conditional_edges(
        "verify",
        lambda state: {
            # Route based on verification status and retry count
            # If already retried too many times, force completion to avoid infinite loops
            "compose": state["retry_count"] >= 3 or state["valid"],
            "classify": state["retry_count"] < 3 and not state["valid"]
        }
    )
    
    workflow.add_edge("compose", END)
    
    # Set entry point
    workflow.set_entry_point("preprocess")
    
    return workflow


def run_classifier_debug(
    json_data: Dict[str, Any],
    model_provider: str = None,
    model_name: str = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run the RTX Classifier on the input JSON data and capture intermediate results
    from each node in the LangGraph pipeline.
    
    Args:
        json_data: Input JSON with entries and optional context fields
        model_provider: Provider of the model ('openai' or 'google')
        model_name: Name of the model to use
        openai_api_key: OpenAI API key
        google_api_key: Google API key
        
    Returns:
        List of intermediate states from each node in the pipeline
    """
    # Load environment variables if not already loaded
    dotenv.load_dotenv()
    
    # Initialize nodes directly
    preprocess_node = PreprocessNode()
    retrieve_node = RetrieveNode(
        top_k=TOP_K_RETRIEVAL,
        persist_directory=True,
    )
    
    # Initialize LLM
    llm = get_llm(
        model_provider=model_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
    )
    
    classify_node = ClassifyFanOutNode(
        llm=llm,
        num_parallel=3,
    )
    majority_vote_node = MajorityVoteNode()
    calibrate_node = CalibrateNode(temperature=CALIBRATION_TEMP)
    verify_node = VerifyNode(llm=llm)
    compose_node = ComposeNode()
    
    # Initialize state with ClassifierState object
    initial_state = ClassifierState(
        entries=json_data.get("entries", []),
        context=json_data.get("context"),
        adjustments=json_data.get("adjustments"),
        accounting_treatment=json_data.get("accounting_treatment"),
        valid=True,  # Start with valid=True
    )
    
    # List to store intermediate states
    intermediate_results = []
      # Step 1: Preprocess
    print("Running Preprocess Node")
    preprocess_result = preprocess_node(initial_state)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(preprocess_result, '__dataclass_fields__'):
        preprocess_state = asdict(preprocess_result.copy())
    else:
        preprocess_state = dict(preprocess_result)
        preprocess_state = preprocess_result.copy() if hasattr(preprocess_result, 'copy') else preprocess_state.copy()
    preprocess_state['node'] = 'preprocess'
    intermediate_results.append(preprocess_state)
    
    if not preprocess_result.get("valid", False):
        print("Preprocessing failed. Returning early.")
        return intermediate_results
    
    # Step 2: Retrieve
    print("Running Retrieve Node")
    retrieve_result = retrieve_node(preprocess_result)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(retrieve_result, '__dataclass_fields__'):
        retrieve_state = asdict(retrieve_result.copy())
    else:
        retrieve_state = dict(retrieve_result)
        retrieve_state = retrieve_result.copy() if hasattr(retrieve_result, 'copy') else retrieve_state.copy()
    retrieve_state['node'] = 'retrieve'
    intermediate_results.append(retrieve_state)
      # Step 3: Classify
    print("Running Classify Node")
    classify_result = classify_node(retrieve_result)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(classify_result, '__dataclass_fields__'):
        classify_state = asdict(classify_result.copy())
    else:
        classify_state = dict(classify_result)
        classify_state = classify_result.copy() if hasattr(classify_result, 'copy') else classify_state.copy()
    classify_state['node'] = 'classify'
    intermediate_results.append(classify_state)
    
    # Step 4: Majority Vote
    print("Running Majority Vote Node")
    vote_result = majority_vote_node(classify_result)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(vote_result, '__dataclass_fields__'):
        vote_state = asdict(vote_result.copy())
    else:
        vote_state = dict(vote_result)
        vote_state = vote_result.copy() if hasattr(vote_result, 'copy') else vote_state.copy()
    vote_state['node'] = 'majority_vote'
    intermediate_results.append(vote_state)
      # Step 5: Calibrate
    print("Running Calibrate Node")
    calibrate_result = calibrate_node(vote_result)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(calibrate_result, '__dataclass_fields__'):
        calibrate_state = asdict(calibrate_result.copy())
    else:
        calibrate_state = dict(calibrate_result)
        calibrate_state = calibrate_result.copy() if hasattr(calibrate_result, 'copy') else calibrate_state.copy()
    calibrate_state['node'] = 'calibrate'
    intermediate_results.append(calibrate_state)
    
    # Step 6: Verify
    print("Running Verify Node")
    verify_result = verify_node(calibrate_result)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(verify_result, '__dataclass_fields__'):
        verify_state = asdict(verify_result.copy())
    else:
        verify_state = dict(verify_result)
        verify_state = verify_result.copy() if hasattr(verify_result, 'copy') else verify_state.copy()
    verify_state['node'] = 'verify'
    intermediate_results.append(verify_state)
    
    # Step 7: Compose (final output)
    print("Running Compose Node")
    compose_result = compose_node(verify_result)
    # Handle both dictionaries and dataclass instances properly
    if hasattr(compose_result, '__dataclass_fields__'):
        compose_state = asdict(compose_result.copy())
    else:
        compose_state = dict(compose_result)
        compose_state = compose_result.copy() if hasattr(compose_result, 'copy') else compose_state.copy()
    compose_state['node'] = 'compose'
    intermediate_results.append(compose_state)
      # Add final label and rationale for easier access
    if 'final_classification_label' in compose_state and compose_state['final_classification_label']:
        compose_state['label'] = compose_state['final_classification_label']
    
    if 'provisional_rationale' in compose_state and compose_state['provisional_rationale']:
        compose_state['rationale'] = compose_state['provisional_rationale']
    
    return intermediate_results
