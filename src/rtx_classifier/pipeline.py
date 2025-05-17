"""
LangGraph pipeline for the Reverse-Transaction Classifier.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
import os
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
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
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
    avg_logits: List[float] = field(default_factory=list) # From MajorityVoteNode
    provisional_label: Optional[int] = None # Retained as it might be used by calibration/voting before verification
    provisional_rationale: Optional[str] = None
    provisional_label_name: Optional[str] = None # Added for clarity
    final_classification_label: Optional[str] = None # Added by VerifyNode
    retry_count: int = 0
    verification_explanation : Optional[str] = None # Added by VerifyNode
    final_rationale: Optional[str] = None # Added by VerifyNode
    alternative_standards: List[str] = field(default_factory=list) # Added by VerifyNode
    api_retrieve_response: Optional[dict] = None  # Added to pass API response from retrieve node
    
    # Final output focus
    prob_vector: List[float] = field(default_factory=list)
    
    # Removed fields:
    # sources: List[str] = field(default_factory=list)
    # correction_notes: Optional[str] = None
    # provisional_explanation: Optional[str] = None (was never formally here, but as a note)
    # final_explanation: Optional[str] = None (was never formally here, but as a note)
    
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
    Create and return an LLM instance based on the provider and model name.
    
    Args:
        model_provider: Provider of the model ('openai' or 'google')
        model_name: Name of the model to use
        openai_api_key: OpenAI API key
        google_api_key: Google API key
        temperature: Temperature parameter for the model
        
    Returns:
        A chat model instance that can be used with LangChain
    """
    # Use defaults if not provided
    model_provider = model_provider or DEFAULT_MODEL_PROVIDER
    model_name = model_name or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    if model_provider.lower() == "openai":
        # Handle OpenAI models
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
            temperature=0.4,
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
        temperature= temperature,
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


def run_classifier(
    json_data: Dict[str, Any],
    model_provider: str = None,
    model_name: str = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the RTX Classifier on the input JSON data.
    
    Args:
        json_data: Input JSON with entries and optional context fields
        model_provider: Provider of the model ('openai' or 'google')
        model_name: Name of the model to use
        openai_api_key: OpenAI API key
        google_api_key: Google API key
        
    Returns:
        Classification results with probabilities, rationale, and sources
    """
    # Load environment variables if not already loaded
    dotenv.load_dotenv()
    
    # Build the graph without using with_config
    graph = build_classifier_graph(
        model_provider=model_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
    )
    
    # Initialize state with ClassifierState object
    initial_state = ClassifierState(
        entries=json_data.get("entries", []),
        context=json_data.get("context"),
        adjustments=json_data.get("adjustments"),
        accounting_treatment=json_data.get("accounting_treatment"),
        valid=True,  # Start with valid=True
    )
    
    print(f"Starting classification with model provider: {model_provider or DEFAULT_MODEL_PROVIDER}")
    
    # Run the graph using invoke (preferred method for current LangGraph versions)
    try:
        # Use invoke method for current LangGraph versions
        result = graph.invoke(initial_state)
        print("Successfully executed graph with .invoke()")
    except AttributeError:
        # Fallback for very old versions that might use run
        try:
            print("Trying .run() method...")
            result = graph.run(initial_state)
            print("Successfully executed graph with .run()")
        except Exception as e2:
            print(f"Error running graph with .run(): {str(e2)}")
            try:
                # Try using the compiled graph approach
                print("Trying compiled graph approach...")
                compiled_graph = graph.compile()
                result = compiled_graph.invoke(initial_state)
                print("Successfully executed compiled graph")
            except Exception as e3:
                print(f"All execution attempts failed. Last error: {str(e3)}")
                return {
                    "prob_vector": [],
                    "explanation": f"Error executing graph: {str(e3)}",
                    "sources": []
                }
    except Exception as e:
        print(f"Error running graph with .invoke(): {str(e)}")
        try:
            # Try using the compiled graph approach
            print("Trying compiled graph approach...")
            compiled_graph = graph.compile()
            result = compiled_graph.invoke(initial_state)
            print("Successfully executed compiled graph")
        except Exception as e3:
            print(f"All execution attempts failed. Last error: {str(e3)}")
            return {
                "prob_vector": [],
                "explanation": f"Error executing graph: {str(e3)}",
                "sources": []
            }
    
    # Convert ClassifierState to dict if needed
    if not isinstance(result, dict):
        try:
            result = asdict(result)
        except Exception as e:
            print(f"Error converting result to dict: {str(e)}")
            # If conversion fails, create a minimal valid result
            return {
                "prob_vector": [],
                "explanation": f"Error processing result: {str(e)}",
                "sources": []
            }
    
    # Debug the final result
    # print(f"Final result keys: {list(result.keys())}")
    # print(f"Final result valid: {result.get('valid', False)}")
    
    # Prepare the final output based on the simplified requirements
    output_prob_vector = result.get("prob_vector", [])
    output_valid = result.get("valid", False)
    output_error_message = result.get("error_message")
    rationale = result.get("final_rationale", None)
    label = result.get("final_classification_label", None)
    alternative_standards = result.get("alternative_standards", None)

    if not isinstance(output_prob_vector, list) or len(output_prob_vector) != 5:
        output_prob_vector = [0.0, 0.0, 0.0, 0.0, 0.0] # Default error state
        output_valid = False
        if not output_error_message:
            output_error_message = "Output probability vector malformed or missing."

    final_output = {
        "prob_vector": output_prob_vector,
        "label": label,
        "valid": output_valid,
        "rationale": rationale,
        "alternative_standards": alternative_standards,
        "error_message": None,  # Initialize to None, will be set if invalid
    }
    if not output_valid:
        final_output["error_message"] = output_error_message or "Classification failed for an unspecified reason."
    
    return final_output


import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def execute(self, entries, **kwargs):
    """Execute the classifier pipeline on a list of entries."""
    logger.info(f"Executing classifier pipeline on {len(entries)} entries")
    
    # Initialize state
    state = ClassifierState(entries=entries, **kwargs)
    
    # Try multiple execution methods for robustness
    result = None
    methods_tried = []
    
    # Method 1: Try invoking the graph directly with state
    try:
        methods_tried.append("invoke")
        logger.debug("Attempting to execute graph with invoke() method")
        result = self.graph.invoke(state)
        logger.info("Graph executed successfully with invoke()")
        return result
    except Exception as e:
        logger.warning(f"Failed to execute graph with invoke(): {str(e)}")
    
    # Method 2: Try running the graph with state as dict
    try:
        methods_tried.append("run")
        logger.debug("Attempting to execute graph with run() method")
        result = self.graph.run(asdict(state))
        logger.info("Graph executed successfully with run()")
        return ClassifierState(**result) if isinstance(result, dict) else result
    except Exception as e:
        logger.warning(f"Failed to execute graph with run(): {str(e)}")
    
    # Method 3: Try compiled version of the graph
    try:
        methods_tried.append("compiled_graph")
        logger.debug("Attempting to execute graph with compiled graph")
        app = self.graph.compile()
        result = app.invoke(state)
        logger.info("Graph executed successfully with compiled graph")
        return result
    except Exception as e:
        logger.warning(f"Failed to execute graph with compiled graph: {str(e)}")
    
    # Method 4: Try with direct node execution as fallback
    try:
        methods_tried.append("manual_execution")
        logger.debug("Attempting manual execution of graph nodes")
        
        # Process the entries manually through each node
        state = self.node_preprocess(state)
        state = self.node_retrieve(state)
        state = self.node_classify(state)
        state = self.node_calibrate(state)
        state = self.node_verify(state)
        state = self.node_voting(state)
        state = self.node_compose(state)
        
        logger.info("Graph executed successfully with manual node execution")
        return state
    except Exception as e:
        logger.warning(f"Failed to execute graph with manual execution: {str(e)}")
    
    # If all methods failed, raise an error
    error_msg = f"Failed to execute graph with methods: {', '.join(methods_tried)}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)