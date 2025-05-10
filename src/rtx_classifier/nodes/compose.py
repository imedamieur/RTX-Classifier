"""
ComposeNode for the RTX Classifier.

Composes the final output JSON with probabilities, rationale, and sources.
"""
from typing import Dict, List, Any, Optional


class ComposeNode:
    """
    ComposeNode that composes the final output for the RTX Classifier.
    
    This node prepares the final output, focusing on the probability vector.
    """
    
    def __init__(self):
        """Initialize the ComposeNode."""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and compose the final output.
        
        Args:
            state: Input state with probability vector and validity status.
            
        Returns:
            Final output state with prob_vector and validity.
        """
        new_state = dict(state)
        
        # print(f"ComposeNode input state: valid={new_state.get('valid', False)}, error_message={new_state.get('error_message')}")
        
        prob_vector_to_return = new_state.get("prob_vector", [0.0, 0.0, 0.0, 0.0, 0.0])
        is_valid = new_state.get("valid", False)
        error_msg_to_propagate = new_state.get('error_message')

        # Ensure prob_vector has 5 elements, providing a default if not, especially on error
        if not isinstance(prob_vector_to_return, list) or len(prob_vector_to_return) != 5:
            # print(f"Warning: prob_vector in ComposeNode is not a list of 5 elements. Received: {prob_vector_to_return}. Using default error state vector.")
            prob_vector_to_return = [0.0, 0.0, 0.0, 0.0, 0.0] # Default error state
            is_valid = False # If prob_vector is malformed, consider it an invalid state for output
            if not error_msg_to_propagate:
                error_msg_to_propagate = "Output probability vector malformed."

        final_output = {
            "prob_vector": prob_vector_to_return,
            "valid": is_valid
        }

        if not is_valid:
            final_output["error_message"] = error_msg_to_propagate or "Processing failed for an unspecified reason."
        
        # print(f"ComposeNode final output: {final_output}")
        
        # Remove fields that are no longer part of the primary output focus
        # These might still be in the state from previous nodes but are not part of the final desired output.
        keys_to_remove_from_final_state_dict = ["final_rationale", "sources", "correction_notes", 
                                                "final_explanation", "provisional_explanation", "citations",
                                                "retrieved_texts", "logits", "avg_logits", "provisional_label",
                                                "entries", "context", "adjustments", "accounting_treatment",
                                                "final_classification_label", "issue_note"]
        
        # The graph execution will return the full state dictionary from the last node.
        # We are modifying the state `new_state` here which will be returned by the graph.
        # The `run_classifier` function will then select which keys from this state to put in its own return dict.
        
        # For the state that ComposeNode itself returns (which becomes the graph's final state):
        # We want it to be clean, primarily containing what `run_classifier` will pick up.
        # However, LangGraph typically returns the full state of the last executed node.
        # So, the filtering should ideally happen in `run_classifier` when preparing its return value.
        # For now, just ensure the primary targeted outputs are correct in `new_state`.

        new_state["prob_vector"] = final_output["prob_vector"]
        new_state["valid"] = final_output["valid"]
        if "error_message" in final_output:
            new_state["error_message"] = final_output["error_message"]
        else:
            new_state.pop("error_message", None)

        # Clean up other keys from the state that are no longer relevant to the final output
        for key_to_remove in keys_to_remove_from_final_state_dict:
            new_state.pop(key_to_remove, None)
            
        return new_state