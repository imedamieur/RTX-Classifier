"""
ComposeNode for the RTX Classifier.

Composes the final output JSON with probabilities, rationale, and sources.
"""
from typing import Dict, List, Any, Optional


class ComposeNode:
    """
    ComposeNode that composes the final output for the RTX Classifier.
    
    This node prepares the final output with:
    - Probability vector
    - Final rationale
    - Sources (retrieved paragraphs)
    """
    
    def __init__(self):
        """Initialize the ComposeNode."""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and compose the final output.
        
        Args:
            state: Input state with probability vector, rationale, and sources
            
        Returns:
            Final output state with required fields
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)
        
        # Debug: Print key state values
        print(f"ComposeNode state: valid={new_state.get('valid', False)}, retry_count={new_state.get('retry_count', 0)}")
        
        # Check if we have valid data
        if not new_state.get("valid", False):
            # If processing failed, pass through relevant error information from the state
            error_msg_to_propagate = new_state.get('error_message', 'Unknown error during processing.')
            # final_rationale from the input state might be a detailed failure message from VerifyNode
            final_rationale_to_propagate = new_state.get('final_rationale')
            sources_to_propagate = new_state.get('sources', []) # VerifyNode might set sources even on failure

            current_prob_vector = new_state.get("prob_vector")
            if isinstance(current_prob_vector, list) and len(current_prob_vector) == 5:
                prob_vector_to_return = current_prob_vector
            else:
                prob_vector_to_return = [0.0, 0.0, 0.0, 0.0, 0.0] # Default error state
            
            return {
                "prob_vector": prob_vector_to_return,
                "final_rationale": final_rationale_to_propagate, # Pass through from input state
                "sources": sources_to_propagate,                 # Pass through from input state
                "valid": False,
                "error_message": error_msg_to_propagate          # Pass through from input state, serves as fallback
            }
        
        # Extract the required fields
        prob_vector = new_state.get("prob_vector", [])
        final_rationale = new_state.get("final_rationale", "") # This is the rationale from verify/earlier successful step
        sources = new_state.get("sources", [])
        
        # Ensure prob_vector has 5 elements, providing a default if not
        if not isinstance(prob_vector, list) or len(prob_vector) != 5:
            # Log this situation, as it might indicate an upstream issue if 'valid' is True.
            print(f"Warning: prob_vector is not a list of 5 elements. Received: {prob_vector}. Using default.")
            prob_vector = [0.2, 0.2, 0.2, 0.2, 0.2] # Placeholder default
        
        # Construct a more informative rationale if parts are missing
        composed_rationale = final_rationale
        correction_notes = new_state.get("correction_notes")

        if not isinstance(final_rationale, str) or not final_rationale.strip():
            if not sources: # No rationale and no sources
                composed_rationale = ("Classification process completed, but a detailed rationale could not be generated, "
                                      "and no supporting sources were identified. "
                                      "Please review the input or try a more specific query.")
            else: # No rationale but sources exist
                composed_rationale = ("Classification process completed, but a detailed rationale could not be generated. "
                                      "Supporting sources are provided below.")
        elif not sources: # Rationale exists, but no sources
            composed_rationale = (f"{final_rationale.strip()} "
                                  "(Note: No specific supporting sources were identified for this rationale.)")
        
        # Append correction notes if they exist and the rationale is not a generic error message
        if correction_notes and composed_rationale and "Classification process completed, but" not in composed_rationale:
            composed_rationale = f"{composed_rationale.strip()} [Verifier notes: {correction_notes}]"
        
        # Debug: Print output values
        print(f"Output: prob_vector={prob_vector}, rationale_len={len(composed_rationale)}, sources_count={len(sources)}")
        
        # Return the final output
        return {
            "prob_vector": prob_vector,
            "final_rationale": composed_rationale.strip(), # Use correct field name
            "sources": sources,
            "valid": True, # Explicitly set valid status for success
            "error_message": None # Clear error message on successful composition
        }