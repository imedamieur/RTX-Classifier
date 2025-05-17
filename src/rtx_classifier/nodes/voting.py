"""
MajorityVoteNode for the RTX Classifier.

Aggregates multiple logits and selects the most likely classification.
"""
from typing import Dict, List, Any, Optional
import numpy as np

# Standard labels for classification
LABELS = ["FAS4", "FAS7", "FAS10", "FAS28", "FAS32"]


class MajorityVoteNode:
    """
    MajorityVoteNode that averages logits from multiple classifiers and selects
    a provisional label based on the highest average score.
    """
    
    def __init__(self, labels: Optional[List[str]] = None):
        """
        Initialize the MajorityVoteNode.
        
        Args:
            labels: List of label names (defaults to standard AAOIFI FAS labels)
        """
        self.labels = labels or LABELS
    
    def _average_logits(self, logits_list: List[List[float]]) -> List[float]:
        """
        Average multiple sets of logits.
        
        Args:
            logits_list: List of logits arrays
            
        Returns:
            Average logits
        """
        if not logits_list:
            return [1.0 / len(self.labels)] * len(self.labels)
        
        # Convert to numpy array
        logits_array = np.array(logits_list)
        
        # Average across first dimension
        avg_logits = np.mean(logits_array, axis=0).tolist()
        
        return avg_logits
    
    def _get_max_label_index(self, logits: List[float]) -> int:
        """
        Get the index of the maximum value in logits.
        
        Args:
            logits: Logits array
            
        Returns:
            Index of maximum value
        """
        return np.argmax(logits)
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and aggregate logits.
        
        Args:
            state: Input state with multiple logits arrays, rationales, classifications, and standard citations
            
        Returns:
            Updated state with averaged logits, provisional label, rationale, and standard citation
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)
        print(f"DEBUG: MajorityVoteNode: Entry state valid: {new_state.get('valid')}")
        # print(f"DEBUG: MajorityVoteNode: Entry state logits: {new_state.get('logits')}") # Can be verbose

        # Check if previous steps failed
        if not new_state.get("valid", False):
            print("DEBUG: MajorityVoteNode: Exiting early, state not valid.")
            return new_state
        
        # Get logits lists
        logits_list = new_state.get("logits", [])
        
        if not logits_list:
            new_state["error_message"] = "No logits to aggregate"
            new_state["valid"] = False
            print("DEBUG: MajorityVoteNode: Exiting, no logits to aggregate.")
            return new_state
        
        # Average logits
        avg_logits = self._average_logits(logits_list)
        new_state["avg_logits"] = avg_logits
        print(f"DEBUG: MajorityVoteNode: Calculated avg_logits: {avg_logits}")
        
        # Get provisional label index and name
        max_idx = self._get_max_label_index(avg_logits)
        new_state["provisional_label"] = max_idx # Store index
        provisional_label_name_from_voting = "Error: Label index out of bounds"
        if max_idx < len(self.labels):
            provisional_label_name_from_voting = self.labels[max_idx]
            new_state["provisional_label_name"] = provisional_label_name_from_voting
        else:
            new_state["provisional_label_name"] = provisional_label_name_from_voting
            new_state["valid"] = False # Cannot proceed without a valid label name
            new_state["error_message"] = "Provisional label index out of bounds"
            return new_state

        # Get rationales, classifications, and standard citations from the state
        rationales = new_state.get("rationales")
        


        classifications = new_state.get("classifications", [])  # From ClassifyFanOutNode
        standard_citations = new_state.get("standard_citations", [])  # From ClassifyFanOutNode

        chosen_rationale = "Rationale could not be determined from individual runs."
        chosen_standard_citation = "Standard citation could not be determined from individual runs."
        found_matching_run = False

        # Attempt 1: Find a run that matches the provisional_label_name
        if classifications and rationales and standard_citations:
            for i in range(len(classifications)):
                if classifications[i] == provisional_label_name_from_voting: # Use the one from voting node logic
                    if i < len(rationales) and i < len(standard_citations):
                        chosen_rationale = rationales[i]
                        chosen_standard_citation = standard_citations[i]
                        found_matching_run = True
                        print(f"DEBUG: MajorityVoteNode: Found matching run {i} for label {provisional_label_name_from_voting}")
                        break 
        
        # Attempt 2 (Fallback): If no direct match, find the run with the highest logit for the winning class
        if not found_matching_run and logits_list:
            print(f"DEBUG: MajorityVoteNode: No direct matching run found. Using fallback logic for rationale/citation for label {provisional_label_name_from_voting} (index {max_idx}).")
            best_run_idx_for_winning_class = -1
            max_logit_for_winning_class = -float('inf')

            for run_idx, run_logits in enumerate(logits_list):
                if max_idx < len(run_logits):  # Ensure the class index is valid for this run's logits
                    if run_logits[max_idx] > max_logit_for_winning_class:
                        max_logit_for_winning_class = run_logits[max_idx]
                        best_run_idx_for_winning_class = run_idx
            
            if best_run_idx_for_winning_class != -1:
                print(f"DEBUG: MajorityVoteNode: Fallback run index {best_run_idx_for_winning_class} had max logit {max_logit_for_winning_class} for class index {max_idx}.")
                if best_run_idx_for_winning_class < len(rationales):
                    chosen_rationale = rationales[best_run_idx_for_winning_class]
                else:
                    print(f"DEBUG: MajorityVoteNode: Fallback run index {best_run_idx_for_winning_class} out of bounds for rationales (len {len(rationales)}).")
                
                if best_run_idx_for_winning_class < len(standard_citations):
                    chosen_standard_citation = standard_citations[best_run_idx_for_winning_class]
                else:
                    print(f"DEBUG: MajorityVoteNode: Fallback run index {best_run_idx_for_winning_class} out of bounds for standard_citations (len {len(standard_citations)}).")
            else:
                print(f"DEBUG: MajorityVoteNode: Fallback logic could not determine a best run.")
        
        new_state["provisional_rationale"] = chosen_rationale
        new_state["provisional_standard_citation"] = chosen_standard_citation
        
        print(f"DEBUG: MajorityVoteNode: Provisional label name: {new_state['provisional_label_name']}")
        print(f"DEBUG: MajorityVoteNode: Provisional rationale: {chosen_rationale[:100]}...") # Print snippet
        print(f"DEBUG: MajorityVoteNode: Provisional standard citation: {chosen_standard_citation}")

      

        return new_state