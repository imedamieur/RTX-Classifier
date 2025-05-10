"""
CalibrateNode for the RTX Classifier.

Applies temperature scaling to logits to convert them to calibrated probabilities.
"""
from typing import Dict, List, Any, Optional
import os
import dotenv
import numpy as np
from scipy.special import softmax

# Load environment variables
dotenv.load_dotenv()

# Get configuration from environment variables
DEFAULT_CALIBRATION_TEMP = float(os.getenv("CALIBRATION_TEMP", "0.4"))


class CalibrateNode:
    """
    CalibrateNode that calibrates the averaged logits using temperature scaling.
    
    Temperature scaling is a simple but effective calibration method that
    divides the logits by a temperature parameter before applying softmax.
    Higher temperature values smooth out the probability distribution.
    """
    
    def __init__(self, temperature: float = None):
        """
        Initialize the CalibrateNode.
        
        Args:
            temperature: Temperature scaling parameter (T > 1.0 smooths probabilities)
        """
        self.temperature = temperature if temperature is not None else DEFAULT_CALIBRATION_TEMP
    
    def _apply_temperature_scaling(self, logits: List[float]) -> List[float]:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logit values
            
        Returns:
            Calibrated probabilities
        """
        # Convert to numpy array
        logits_array = np.array(logits)
        
        # Apply temperature scaling
        scaled_logits = logits_array / 0.4
        
        # Apply softmax to get probabilities
        probabilities = softmax(scaled_logits).tolist()
        
        return probabilities
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and calibrate logits to probabilities.
        
        Args:
            state: Input state with averaged logits
            
        Returns:
            Updated state with calibrated probabilities
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)
        print(f"DEBUG: CalibrateNode: Entry state valid: {new_state.get('valid')}")
        print(f"DEBUG: CalibrateNode: Entry state avg_logits: {new_state.get('avg_logits')}")

        # Check if previous steps failed
        if not new_state.get("valid", False):
            print("DEBUG: CalibrateNode: Exiting early, state not valid.")
            return new_state
        
        # Get averaged logits
        avg_logits = new_state.get("avg_logits")
        
        # Check if we have valid data
        if avg_logits is None:
            new_state["error_message"] = "No logits to calibrate"
            new_state["valid"] = False
            print("DEBUG: CalibrateNode: Exiting, avg_logits is None.")
            # Ensure prob_vector is empty if calibration fails this way
            new_state["prob_vector"] = [] 
            return new_state
        
        # Ensure avg_logits is not empty before scaling
        if not avg_logits: # Check for empty list specifically
            new_state["error_message"] = "Cannot calibrate empty avg_logits"
            new_state["valid"] = False
            new_state["prob_vector"] = []
            print("DEBUG: CalibrateNode: Exiting, avg_logits is empty list.")
            return new_state
            
        # Apply temperature scaling
        probabilities = self._apply_temperature_scaling(avg_logits)
        
        # Store the probabilities in the state
        new_state["prob_vector"] = probabilities
        print(f"DEBUG: CalibrateNode: Calculated prob_vector: {probabilities}")
        
        return new_state