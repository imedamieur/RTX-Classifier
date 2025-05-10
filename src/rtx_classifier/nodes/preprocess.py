"""
PreprocessNode for the RTX Classifier.

Responsible for schema validation, balance checks, and forbidden account validation.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, cast

from langgraph.graph import StateGraph


@dataclass
class AccountEntry:
    """Represents a single accounting entry."""
    account: str
    debit: float
    credit: float


class PreprocessNode:
    """
    Preprocessing node that validates input data:
    - Schema validation
    - Balance checks (debits = credits)
    - Forbidden account checks
    """
    
    def __init__(self, forbidden_accounts: Optional[List[str]] = None):
        """
        Initialize the PreprocessNode.
        
        Args:
            forbidden_accounts: List of accounts that should not appear in valid transactions
        """
        self.forbidden_accounts = forbidden_accounts or []
    
    def validate_schema(self, state: Dict[str, Any]) -> bool:
        """
        Validate the input schema.
        
        Args:
            state: Current state with entries to validate
            
        Returns:
            True if valid, False otherwise
        """
        entries = state.get("entries", [])
        
        # Check if entries is a list
        if not isinstance(entries, list):
            state["error_message"] = "Entries must be a list of account objects"
            return False
        
        # Check if all entries have required fields
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                state["error_message"] = f"Entry {i} is not an object"
                return False
            
            if "account" not in entry:
                state["error_message"] = f"Entry {i} missing 'account' field"
                return False
            
            if not isinstance(entry["account"], str):
                state["error_message"] = f"Entry {i} 'account' must be a string"
                return False
            
            # Check debit and credit fields
            if "debit" not in entry and "credit" not in entry:
                state["error_message"] = f"Entry {i} missing both 'debit' and 'credit' fields"
                return False
            
            if "debit" in entry and not isinstance(entry["debit"], (int, float)):
                state["error_message"] = f"Entry {i} 'debit' must be a number"
                return False
            
            if "credit" in entry and not isinstance(entry["credit"], (int, float)):
                state["error_message"] = f"Entry {i} 'credit' must be a number"
                return False
        
        return True
    
    def check_balance(self, state: Dict[str, Any]) -> bool:
        """
        Check if debits equal credits.
        
        Args:
            state: Current state with entries to validate
            
        Returns:
            True if balanced, False otherwise
        """
        entries = state.get("entries", [])
        total_debits = sum(entry.get("debit", 0.0) for entry in entries)
        total_credits = sum(entry.get("credit", 0.0) for entry in entries)
        
        # Use a small epsilon to account for floating point rounding errors
        epsilon = 0.0001
        if abs(total_debits - total_credits) > epsilon:
            state["error_message"] = f"Transaction is not balanced: debits ({total_debits}) ≠ credits ({total_credits})"
            return False
        
        return True
    
    def check_forbidden_accounts(self, state: Dict[str, Any]) -> bool:
        """
        Check if any forbidden accounts are used.
        
        Args:
            state: Current state with entries to validate
            
        Returns:
            True if no forbidden accounts, False otherwise
        """
        if not self.forbidden_accounts:
            return True
            
        entries = state.get("entries", [])
        for entry in entries:
            if entry.get("account") in self.forbidden_accounts:
                state["error_message"] = f"Forbidden account used: {entry.get('account')}"
                return False
        
        return True
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and validate the transaction data.
        
        Args:
            state: Input state with transaction entries
            
        Returns:
            Updated state with validation results
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)
        
        # Validate schema
        if not self.validate_schema(new_state):
            new_state["valid"] = False
            return new_state
        
        # Check balance
        if not self.check_balance(new_state):
            new_state["valid"] = False
            return new_state
        
        # Check forbidden accounts
        if not self.check_forbidden_accounts(new_state):
            new_state["valid"] = False
            return new_state
        
        # All checks passed
        new_state["valid"] = True
        return new_state