"""
Tests for the PreprocessNode balance checking functionality.
"""
import pytest
from rtx_classifier.nodes.preprocess import PreprocessNode


def test_balance_check_balanced():
    """Test that balanced transactions pass the check."""
    # Create a PreprocessNode
    node = PreprocessNode()
    
    # Create a balanced state
    state = {
        "entries": [
            {"account": "Cash", "debit": 100.0},
            {"account": "Revenue", "credit": 100.0}
        ]
    }
    
    # Process the state
    result = node.check_balance(state)
    
    # Check that the transaction is balanced
    assert result is True


def test_balance_check_unbalanced():
    """Test that unbalanced transactions fail the check."""
    # Create a PreprocessNode
    node = PreprocessNode()
    
    # Create an unbalanced state
    state = {
        "entries": [
            {"account": "Cash", "debit": 100.0},
            {"account": "Revenue", "credit": 99.0}
        ]
    }
    
    # Process the state
    result = node.check_balance(state)
    
    # Check that the transaction is unbalanced
    assert result is False
    assert "not balanced" in state["error_message"]


def test_balance_check_nearly_balanced():
    """Test that nearly balanced transactions (within epsilon) pass the check."""
    # Create a PreprocessNode
    node = PreprocessNode()
    
    # Create a nearly balanced state (small rounding error)
    state = {
        "entries": [
            {"account": "Cash", "debit": 100.0},
            {"account": "Revenue", "credit": 100.0000001}
        ]
    }
    
    # Process the state
    result = node.check_balance(state)
    
    # Check that the transaction is considered balanced
    assert result is True


def test_forbidden_accounts():
    """Test that forbidden accounts are detected."""
    # Create a PreprocessNode with forbidden accounts
    node = PreprocessNode(forbidden_accounts=["Secret Account"])
    
    # Create a state with a forbidden account
    state = {
        "entries": [
            {"account": "Secret Account", "debit": 100.0},
            {"account": "Revenue", "credit": 100.0}
        ]
    }
    
    # Process the state
    result = node.check_forbidden_accounts(state)
    
    # Check that the forbidden account is detected
    assert result is False
    assert "Forbidden account" in state["error_message"]


def test_schema_validation():
    """Test that schema validation works."""
    # Create a PreprocessNode
    node = PreprocessNode()
    
    # Test with invalid schema (missing account)
    state = {
        "entries": [
            {"debit": 100.0},
            {"account": "Revenue", "credit": 100.0}
        ]
    }
    
    # Process the state
    result = node.validate_schema(state)
    
    # Check that schema validation fails
    assert result is False
    assert "missing 'account'" in state["error_message"]