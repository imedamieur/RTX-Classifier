"""
VerifyNode for the RTX Classifier.

Verifies that the rationale cites only retrieved text and contains no forbidden terms.
Supports both OpenAI and Google Gemini models.
"""
from typing import Dict, List, Any, Optional, Set
import os
import re
import dotenv

# Load environment variables
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Get configuration from environment variables
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


class VerifyNode:
    """
    VerifyNode that verifies the rationale from the classification.
    
    Checks that:
    1. The rationale cites only content from retrieved texts
    2. The rationale contains no forbidden terms
    3. The rationale properly supports the classification
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        forbidden_terms: Optional[List[str]] = None,
        template_path: Optional[str] = None,
    ):
        """
        Initialize the VerifyNode.
        
        Args:
            llm: LangChain chat model instance
            forbidden_terms: List of terms that should not appear in the rationale
            template_path: Path to the Jinja template for the verification prompt
        """
        self.llm = llm
        self.forbidden_terms = set(forbidden_terms or [])
        self.template_path = template_path
        
        # Default verification prompt
        self._prompt_template = self._get_default_template()
        self.output_parser = JsonOutputParser()
        
        # Create default LLM if none provided
        if self.llm is None:
            model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").lower()
            if model_provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                )
            elif model_provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                    convert_system_message_to_human=True,
                )
    
    def _get_default_template(self) -> ChatPromptTemplate:
        """Get default template for verification."""
        template = """\
        You are a financial auditor reviewing a classification of an accounting transaction 
        according to AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) standards.

        The transaction details are as follows:
        Entries:
        {% for entry in entries %}
        - {{ entry }}
        {% else %}
        No specific journal entries provided.
        {% endfor %}

        Context: {{ context if context else "No additional context provided." }}
        Adjustments: {{ adjustments if adjustments else "No specific adjustments mentioned." }}
        Accounting Treatment: {{ accounting_treatment if accounting_treatment else "No specific accounting treatment mentioned." }}

        The transaction has been classified as: {{ provisional_label_name }}
        
        The provisional rationale provided is:
        {% if provisional_rationale and provisional_rationale != "Rationale could not be determined from individual runs." and provisional_rationale != "No rationale generated" %}
        {{ provisional_rationale }}
        {% else %}
        No detailed provisional rationale was provided.
        {% endif %}
        
        The classification was based on the following retrieved sources:
        {% for text in retrieved_texts %}
        --- Source {{loop.index}} ---
        {{ text }}
        {% endfor %}
        
        Please perform the following:
        1. Assess the provisional rationale (if a detailed one was provided) against the sources and the classification.
        2. Check for hallucinated information and ensure it only cites content that is actually present in the retrieved sources.
        3. Ensure it properly supports the classification and makes clear AAOIFI standard citations (e.g., "FAS 28, paragraph 3.1") for each claim or statement made.

        If a detailed provisional rationale was provided AND it meets all verification criteria, use it as the basis for your response.
        Otherwise (if no detailed provisional rationale was provided, or if the provided one has issues that cannot be simply corrected), YOU MUST CONSTRUCT a new, valid, and comprehensive rationale based on the '{{ provisional_label_name }}' classification and the retrieved sources.
        This new rationale must meet all criteria mentioned above (citing sources, no hallucination, supporting classification, AAOIFI citations).
        
        Return your verification in this JSON format:
        {
          "valid": true/false,  // true if a satisfactory rationale (either the original verified, or a newly constructed one that meets all criteria) is present in the 'rationale' field. false if issues persist or a new rationale could not be constructed.
          "issues": ["list", "of", "issues found in the original rationale if one was provided and it was problematic", "or 'No detailed provisional rationale provided, new rationale constructed/attempted.' if applicable"],
          "rationale": "The verified or newly constructed rationale. This should be a single, consolidated text. If construction of a new valid rationale failed (e.g. due to insufficient information in sources for the given classification), explain why clearly.",
          "citations": ["list", "of", "specific standard citations like FAS 10, para 3.1 or an empty list if none are applicable/found"]
        }
        """
        return ChatPromptTemplate.from_template(template, template_format="jinja2")
    
    def _contains_forbidden_terms(self, text: str) -> List[str]:
        """
        Check if the text contains any forbidden terms.
        
        Args:
            text: Text to check
            
        Returns:
            List of found forbidden terms, empty if none
        """
        if not self.forbidden_terms:
            return []
            
        found_terms = []
        for term in self.forbidden_terms:
            if term.lower() in text.lower():
                found_terms.append(term)
                
        return found_terms
    
    def _run_llm_verification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the LLM-based verification on the rationale.
        
        Args:
            state: Current state with provisional rationale and transaction details
            
        Returns:
            Verification results
        """
        if not self.llm:
            return {
                "valid": False, 
                "issues": ["No LLM available for verification"],
                "rationale": state.get("provisional_rationale", "Verification skipped: No LLM."),
                "citations": []
            }
        
        # Prepare variables for the template
        inputs = {
            "entries": state.get("entries", []),
            "context": state.get("context"),
            "adjustments": state.get("adjustments"),
            "accounting_treatment": state.get("accounting_treatment"),
            "provisional_label_name": state.get("provisional_label_name", ""),
            "provisional_rationale": state.get("provisional_rationale", ""),
            "retrieved_texts": state.get("retrieved_texts", [])[:5],  # Limit to avoid context overflow
        }
        
        # Create the chain
        chain = self._prompt_template | self.llm | self.output_parser
        
        # Call the LLM
        try:
            result = chain.invoke(inputs)
            # Ensure basic structure even if LLM output is partial but parsable
            if "rationale" not in result:
                result["rationale"] = ""
            if "citations" not in result:
                result["citations"] = []
            if "issues" not in result:
                result["issues"] = [] if result.get("valid") else ["Incomplete response from verifier LLM."]
            return result
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Verification LLM error: {str(e)}"],
                "rationale": state.get("provisional_rationale", "Error during LLM verification call."),
                "citations": []
            }
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and verify the rationale.
        
        Args:
            state: Input state with provisional label and rationale
            
        Returns:
            Updated state with verification results
        """
        # Make a copy to avoid modifying the input
        new_state = dict(state)
        
        # Check if previous steps failed. If RationaleNode (or any earlier node) set valid=False,
        # VerifyNode should not proceed with its specific logic and just return the state.
        if not new_state.get("valid", False):
            return new_state
        
        # provisional_rationale and provisional_label_name are expected to be set by RationaleNode
        # if the state is still valid at this point. An empty provisional_rationale is a valid input
        # for verification (it will likely be found invalid by the verifier LLM, which is correct).
        provisional_rationale = new_state.get("provisional_rationale", "")
        # provisional_label_name = new_state.get("provisional_label_name", "") # This is used by _run_llm_verification
        print(f"Provisional Rationale for {new_state.get('provisional_label_name')}: {provisional_rationale}")

        # The block previously here for checking missing provisional_rationale or provisional_label_name
        # has been removed. If RationaleNode failed to provide them due to its own input issues,
        # it should have set new_state["valid"] = False, and this node would have returned early.

        # Check for forbidden terms
        rationale_to_verify = provisional_rationale
        
        forbidden_found = self._contains_forbidden_terms(rationale_to_verify)
        
        if forbidden_found:
            new_state["valid"] = False
            new_state["error_message"] = f"Rationale contains forbidden terms: {', '.join(forbidden_found)}"
            new_state["retry_count"] = new_state.get("retry_count", 0) + 1
            new_state["final_rationale"] = rationale_to_verify
            new_state["sources"] = [] # No sources if forbidden terms found
            return new_state
        
        # Run LLM verification
        verification_result = self._run_llm_verification(new_state)
        
        verified_rationale = verification_result.get("rationale", "")
        verified_citations = verification_result.get("citations", [])
        verification_issues = verification_result.get("issues", [])

        if verification_result.get("valid", False):
            new_state["valid"] = True
            verified_rationale_content = verification_result.get("rationale", "")
            verified_citations_content = verification_result.get("citations", [])
            verification_issues_content = verification_result.get("issues", [])

            new_state["final_rationale"] = verified_rationale_content
            new_state["sources"] = verified_citations_content
            
            original_rationale_text = new_state.get("provisional_rationale", "")
            current_label_name = new_state.get('provisional_label_name', 'Unknown Label')

            if verification_issues_content:
                new_state["correction_notes"] = f"Provisional rationale for '{current_label_name}' was verified. Issues found/addressed by verifier: {'; '.join(verification_issues_content)}"
            elif verified_rationale_content != original_rationale_text:
                new_state["correction_notes"] = f"Provisional rationale for '{current_label_name}' was refined by verifier for clarity/accuracy."
            else:
                new_state["correction_notes"] = f"Provisional rationale for '{current_label_name}' was verified without changes."
        else:
            new_state["valid"] = False
            # If LLM didn't provide issues, create a generic one.
            error_message_parts = list(verification_result.get("issues", [])) # Start with issues from LLM
            if not error_message_parts:
                error_message_parts.append("Verification failed for unknown reasons or LLM did not provide details.")
            
            verified_rationale_content = verification_result.get("rationale", "") # Rationale from LLM, even if invalid
            if not verified_rationale_content and "Missing Rationale" not in str(error_message_parts): # Check if already covered
                 error_message_parts.append("Missing Rationale from verifier.")

            # If the label is not N/A, and citations are missing, and the LLM hasn't already flagged it.
            is_missing_citation_issue = any("citation" in issue.lower() for issue in error_message_parts)
            verified_citations_content = verification_result.get("citations", [])
            if new_state.get("provisional_label_name") and new_state.get("provisional_label_name") != "N/A" and not verified_citations_content and not is_missing_citation_issue:
                error_message_parts.append("Missing AAOIFI Standard Citation from verifier.")

            new_state["error_message"] = "Verification failed: " + "; ".join(error_message_parts)
            new_state["retry_count"] = new_state.get("retry_count", 0) + 1
            
            # Store whatever the verifier LLM returned, even if it's an error or incomplete
            new_state["final_rationale"] = verified_rationale_content 
            new_state["sources"] = verified_citations_content
        
        print(f"Final Rationale for {new_state.get('provisional_label_name')}: {new_state.get('final_rationale')}")
        return new_state