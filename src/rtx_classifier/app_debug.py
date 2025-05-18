import streamlit as st
import json
from rtx_classifier.pipeline import run_classifier
from rtx_classifier.pipeline_debug import run_classifier_debug
from rtx_classifier.run import load_dotenv
from rtx_classifier.nodes.classify import LABELS, STANDARD_FULL_NAMES

# Load environment variables
load_dotenv()

st.title("RTX Classifier")

# --- Input Method Selection ---
input_method = st.radio(
    "Choose input method:",
    ('Form', 'Upload JSON'),
    key='input_method_selector',
    horizontal=True
)

# Initialize session state for entries if it doesn't exist (for Form method)
if 'entries' not in st.session_state:
    st.session_state.entries = []

# To store data if JSON is uploaded
json_data_from_file = None
# To store the uploaded file reference itself to check if a new file was uploaded
if 'uploaded_file_ref' not in st.session_state:
    st.session_state.uploaded_file_ref = None

# For storing debug mode pipeline results
if 'debug_results' not in st.session_state:
    st.session_state.debug_results = None


# --- Define entry manipulation functions (used by Form input) ---
def add_entry():
    st.session_state.entries.append({"account": "", "debit": 0.0, "credit": 0.0})

def remove_entry(index):
    st.session_state.entries.pop(index)


# --- Conditional UI for Form or JSON Upload ---
if input_method == 'Form':
    st.header("Enter Transaction Details (Form)")
    context_input = st.text_area("Context", height=100, help="Provide the contextual information for the transaction.", key="form_context")
    adjustments_input = st.text_area("Adjustments", height=100, help="Detail any adjustments made or required.", key="form_adjustments")
    accounting_treatment_input = st.text_area("Accounting Treatment", height=100, help="Describe the proposed or actual accounting treatment.", key="form_accounting_treatment")

    st.subheader("Journal Entries")
    for i, entry in enumerate(st.session_state.entries):
        cols = st.columns([3, 2, 2, 1])
        # Ensure keys are unique for each part of the entry
        entry["account"] = cols[0].text_input(f"Account##form_account_{i}", value=entry.get("account", ""), key=f"form_account_{i}", label_visibility="collapsed", placeholder="Account Name")
        entry["debit"] = cols[1].number_input(f"Debit##form_debit_{i}", value=entry.get("debit", 0.0), key=f"form_debit_{i}", format="%.2f", label_visibility="collapsed", help="Debit Amount")
        entry["credit"] = cols[2].number_input(f"Credit##form_credit_{i}", value=entry.get("credit", 0.0), key=f"form_credit_{i}", format="%.2f", label_visibility="collapsed", help="Credit Amount")
        cols[3].button("🗑️", key=f"form_remove_{i}", on_click=remove_entry, args=(i,), help="Remove this entry")

    if st.button("Add Entry", on_click=add_entry, key="form_add_entry"):
        pass # on_click handles the logic

elif input_method == 'Upload JSON':
    st.header("Upload Transaction JSON")
    uploaded_file = st.file_uploader("Upload a JSON file with transaction data", type="json", key="json_uploader")

    if uploaded_file is not None:
        # Check if this is a new file upload or the same one from a previous run
        if st.session_state.uploaded_file_ref != uploaded_file:
            st.session_state.uploaded_file_ref = uploaded_file # Store new file reference
            try:
                # Make sure to reset the file pointer before reading
                uploaded_file.seek(0)
                json_data_from_file = json.load(uploaded_file)
                st.session_state.json_data_from_file_cache = json_data_from_file # Cache it
                st.subheader("Uploaded JSON Data:")
                st.json(json_data_from_file)
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid JSON file.")
                st.session_state.json_data_from_file_cache = None # Clear cache on error
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.session_state.json_data_from_file_cache = None # Clear cache on error
        elif 'json_data_from_file_cache' in st.session_state and st.session_state.json_data_from_file_cache is not None:
            # If the file is the same, use the cached data
            json_data_from_file = st.session_state.json_data_from_file_cache
            st.subheader("Uploaded JSON Data (cached):") # Indicate it's cached
            st.json(json_data_from_file)
    else:
        st.session_state.uploaded_file_ref = None # Clear file reference if no file
        st.session_state.json_data_from_file_cache = None # Clear cache if no file


# --- "Run Classifier" Button Logic ---
cols = st.columns([1, 1])
debug_mode = cols[0].checkbox("Debug Mode (Show step-by-step results)", key="debug_mode")
run_button = cols[1].button("Run Classifier", key="run_classifier_button")

if run_button:
    json_data_to_process = None
    valid_input = False

    if input_method == 'Form':
        # Retrieve current values from form inputs
        current_context = st.session_state.get("form_context", "")
        current_adjustments = st.session_state.get("form_adjustments", "")
        current_accounting_treatment = st.session_state.get("form_accounting_treatment", "")

        if not current_context or not current_adjustments or not current_accounting_treatment:
            st.warning("Please fill in Context, Adjustments, and Accounting Treatment fields for the form.")
        elif not st.session_state.entries:
            st.warning("Please add at least one journal entry for the form.")
        else:
            all_entries_filled = True
            for entry in st.session_state.entries:
                if not entry["account"] or (entry["debit"] == 0.0 and entry["credit"] == 0.0):
                    all_entries_filled = False
                    break
            if not all_entries_filled:
                st.warning("Please ensure all parts of each journal entry (Account, and at least one of Debit/Credit) are filled for the form.")
            else:
                json_data_to_process = {
                    "context": current_context,
                    "adjustments": current_adjustments,
                    "accounting_treatment": current_accounting_treatment,
                    "entries": st.session_state.entries
                }
                st.subheader("Input Data (from form):")
                st.json(json_data_to_process)
                valid_input = True

    elif input_method == 'Upload JSON':
        # Use the potentially cached data from session state
        json_data_from_file_to_use = st.session_state.get('json_data_from_file_cache')

        if st.session_state.get('uploaded_file_ref') is None: # Check if a file was ever uploaded
            st.warning("Please upload a JSON file.")
        elif json_data_from_file_to_use is None:
            st.error("There was an issue with the uploaded JSON file (it might be invalid or empty). Please check and re-upload if necessary.")
        else:
            if not isinstance(json_data_from_file_to_use, dict):
                st.error("The uploaded JSON content must be a valid JSON object (dictionary).")
            else:
                # Add more specific validation for uploaded JSON structure if needed here
                # For example, check for "context", "entries" keys
                required_keys = ["context", "adjustments", "accounting_treatment", "entries"]
                if not all(key in json_data_from_file_to_use for key in required_keys):
                    st.warning(f"Uploaded JSON is missing one or more required keys: {', '.join(required_keys)}. Proceeding, but classifier might not work as expected.")
                    # You could choose to set valid_input = False here if keys are strictly mandatory
                json_data_to_process = json_data_from_file_to_use
                # Data already displayed when uploaded and cached
                valid_input = True

    if valid_input and json_data_to_process is not None:
        with st.spinner("Running classifier..."):
            try:
                if debug_mode:
                    # Run the debug pipeline which returns intermediate results
                    debug_results = run_classifier_debug(
                        json_data_to_process, 
                        model_provider=st.session_state.model_provider, 
                        model_name=st.session_state.model_name
                    )
                    st.session_state.debug_results = debug_results
                    
                    # Display the final result
                    if debug_results and len(debug_results) > 0:
                        final_result = debug_results[-1]  # Get the last result (compose node)
                        
                        # Display final classification
                        if "final_classification_label" in final_result and final_result["final_classification_label"]:
                            label_code = final_result["final_classification_label"]
                            full_name = STANDARD_FULL_NAMES.get(label_code, label_code)
                            st.subheader(f"Classification: {full_name} ({label_code})")
                          # Display the rationale
                        if "provisional_rationale" in final_result and final_result["provisional_rationale"]:
                            st.subheader("Rationale:")
                            st.write(final_result["provisional_rationale"])
                            
                        # Display alternative standards if available
                        if "alternative_standards" in final_result and final_result["alternative_standards"]:
                            st.subheader("Alternative Standards to Consider:")
                            for alt_standard in final_result["alternative_standards"]:
                                # Handle both string and dictionary formats
                                if isinstance(alt_standard, dict) and "standard" in alt_standard and "reason" in alt_standard:
                                    # Handle structured format if the LLM returns it this way
                                    standard_code = alt_standard["standard"]
                                    explanation = alt_standard["reason"]
                                    standard_name = STANDARD_FULL_NAMES.get(standard_code.replace(" ", ""), "")
                                    formatted_alt = f"**{standard_code}** ({standard_name}): {explanation}"
                                    st.markdown(formatted_alt)
                                else:
                                    # For string format, try to parse it
                                    explanation = alt_standard
                                    
                                    # Try to find standard code patterns like "FAS X" or "FAS XX"
                                    import re
                                    match = re.match(r'^(FAS\s*\d+)', alt_standard, re.IGNORECASE)
                                    if match:
                                        standard_code = match.group(1).strip()
                                        # Get the full name if available
                                        standard_name = STANDARD_FULL_NAMES.get(standard_code.replace(" ", ""), "")
                                        # Remove the standard code from the beginning of the explanation
                                        explanation = alt_standard[len(match.group(1)):].strip()
                                        # Remove leading punctuation if any
                                        explanation = explanation.lstrip(":- ")
                                        
                                        # Format with the standard name and explanation
                                        formatted_alt = f"**{standard_code}** ({standard_name}): {explanation}"
                                        st.markdown(formatted_alt)
                                    else:
                                        # If no standard code pattern is found, look for common patterns in the text
                                        pattern = r'(FAS\s*\d+)'
                                        matches = re.findall(pattern, alt_standard)
                                        if matches:
                                            # Get the first FAS mention
                                            standard_code = matches[0].strip()
                                            standard_name = STANDARD_FULL_NAMES.get(standard_code.replace(" ", ""), "")
                                            # Format with the standard name highlighted
                                            formatted_alt = alt_standard.replace(standard_code, f"**{standard_code}** ({standard_name})", 1)
                                            st.markdown(formatted_alt)
                                        else:
                                            # If no patterns found, just display as is with bullet point
                                            st.markdown(f"- {alt_standard}")
                            
                        # Note: Probability scores are intentionally not shown in the main response
                        # They are still available in the Step-by-Step Pipeline Results -> Compose tab
                
                else:
                    # Run the standard pipeline
                    result = run_classifier(
                        json_data_to_process, 
                        model_provider=st.session_state.model_provider, 
                        model_name=st.session_state.model_name
                    )
                    
                    if "label" in result and result["label"]:
                        label_code = result["label"]
                        full_name = STANDARD_FULL_NAMES.get(label_code, label_code)
                        st.subheader(f"Classification: {full_name} ({label_code})")
                    
                    if "rationale" in result and result["rationale"]:
                        st.subheader("Rationale:")
                        st.write(result["rationale"])
                    
                    if "alternative_standards" in result and result["alternative_standards"]:
                        st.subheader("Alternative Standards to Consider:")
                        for alt_standard in result["alternative_standards"]:
                            # Handle both string and dictionary formats
                            if isinstance(alt_standard, dict) and "standard" in alt_standard and "reason" in alt_standard:
                                # Handle structured format if the LLM returns it this way
                                standard_code = alt_standard["standard"]
                                explanation = alt_standard["reason"]
                                standard_name = STANDARD_FULL_NAMES.get(standard_code.replace(" ", ""), "")
                                formatted_alt = f"**{standard_code}** ({standard_name}): {explanation}"
                                st.markdown(formatted_alt)
                            else:
                                # For string format, try to parse it
                                explanation = alt_standard
                                
                                # Try to find standard code patterns like "FAS X" or "FAS XX"
                                import re
                                match = re.match(r'^(FAS\s*\d+)', alt_standard, re.IGNORECASE)
                                if match:
                                    standard_code = match.group(1).strip()
                                    # Get the full name if available
                                    standard_name = STANDARD_FULL_NAMES.get(standard_code.replace(" ", ""), "")
                                    # Remove the standard code from the beginning of the explanation
                                    explanation = alt_standard[len(match.group(1)):].strip()
                                    # Remove leading punctuation if any
                                    explanation = explanation.lstrip(":- ")
                                    
                                    # Format with the standard name and explanation
                                    formatted_alt = f"**{standard_code}** ({standard_name}): {explanation}"
                                    st.markdown(formatted_alt)
                                else:
                                    # If no standard code pattern is found, look for common patterns in the text
                                    pattern = r'(FAS\s*\d+)'
                                    matches = re.findall(pattern, alt_standard)
                                    if matches:
                                        # Get the first FAS mention
                                        standard_code = matches[0].strip()
                                        standard_name = STANDARD_FULL_NAMES.get(standard_code.replace(" ", ""), "")
                                        # Format with the standard name highlighted
                                        formatted_alt = alt_standard.replace(standard_code, f"**{standard_code}** ({standard_name})", 1)
                                        st.markdown(formatted_alt)
                                    else:
                                        # If no patterns found, just display as is with bullet point
                                        st.markdown(f"- {alt_standard}")

            except Exception as e:
                st.error(f"Error running classifier: {e}")
                st.exception(e) # Provides more detailed traceback for debugging
    elif st.session_state.get('run_classifier_button'): # If button was pressed but input wasn't valid
        # Specific warnings should have already been displayed.
        pass

# --- Display Debug Results in Tabs if Available
if debug_mode and st.session_state.debug_results is not None:
    debug_results = st.session_state.debug_results
    
    st.header("Step-by-Step Pipeline Results")
    
    # Create a tab for each node
    node_names = [result.get('node', f"Step {i}") for i, result in enumerate(debug_results)]
    node_display_names = {
        'preprocess': '1. Preprocess',
        'retrieve': '2. Retrieve',
        'classify': '3. Classify',
        'majority_vote': '4. Majority Vote',
        'calibrate': '5. Calibrate',
        'verify': '6. Verify',
        'compose': '7. Compose (Final)'
    }
    
    tabs = st.tabs([node_display_names.get(name, name) for name in node_names])
    
    # Fill each tab with the appropriate content
    for i, tab in enumerate(tabs):
        with tab:
            result = debug_results[i]
            node = result.get('node', 'Unknown')
            
            if node == 'preprocess':
                st.write("### Preprocessing")
                st.write("Validates input data (schema, balance checks, forbidden accounts)")
                
                st.write("#### Validation Results:")
                valid = result.get('valid', False)
                st.success("✅ Input data is valid") if valid else st.error("❌ Input data is invalid")
                
                if not valid and 'error_message' in result:
                    st.error(f"Error: {result['error_message']}")
                    
                st.write("#### Input Data:")
                st.write("Entries:")
                st.json(result.get('entries', []))
                
                if 'context' in result and result['context']:
                    st.write("Context:")
                    st.text(result['context'])
                    
                if 'adjustments' in result and result['adjustments']:
                    st.write("Adjustments:")
                    st.text(result['adjustments'])
                    
                if 'accounting_treatment' in result and result['accounting_treatment']:
                    st.write("Accounting Treatment:")
                    st.text(result['accounting_treatment'])
                
            elif node == 'retrieve':
                st.write("### Retrieve")
                st.write("Retrieves relevant document chunks for classification")
                # First show raw API retrieval results if available
                if 'api_retrieve_response' in result:
                    st.write("#### Raw API Retrieval Results:")
                    api_results = result.get('api_retrieve_response', {})
                    if api_results:
                        with st.expander("API Response Details"):
                            st.json(api_results)
                    else:
                        st.warning("No raw API retrieval results available")
                
                # Then show the processed retrieved texts
                st.write("#### Processed Retrieved Texts:")
                retrieved_texts = result.get('retrieved_texts', [])
                if retrieved_texts:
                    for i, text in enumerate(retrieved_texts):
                        with st.expander(f"Document Chunk {i+1}"):
                            st.text(text)
                else:
                    st.warning("No processed documents were retrieved")
                
            elif node == 'classify':
                st.write("### Classify")
                st.write("Runs multiple parallel LLM calls to classify the transaction")
                
                logits_list = result.get('logits', [])
                rationales = result.get('rationales', [])
                classifications = result.get('classifications', [])
                
                if logits_list and rationales and len(logits_list) == len(rationales):
                    for i, (logits, rationale) in enumerate(zip(logits_list, rationales)):
                        with st.expander(f"Classification Run {i+1}"):
                            if i < len(classifications):
                                st.write(f"**Classification:** {classifications[i]}")
                            
                            st.write("**Logits:**")
                            for j, label in enumerate(LABELS):
                                if j < len(logits):
                                    st.write(f"- {label}: {logits[j]:.4f}")
                            
                            st.write("**Rationale:**")
                            st.write(rationale)
                else:
                    st.warning("No parallel classification results available")
                
            elif node == 'majority_vote':
                st.write("### Majority Vote")
                st.write("Aggregates results from multiple classification runs")
                
                avg_logits = result.get('avg_logits', [])
                provisional_label_name = result.get('provisional_label_name', 'N/A')
                provisional_rationale = result.get('provisional_rationale', '')
                
                st.write(f"#### Provisional Label: {provisional_label_name}")
                
                if avg_logits and len(avg_logits) == len(LABELS):
                    st.write("#### Average Logits:")
                    for i, (label, logit) in enumerate(zip(LABELS, avg_logits)):
                        st.write(f"- {label}: {logit:.4f}")
                else:
                    st.warning("No average logits available")
                
                if provisional_rationale:
                    st.write("#### Chosen Rationale:")
                    st.write(provisional_rationale)
                    
            elif node == 'calibrate':
                st.write("### Calibrate")
                st.write("Applies temperature scaling to the logits to get final probabilities")
                
                prob_vector = result.get('prob_vector', [])
                
                if prob_vector and len(prob_vector) == len(LABELS):
                    st.write("#### Probability Vector (after calibration):")
                    for i, (label, prob) in enumerate(zip(LABELS, prob_vector)):
                        full_name = STANDARD_FULL_NAMES.get(label, label)
                        st.write(f"- **{full_name} ({label})**: {prob:.4f}")
                else:
                    st.warning("No probability vector available")
                    
            elif node == 'verify':
                st.write("### Verify")
                st.write("Verifies the classification and provides the final label")
                
                valid = result.get('valid', False)
                final_label = result.get('final_classification_label', 'N/A')
                
                if valid:
                    st.success(f"✅ Verification successful - Final label: {final_label}")
                else:
                    st.error("❌ Verification failed")
                    if 'error_message' in result:
                        st.error(f"Error: {result['error_message']}")
                    
                retry_count = result.get('retry_count', 0)
                st.write(f"Retry count: {retry_count}")
                
            elif node == 'compose':
                st.write("### Compose (Final Output)")
                st.write("Composes the final output for the API")
                
                prob_vector = result.get('prob_vector', [])
                valid = result.get('valid', False)
                
                st.write("#### Final Result:")
                if valid:
                    st.success("✅ Classification successful")
                    
                    if 'final_classification_label' in result and result['final_classification_label']:
                        label_code = result['final_classification_label']
                        full_name = STANDARD_FULL_NAMES.get(label_code, label_code)
                        st.write(f"**Classification:** {full_name} ({label_code})")
                    
                    if prob_vector and len(prob_vector) == len(LABELS):
                        st.write("**Probability Scores:**")
                        for i, (label, prob) in enumerate(zip(LABELS, prob_vector)):
                            full_name = STANDARD_FULL_NAMES.get(label, label)
                            st.write(f"- {full_name} ({label}): {prob:.4f}")
                    
                    if 'provisional_rationale' in result and result['provisional_rationale']:
                        st.write("**Rationale:**")
                        st.write(result['provisional_rationale'])
                else:
                    st.error("❌ Classification failed")
                    if 'error_message' in result:
                        st.error(f"Error: {result['error_message']}")
            
            # Show raw state for debugging
            with st.expander("View raw state data"):
                # Remove long text fields from display to make it cleaner
                display_data = {k: v for k, v in result.items() if k not in ['retrieved_texts', 'rationales', 'provisional_rationale']}
                st.json(display_data)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
st.sidebar.selectbox("Model Provider", ["openai", "google"], index=0, key="model_provider")
st.sidebar.text_input("Model Name (optional)", "", key="model_name", help="Default will be used if empty (e.g., gpt-4o or gemini-1.5-pro-latest)")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses the RTX Classifier to classify accounting transactions "
    "based on AAOIFI standards."
)

if __name__ == "__main__":
    # Streamlit runs the script from top to bottom on every interaction.
    # The run_classifier function picks up model_provider and model_name
    # from st.session_state, which are set by the sidebar widgets.
    pass
