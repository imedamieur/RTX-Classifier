import streamlit as st
import json
from rtx_classifier.pipeline import run_classifier
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
if st.button("Run Classifier", key="run_classifier_button"):
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
        st.subheader("Classification Result:")
        with st.spinner("Running classifier..."):
            try:
                result = run_classifier(json_data_to_process, model_provider=st.session_state.model_provider, model_name=st.session_state.model_name)
                
                if "rationale" in result:
                    st.write("**Rationale:**")
                    st.write(result["rationale"])

                if "sources" in result and result["sources"]:
                    st.write("**Relevant FAS Documents:**")
                    for source in result["sources"]:
                        st.markdown(f"- {source}") 
                else:
                    st.write("No specific FAS documents identified.")

                if "prob_vector" in result and result["prob_vector"]:
                    st.write("**Probability Vector:**")
                    if isinstance(result["prob_vector"], list) and len(result["prob_vector"]) == len(LABELS):
                        for i, label_code in enumerate(LABELS):
                            full_name = STANDARD_FULL_NAMES.get(label_code, label_code)
                            probability = result["prob_vector"][i]
                            st.markdown(f"- **{full_name} ({label_code})**: {probability:.4f}")
                    else:
                        st.write("Probability vector format is unexpected or does not match known labels. Displaying raw vector:")
                        st.json(result["prob_vector"])
                else:
                    st.write("No probability vector provided in the result.")
                
                # Optionally display full result for debugging
                # st.write("**Full Result (for debugging):**")
                # st.json(result)

            except Exception as e:
                st.error(f"Error running classifier: {e}")
                st.exception(e) # Provides more detailed traceback for debugging
    elif st.session_state.get('run_classifier_button'): # If button was pressed but input wasn't valid
        # Specific warnings should have already been displayed.
        # You could add a generic one here if needed, but it might be redundant.
        # st.error("Input is not valid. Please check the warnings above.")
        pass

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
