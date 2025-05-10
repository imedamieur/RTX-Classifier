"""
CLI entry point for the RTX Classifier.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import dotenv

from rtx_classifier.pipeline import run_classifier


def load_dotenv():
    """Load environment variables from .env file."""
    # Look for .env file in the project root (2 directories up from this file)
    dotenv_path = Path(__file__).parent.parent.parent / ".env"
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
        return True
    return False


def load_json_input(file_path: str) -> Dict[str, Any]:
    """
    Load the input JSON from the provided file path.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        JSON payload as a dictionary
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file - {file_path}")
        sys.exit(1)


def main():
    # Load environment variables
    env_loaded = load_dotenv()
    if not env_loaded:
        print("Warning: .env file not found. Using default values or expecting command-line parameters.")
    
    # Get default model info from environment variables
    default_model = os.getenv("DEFAULT_MODEL", "gpt-4o")
    default_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Reverse-Transaction Classifier for AAOIFI Standards"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSON file containing the transaction entries"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to write the output JSON (defaults to stdout)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "google"],
        default=default_provider,
        help=f"Model provider to use (default: {default_provider})"
    )
    parser.add_argument(
        "--model", "-m",
        default=default_model,
        help=f"Model name to use for classification (default: {default_model})"
    )
    parser.add_argument(
        "--openai-key", "-ok",
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--google-key", "-gk",
        help="Google API key (defaults to GOOGLE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Shortcut to use the gemini-2.5-pro-preview-05-06 model"
    )
    
    args = parser.parse_args()
    
    # Handle the gemini shortcut flag
    if args.gemini:
        args.provider = "google"
        args.model = "gemini-2.5-pro-preview-05-06"
    
    # Load input JSON
    json_data = load_json_input(args.input_file)
    
    # Set up API keys based on the provider
    openai_api_key = None
    google_api_key = None
    
    if args.provider.lower() == "openai":
        openai_api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.provider.lower() == "google":
        google_api_key = args.google_key or os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print("Error: Google API key not provided and GOOGLE_API_KEY environment variable not set")
            sys.exit(1)
    
    # Run the classifier
    try:
        result = run_classifier(
            json_data=json_data,
            model_provider=args.provider,
            model_name=args.model,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
        )
    except Exception as e:
        print(f"Error running classifier: {str(e)}")
        sys.exit(1)
    
    # Output the result
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"Result written to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {str(e)}")
            sys.exit(1)
    else:
        print(output_json)


if __name__ == "__main__":
    main()