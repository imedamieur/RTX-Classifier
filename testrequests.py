#!/usr/bin/env python
# filepath: d:\Cat2\testrequests.py
"""
Simple script to test an API request.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if you have one)
load_dotenv()

def test_api_request():
    """
    Test a basic API request and print the response.
    This is a sample function that makes a GET request to a public API.
    Modify the URL and parameters as needed for your specific API.
    """
    # Example API endpoint - replace with your actual API endpoint
    url = "192.168.72.170:8000/docs"
    
    # Add any headers you might need
    headers = {
        "Content-Type": "application/json",
        # If you need an API key, you could get it from environment variables
        # "Authorization": f"Bearer {os.getenv('API_KEY')}"
    }
    
    # For GET requests
    try:
        response = requests.get(url, headers=headers)
        # Check if the request was successful
        response.raise_for_status()
        
        # Print response details
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print("\nResponse Content:")
        print(json.dumps(response.json(), indent=4))
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None

def test_post_request():
    
    url = "http://172.20.10.12:8000/retrieve-fas-chunks" 

    payload = {
        "query": "Reverse transaction #2:\nContext: The client pays all outstanding amounts on time, reducing expected losses.\nAdjustments:\nLoss provision reversed.\nRecognized revenue adjusted.\nAccounting Treatment:\nReduction in impairment expense.\nRecognition of full contract revenue.\nJournal Entry for Loss Provision Reversal:\nDr. Allowance for Impairment $500,000\n Cr. Provision for Losses $500,000\nThis restores revenue after full payment.\nChallenge:\nWhat financial standard applies to impairment and expected loss provisions?",
        "top_k_fas_chunks": 10,
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        print(f"Status Code: {response.status_code}")
        print("\nResponse Content (POST request):")
        print(json.dumps(response.json(), indent=4))
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making POST request: {e}")
        return None

if __name__ == "__main__":
    
    
    print("Testing POST request...")
    test_post_request()