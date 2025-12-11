import os
import requests
import json

# Read token from environment variable
# Read token from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    # Fallback to the provided OpenRouter key
    print("WARNING: OPENROUTER_API_KEY not set, using fallback.")

def get_profile():
    """
    Simulates getting the profile.
    """
    return {"token": OPENROUTER_API_KEY}

def chat_completion(messages, model="mistralai/devstral-2512:free"):
    """
    Sends a chat completion request to OpenRouter.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo", # Optional, for OpenRouter rankings
        "X-Title": "LLM Quiz Solver" # Optional
    }
    payload = {
        "model": model,
        "messages": messages,
        "tools": [
        {
            "type": "code_interpreter",
        }
    ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter: {e}")
        if 'response' in locals() and response:
            print(f"Response content: {response.text}")
        return None

if __name__ == "__main__":
    # Test the connection
    print("Testing AI Pipe connection...")
    test_messages = [{"role": "user", "content": "What tools do you have? can you run code, if so prove it to me."}]
    result = chat_completion(test_messages)
    print(json.dumps(result, indent=2))
