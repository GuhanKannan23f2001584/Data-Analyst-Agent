import os
import base64
import requests
import json
from dotenv import load_dotenv

load_dotenv()

AI_PIPE_API_KEY = os.getenv("AI_PIPE_API_KEY")
if not AI_PIPE_API_KEY:
    print("Error: AI_PIPE_API_KEY not found in environment variables")
    exit(1)

file_path = "temp_audio.opus"

if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found")
    exit(1)

print(f"Reading {file_path}...")
with open(file_path, "rb") as f:
    audio_data = f.read()

b64 = base64.b64encode(audio_data).decode("utf-8")
audio_format = "ogg" 

print("Sending to Audio LLM (via AI Pipe - Gemini)...")

# Gemini Native API via AI Pipe
# Model: gemini-2.0-flash-lite
url = "https://aipipe.org/geminiv1beta/models/gemini-2.0-flash-lite:generateContent"

headers = {
    "Authorization": f"Bearer {AI_PIPE_API_KEY}",
    "Content-Type": "application/json"
}

# Gemini payload format
# mime_type for opus is usually audio/ogg or audio/opus. Using audio/ogg as per previous logic.
mime_type = "audio/ogg" 

payload = {
    "contents": [{
        "parts": [
            {"text": "Please transcribe this audio file."},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64
                }
            }
        ]
    }]
}

try:
    resp = requests.post(url, headers=headers, json=payload)
    
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} - {resp.text}")
        exit(1)
        
    result = resp.json()
    # Extract text from Gemini response
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        print(f"👂 Transcription Result: {text}")
    except (KeyError, IndexError) as e:
        print(f"Error parsing transcription response: {result} - {e}")

except Exception as e:
    print(f"Error invoking Audio LLM: {e}")
    if 'resp' in locals():
        print(f"Response: {resp.text}")
