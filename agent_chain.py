import os
import sys
import base64
import json
import subprocess
import time
from typing import TypedDict, List, Optional, Any

# Adhering to your provided docs imports
from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
import requests
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    #model_name="x-ai/grok-4.1-fast",
    model_name="google/gemini-2.0-flash-exp:free",
    temperature=0.1
)

# Vision and Audio LLM configurations are moved to their respective tools using requests

# --- 1. Define State (Memory) ---
# As per docs: "Custom state schemas must extend AgentState as a TypedDict"
class QuizState(AgentState):
    email: str
    secret: str
    current_url: str
    screenshot_path: Optional[str]
    # We can store intermediate data here if needed by tools
    downloaded_files: List[str] 

# --- 2. Define Tools ---

@tool
def navigate(url: str) -> str:
    """
    Navigates to the given URL. 
    Returns the page text content and detects links/audio. 
    Saves a screenshot to 'screenshot.png' for the vision tool.
    """
    if url.endswith("/submit"):
        return "Error: You are trying to navigate to a submission URL. DO NOT navigate to it. Use `submit_answer` to POST to it instead."

    print(f"🌐 Navigating to: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle")
            # Save screenshot for vision tool
            page.screenshot(path="screenshot.png")
            
            content = page.evaluate("document.body.innerText")
            
            # Helper to find links
            links = page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({href: a.href, text: a.innerText}))
            }""")
            
            # Helper to find audio
            audio = page.evaluate("document.querySelector('audio') ? document.querySelector('audio').src : null")

            # Helper to find submission url
            submit_url = page.evaluate("""() => {
                const form = document.querySelector('form');
                if (form && form.action) return form.action;
                const submitLink = Array.from(document.querySelectorAll('a')).find(a => a.innerText.toLowerCase().includes('submit'));
                if (submitLink) return submitLink.href;
                return null;
            }""")

            return json.dumps({
                "text_preview": content[:2000], # Truncated to save context
                "relevant_links": [l for l in links if "csv" in l['text'].lower() or "download" in l['text'].lower()],
                "audio_url": audio,
                "submission_url": submit_url,
                "status": "Screenshot saved to 'screenshot.png'. content loaded."
            })
        except Exception as e:
            return f"Error: {e}"
        finally:
            browser.close()

@tool
def python_repl(code: str) -> str:
    """
    Executes Python code. Use this for data analysis, scraping, or math.
    The code MUST print the final answer to stdout.
    """
    print("💻 Executing Code...")
    print(f"--- CODE START ---\n{code}\n--- CODE END ---")
    try:
        # We run this in a subprocess to ensure clean state
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, 
            text=True, 
            timeout=30
        )
        print("result", result)
        if result.returncode == 0:
            return f"STDOUT:\n{result.stdout}"
        else:
            return f"STDERR:\n{result.stderr}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

@tool
def analyze_vision(query: str) -> str:
    """
    Analyzes the 'screenshot.png' saved by the navigate tool.
    Use this if the answer requires looking at a chart, image, or layout.
    """
    print("👁️ Analyzing Vision...")
    if not os.path.exists("screenshot.png"):
        return "Error: No screenshot found. Navigate to a page first."
        
    with open("screenshot.png", "rb") as img:
        b64 = base64.b64encode(img.read()).decode("utf-8")
        
    # Using direct request to AI Pipe (OpenRouter endpoint)
    ai_pipe_key = os.getenv("AI_PIPE_API_KEY", "")
    if not ai_pipe_key:
        return "Error: AI_PIPE_API_KEY not found."

    url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {ai_pipe_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-5-nano",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": f"Answer this question based on the image: {query}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}
        ],
        "temperature": 0.1
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"👁️ Vision Result: {content}")
        return content
    except Exception as e:
        return f"Vision Analysis Error: {e}"

@tool
def transcribe_audio(url: str) -> str:
    """
    Downloads audio from the URL and transcribes it using the AI Pipe model.
    """
    print(f"👂 Transcribing: {url}")
    try:
        resp = requests.get(url)
        # Robust mime type detection
        path = urlparse(url).path
        if path.endswith('.opus'):
            content_type = 'audio/ogg'
        elif path.endswith('.mp3'):
            content_type = 'audio/mpeg'
        elif path.endswith('.wav'):
            content_type = 'audio/wav'
        else:
            content_type = resp.headers.get('Content-Type', 'audio/mpeg')
            
        b64 = base64.b64encode(resp.content).decode("utf-8")
        filename = os.path.basename(urlparse(url).path) or "audio.mp3"
        
        # Using direct request to AI Pipe (Gemini endpoint)
        ai_pipe_key = os.getenv("AI_PIPE_API_KEY", "")
        if not ai_pipe_key:
            return "Error: AI_PIPE_API_KEY not found."

        # Gemini Native API via AI Pipe
        url = "https://aipipe.org/geminiv1beta/models/gemini-2.0-flash-lite:generateContent"
        
        headers = {
            "Authorization": f"Bearer {ai_pipe_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Please transcribe this audio file."},
                    {
                        "inline_data": {
                            "mime_type": content_type,
                            "data": b64
                        }
                    }
                ]
            }]
        }
        
        print("Waiting for transcription result...")
        resp = requests.post(url, headers=headers, json=payload)
        
        if resp.status_code != 200:
            return f"Transcription API Error: {resp.status_code} - {resp.text}"
            
        result = resp.json()
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            print(f"👂 Transcription Result: {text}")
            return text
        except (KeyError, IndexError) as e:
            return f"Error parsing transcription response: {result} - {e}"
    except Exception as e:
        return f"Transcription Error: {e}"

@tool
def analyze_file(url: str, query: str) -> str:
    """
    Downloads a file (e.g. CSV, PDF) and analyzes it using the vision LLM.
    Use this for CSV files or other documents.
    """
    print(f"📂 Analyzing File: {url}")
    try:
        resp = requests.get(url)
        content_type = resp.headers.get('Content-Type', 'application/octet-stream')
        b64 = base64.b64encode(resp.content).decode("utf-8")
        filename = os.path.basename(urlparse(url).path) or "file.dat"
        
        # Using direct request to AI Pipe (OpenRouter endpoint for vision/file analysis)
        ai_pipe_key = os.getenv("AI_PIPE_API_KEY", "")
        if not ai_pipe_key:
            return "Error: AI_PIPE_API_KEY not found."

        url = "https://aipipe.org/openrouter/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {ai_pipe_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-5-nano",
            "messages": [
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this file: {query}"
                    },
                    {
                        "type": "file", # Assuming support or fallback
                        "file": {
                            "filename": filename,
                            "file_data": f"data:{content_type};base64,{b64}"
                        }
                    }
                ]}
            ],
            "temperature": 0.1
        }
        
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"📂 File Analysis Result: {content}")
        return content
    except Exception as e:
        return f"File Analysis Error: {e}"

@tool
def submit_answer(submission_url: str, quiz_url: str, email: str, secret: str, answer: Any) -> str:
    """
    Submits the answer to the quiz endpoint.
    submission_url: The URL to POST the answer to (extracted from the page).
    quiz_url: The URL of the current quiz page (for the payload).
    """
    print(f"🚀 Submitting answer: {answer} to {submission_url}")
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url, 
        "answer": answer
    }
    print("payload", payload)
    
    try:
        resp = requests.post(submission_url, json=payload)
        print(f"🚀 Submission Response: {resp.json()}")
        return json.dumps(resp.json())
    except Exception as e:
        return f"Network Error: {e}"

# --- 3. Create Agent ---

# Initialize tools list
tools = [navigate, python_repl, analyze_vision, transcribe_audio, analyze_file, submit_answer]

# Define System Prompt
sys_prompt = """You are an automated quiz solver.
1. Use `navigate` to visit the 'current_url' provided in the user message.
2. Analyze the 'text_preview', links, audio, and **submission_url**.
3. If there is a CSV or file, use `analyze_file` to download and analyze it with the vision model.
4. If there is audio, use `transcribe_audio`.
5. If visual analysis is needed, use `analyze_vision`.
6. Once you have an answer, use `submit_answer`.
   - You MUST use the `submission_url` found by the `navigate` tool.
   - Pass the current quiz URL as `quiz_url`.
   - **DO NOT** use `navigate` to visit the `submission_url`. Only use `submit_answer` to POST to it.
7. IMPORTANT: 
   - If `submit_answer` returns {"correct": true}, extracting the 'next_url' or 'url' from the response.
   - Output the phrase "NEXT_URL: <url>" as your final response so the loop can continue.
   - If it returns false, read the reason and try again.
8. When you call the `transcribe_audio` tool wait until it returns a answer before proceeding.
"""

# Create the agent using the syntax from your docs
agent = create_agent(
    llm,
    tools=tools,
    state_schema=QuizState,
    system_prompt=sys_prompt
)

# --- 4. Execution Loop ---

def solve_quiz(start_url, email, secret):
    current_url = start_url
    
    while True:
        print(f"\\n--- 🏁 Starting Level: {current_url} ---")
        
        # Invoke the agent
        # We pass the state elements required by our QuizState TypedDict
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Solve the quiz at url: {current_url}"}],
            "email": email,
            "secret": secret,
            "current_url": current_url,
            "screenshot_path": None,
            "downloaded_files": []
        })
        
        # The agent finishes when it decides it's done. 
        # We need to parse the final output to find the next URL.
        last_message = result["messages"][-1].content
        print(f"🤖 Agent Final Output: {last_message}")
        
        if "NEXT_URL:" in last_message:
            # Extract next URL
            parts = last_message.split("NEXT_URL:")
            next_url = parts[1].strip().split()[0] # basic parsing
            if next_url == current_url or "null" in next_url:
                print("🎉 Quiz Complete or No New URL.")
                break
            current_url = next_url
        else:
            print("❌ Agent did not return a next URL. Stopping.")
            break

# --- Run ---
if __name__ == "__main__":
    # Configure your credentials here
    MY_EMAIL = "your_email"
    MY_SECRET = "your_secret"
    START_URL = "https://tds-llm-analysis.s-anand.net/demo"
    
    solve_quiz(START_URL, MY_EMAIL, MY_SECRET)
