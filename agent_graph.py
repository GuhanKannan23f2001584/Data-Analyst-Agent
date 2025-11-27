import asyncio
import json
import os
import time
import base64
from typing import TypedDict, List, Optional, Any, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image
import io
import subprocess
import sys
import re
import whisper
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not set, using fallback.")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="x-ai/grok-4.1-fast",
    temperature=0.1
)


# Vision LLM configuration moved to vision_analyst node using requests


# --- State Definition ---
class AgentState(TypedDict):
    url: str
    email: str
    secret: str
    page_content: str
    screenshot_path: Optional[str]
    file_paths: List[str]
    messages: List[BaseMessage]
    attempts: int
    last_error: Optional[str]
    decision: Optional[str]
    
    # Extracted/Derived Data
    csv_link: Optional[str]
    scrape_path: Optional[str]
    audio_url: Optional[str]
    
    # Results
    transcription: Optional[str]
    analysis_result: Optional[str] # Renamed from csv_analysis
    scraped_content: Optional[str]
    vision_analysis: Optional[str]
    
    # Final Submission
    solution: Optional[Dict[str, Any]]
    submission_result: Optional[Dict[str, Any]]
    next_url: Optional[str]

# --- Helper Functions ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def execute_python_code(code, timeout=30):
    try:
        print(f"Executing generated code:\n{code}\n")
        temp_file = f"temp_code_{int(time.time())}.py"
        with open(temp_file, "w") as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if result.returncode == 0:
            print(f"Code execution successful. Output:\n{result.stdout}")
            return result.stdout.strip()
        else:
            print(f"Code execution failed. Error:\n{result.stderr}")
            return f"Error: {result.stderr}"
            
    except Exception as e:
        print(f"Error executing code: {e}")
        return f"Error: {str(e)}"

# --- Nodes ---

async def navigator(state: AgentState):
    print(f"--- NAVIGATOR: Visiting {state['url']} ---")
    url = state['url']
    screenshot_path = f"screenshot_{int(time.time())}.png"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            
            try:
                await page.wait_for_selector("body", timeout=5000)
                await page.wait_for_timeout(1000)
            except:
                pass
                
            await page.screenshot(path=screenshot_path)
            content = await page.evaluate("document.body.innerText")
            
            links = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a')).map(a => ({href: a.href, text: a.innerText}));
            }""")
            
            audio_src = await page.evaluate("""() => {
                const audio = document.querySelector('audio');
                return audio ? audio.src : null;
            }""")
            
            csv_link = None
            for link in links:
                if 'CSV file' in link['text'] or link['href'].endswith('.csv'):
                    csv_link = link['href']
                    break
            
            scrape_path = None
            # 1. Regex Heuristic
            match = re.search(r"Scrape\s+([^\s]+)", content)
            if match:
                path = match.group(1).rstrip(".,;)")
                if path.startswith("http"):
                    scrape_path = path
                elif path.startswith("/"):
                    base = "/".join(url.split("/")[:3])
                    scrape_path = base + path
                else:
                    base = url.rsplit('/', 1)[0]
                    scrape_path = f"{base}/{path}"
            
            # 2. LLM Fallback for general scrape targets
            if not scrape_path:
                print("Navigator: No regex match for scrape path. Asking LLM...")
                prompt = f"""
                Analyze the following page content and links. 
                Is there a specific URL that the user is being asked to 'scrape', 'analyze', or 'process'?
                Look for instructions like "scrape this link", "data is available at", "extract from", etc.
                
                Page Content:
                {content[:2000]}
                
                Links:
                {json.dumps(links[:50])}
                
                Return ONLY the URL. If none found, return 'None'.
                """
                msg = [HumanMessage(content=prompt)]
                response = await llm.ainvoke(msg)
                candidate = response.content.strip().strip('"').strip("'")
                
                if candidate.lower() != 'none':
                    if candidate.startswith("http"):
                        scrape_path = candidate
                    elif candidate.startswith("/"):
                        base = "/".join(url.split("/")[:3])
                        scrape_path = base + candidate
                    else:
                        base = url.rsplit('/', 1)[0]
                        scrape_path = f"{base}/{candidate}"
                    print(f"Navigator: LLM identified scrape path: {scrape_path}")
            
            # Correction: If scrape_path is a CSV, move it to csv_link
            if scrape_path and scrape_path.lower().endswith('.csv'):
                print(f"Navigator: Scrape path {scrape_path} is a CSV. Moving to csv_link.")
                csv_link = scrape_path
                scrape_path = None

            return {
                "page_content": content,
                "screenshot_path": screenshot_path,
                "csv_link": csv_link,
                "audio_url": audio_src,
                "scrape_path": scrape_path,
                "last_error": None
            }
            
        except Exception as e:
            print(f"Navigation Error: {e}")
            return {"last_error": str(e)}
        finally:
            await browser.close()

async def router(state: AgentState):
    print("--- ROUTER: Deciding next step ---")
    
    if state.get('solution'):
        return {"decision": "submitter"}
        
    # Prioritize information gathering
    if state.get('audio_url') and not state.get('transcription'):
        return {"decision": "transcriber"}
        
    if state.get('scrape_path') and not state.get('scraped_content'):
        return {"decision": "scraper"}

    # Then analysis
    if state.get('csv_link') and not state.get('analysis_result'):
        return {"decision": "code_interpreter"}

    print("LLM is making decision...")
        
    # LLM Decision
    context = ""
    if state.get('analysis_result'): context += f"Analysis Result: {state['analysis_result']}\n"
    if state.get('transcription'): context += f"Transcription: {state['transcription']}\n"
    if state.get('scraped_content'): context += f"Scraped Content: {state['scraped_content']}\n"
    if state.get('vision_analysis'): context += f"Vision Analysis: {state['vision_analysis']}\n"

    messages = [
        SystemMessage(content="You are a routing agent. Analyze the state and decide the next step."),
        HumanMessage(content=f"""
        Current URL: {state['url']}
        Page Content: {state['page_content'][:500]}...
        
        Current Knowledge:
        {context}
        
        Tools available:
        - code_interpreter: For CSV analysis, data processing, math, or scraping.
        - vision_analyst: If the question refers to visual elements (charts, layout, "on page 2", etc.) or if text is insufficient.
        - transcriber: For audio.
        - submitter: If you have the answer.
        
        Detected:
        - CSV: {state.get('csv_link')}
        - Audio: {state.get('audio_url')}
        - Scrape: {state.get('scrape_path')}

        You can also scrape the current url that you are in to reveal the DOM of the page.
        
        What should be the next node? Return ONLY the node name: 'code_interpreter', 'vision_analyst', 'transcriber', 'submitter'.
        If we have an 'Analysis Result' that looks like the answer, choose 'submitter'.
        If unsure, default to 'code_interpreter' to try and solve it with code.
        """)
    ]
    
    response = llm.invoke(messages)
    decision = response.content.strip().lower()
    
    if "vision" in decision:
        decision = "vision_analyst"
    elif "transcriber" in decision:
        decision = "transcriber"
    elif "submitter" in decision:
        decision = "submitter"
    else:
        decision = "code_interpreter"
    
    print("LLM Decision: ", decision)
        
    return {"decision": decision}

async def transcriber(state: AgentState):
    print("--- TRANSCRIBER: Processing audio ---")
    audio_url = state['audio_url']
    if not audio_url:
        return {"last_error": "No audio URL found"}
        
    try:
        print(f"Downloading audio from {audio_url}...")
        response = requests.get(audio_url)
        local_filename = f"temp_audio_{int(time.time())}.opus"
        with open(local_filename, "wb") as f:
            f.write(response.content)
            
        print("Loading whisper model...")
        model = whisper.load_model("base")
        
        print("Transcribing audio...")
        result = model.transcribe(local_filename)
        text = result["text"]
        print(f"Transcription: {text}")
        
        if os.path.exists(local_filename):
            os.remove(local_filename)
            
        return {"transcription": text}
    except Exception as e:
        print(f"Transcription Error: {e}")
        return {"last_error": str(e)}

async def scraper(state: AgentState):
    print(f"--- SCRAPER: Scraping {state['scrape_path']} ---")
    url = state['scrape_path']
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            content = await page.evaluate("document.body.innerText")
            print(f"Scraped content length: {len(content)}")
            return {"scraped_content": content}
        except Exception as e:
            print(f"Scraping Error: {e}")
            return {"last_error": str(e)}
        finally:
            await browser.close()

async def vision_analyst(state: AgentState):
    print("--- VISION ANALYST: Analyzing screenshot ---")
    screenshot_path = state['screenshot_path']
    if not screenshot_path or not os.path.exists(screenshot_path):
        return {"last_error": "No screenshot available"}
        
    base64_image = encode_image(screenshot_path)
    
    # Using direct request to AI Pipe (OpenRouter endpoint)
    ai_pipe_key = os.getenv("AI_PIPE_API_KEY", "")
    if not ai_pipe_key:
        return {"last_error": "AI_PIPE_API_KEY not found"}

    url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {ai_pipe_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-5-nano",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": f"Analyze this image. The user needs to solve a quiz. Page text context: {state['page_content'][:200]}..."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.1
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        analysis = result["choices"][0]["message"]["content"]
        print(f"Vision Analysis: {analysis}")
        return {"vision_analysis": analysis}
    except Exception as e:
        print(f"Vision Analysis Error: {e}")
        return {"last_error": str(e)}

async def code_interpreter(state: AgentState):
    print("--- CODE INTERPRETER: Generating/Executing code ---")
    
    # Download CSV if present
    csv_file = None
    if state.get('csv_link'):
        try:
            resp = requests.get(state['csv_link'])
            csv_file = f"temp_data_{int(time.time())}.csv"
            with open(csv_file, "wb") as f:
                f.write(resp.content)
            print(f"Downloaded CSV to {csv_file}")
        except Exception as e:
            print(f"Failed to download CSV: {e}")

    # Construct Prompt
    context = f"Page Content:\n{state['page_content']}\n"
    if state.get('transcription'):
        context += f"Audio Transcription:\n{state['transcription']}\n"
    if state.get('scraped_content'):
        context += f"Scraped Content:\n{state['scraped_content']}\n"
    if state.get('vision_analysis'):
        context += f"Vision Analysis:\n{state['vision_analysis']}\n"
    if state.get('analysis_result'):
        context += f"Previous Analysis Result:\n{state['analysis_result']}\n"
    
    print(state)
    prompt = f"""
    You are a Python expert. Solve the user's question using Python code.
    
    Context:
    {context}
    
    CSV File: {csv_file if csv_file else "None"}
    
    Instructions:
    1. Write a complete Python script to solve the problem.
    2. If a CSV is provided, use pandas to read it: pd.read_csv('{csv_file}')
    3. Print the final answer to stdout.
    4. If the answer is a number, print just the number.
    5. If the answer is text, print just the text.
    
    **IMPORTANT**: If this is a retry attempt (attempt > 1), consider that the CSV might NOT have headers:
    * The first row might be DATA, not column names
    * Try reading with: df = pd.read_csv('{csv_file}', header=None)
    * This uses numeric column indices (0, 1, 2...) instead of named columns
    * Check df.head() to verify if the first row looks like data or headers
    
    Output ONLY the Python code. No markdown.
    """
    print(prompt)
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    result = execute_python_code(code)
    
    if csv_file and os.path.exists(csv_file):
        os.remove(csv_file)
        
    return {"analysis_result": result}

async def submitter(state: AgentState):
    print("--- SUBMITTER: Preparing submission ---")
    
    # Synthesize the answer
    
    # Calculate submission URL
    parsed = urlparse(state['url'])
    submission_url = f"{parsed.scheme}://{parsed.netloc}/submit"

    context = ""
    if state.get('analysis_result'): context += f"Analysis Result: {state['analysis_result']}\n"
    if state.get('transcription'): context += f"Transcription: {state['transcription']}\n"
    if state.get('vision_analysis'): context += f"Vision Analysis: {state['vision_analysis']}\n"
    if state.get('scraped_content'): context += f"Scraped Content: {state['scraped_content']}\n"
    
    prompt = f"""
    You are the final submission agent.
    
    Page Content: {state['page_content']}
    
    Analysis Results:
    {context}
    
    Task:
    1. Identify the answer to the quiz question.
    2. Construct the JSON payload for submission.
    3. Return valid JSON.
    
    Required Fields:
    - email: {state['email']}
    - secret: {state['secret']}
    - url: {state['url']}
    - answer: <THE_ANSWER>
    
    Output Format:
    {{
        "payload": {{ ... }}
    }}
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(content)
        # submission_url = data['submission_url'] # Calculated in python now
        payload = data['payload']
        
        print(f"Submitting to {submission_url} with payload: {payload}")
        resp = requests.post(submission_url, json=payload)
        print(f"Response: {resp.status_code} - {resp.text}")
        
        result = resp.json()
        next_url = result.get('url')
        
        if result.get('correct') or next_url:
            if next_url:
                return {
                    "solution": payload, 
                    "submission_result": result, 
                    "next_url": next_url,
                    "url": next_url,
                    "csv_link": None,
                    "scrape_path": None,
                    "audio_url": None,
                    "transcription": None,
                    "analysis_result": None,
                    "scraped_content": None,
                    "vision_analysis": None,
                    "solution": None,
                    "decision": None,
                    "last_error": None
                }
            else:
                return {"solution": payload, "submission_result": result, "next_url": None}
        else:
            # On failure, clear solution and analysis to allow retry
            return {
                "solution": None,
                "analysis_result": None,
                "last_error": f"Incorrect answer: {result.get('reason')}"
            }
            
    except Exception as e:
        print(f"Submission Error: {e}")
        return {"last_error": str(e)}

# --- Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("navigator", navigator)
workflow.add_node("router", router)
workflow.add_node("transcriber", transcriber)
workflow.add_node("scraper", scraper)
workflow.add_node("vision_analyst", vision_analyst)
workflow.add_node("code_interpreter", code_interpreter)
workflow.add_node("submitter", submitter)

workflow.set_entry_point("navigator")

workflow.add_edge("navigator", "router")

workflow.add_conditional_edges(
    "router",
    lambda x: x['decision'],
    {
        "vision_analyst": "vision_analyst",
        "transcriber": "transcriber",
        "scraper": "scraper",
        "code_interpreter": "code_interpreter",
        "submitter": "submitter"
    }
)

workflow.add_edge("transcriber", "router")
workflow.add_edge("scraper", "router")
workflow.add_edge("vision_analyst", "router")
workflow.add_edge("code_interpreter", "router")

def check_next_url(state):
    if state.get('next_url'):
        return "navigator"
    return END

workflow.add_conditional_edges(
    "submitter",
    check_next_url,
    {
        "navigator": "navigator",
        END: END
    }
)

app_graph = workflow.compile()
