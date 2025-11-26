# solver_legacy.py, solver for the quiz task it has been porting into langgraph. 

import asyncio
from playwright.async_api import async_playwright
import requests
import json
import time
import re
import os
import whisper
from dotenv import load_dotenv
from aipipe import chat_completion
import subprocess
import sys

load_dotenv()

def transcribe_audio(audio_url):
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
        
        if os.path.exists(local_filename):
            os.remove(local_filename)
            
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def execute_python_code(code, timeout=30):
    """Execute Python code and return the output"""
    try:
        print(f"Executing generated code:\n{code}\n")
        
        # Create a temporary file for the code
        temp_file = f"temp_code_{int(time.time())}.py"
        with open(temp_file, "w") as f:
            f.write(code)
        
        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if result.returncode == 0:
            print(f"Code execution successful. Output:\n{result.stdout}")
            return result.stdout.strip()
        else:
            print(f"Code execution failed. Error:\n{result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Code execution timed out")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None
    except Exception as e:
        print(f"Error executing code: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None

async def solve_quiz_task(start_url: str, email: str, secret: str):
    print(f"Starting quiz task for {start_url}")
    
    current_url = start_url
    start_time = time.time()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Create a context with a user agent to look less like a bot, though not strictly required
        context = await browser.new_context()
        page = await context.new_page()
        
        while current_url and (time.time() - start_time < 180):
            print(f"Navigating to {current_url}")
            try:
                await page.goto(current_url)
                await page.wait_for_load_state("networkidle")
                
                # Wait for some content to appear. The prompt mentions #result.
                try:
                    await page.wait_for_selector("body", timeout=5000)
                    # specific wait for the demo page structure if known, or just wait a bit more
                    await page.wait_for_timeout(500) 
                except:
                    pass
                
                # Retry loop for the current question
                solved = False
                attempts = 0
                max_attempts = 3
                
                while not solved and attempts < max_attempts and (time.time() - start_time < 180):
                    attempts += 1
                    print(f"Attempt {attempts} for {current_url}")
                    
                    # Extract content
                    content = await page.content()
                    inner_text = await page.evaluate("document.body.innerText")
                    print(f"Page text content: {inner_text[:500]}...") # Debug print
                    
                    # Check for CSV link and fetch it
                    csv_local_file = None
                    csv_href = ""
                    try:
                        # Find link with text "CSV file" or href ending in .csv
                        csv_href = await page.evaluate("""() => {
                            const links = Array.from(document.querySelectorAll('a'));
                            const csvLink = links.find(a => a.innerText.includes('CSV file') || a.href.endsWith('.csv'));
                            return csvLink ? csvLink.href : null;
                        }""")
                        
                        if csv_href:
                            print(f"Found CSV link: {csv_href}")
                            csv_resp = requests.get(csv_href)
                            if csv_resp.status_code == 200:
                                # Save CSV locally for code execution
                                csv_local_file = f"temp_data_{int(time.time())}.csv"
                                with open(csv_local_file, "wb") as f:
                                    f.write(csv_resp.content)
                                print(f"Downloaded CSV to {csv_local_file}")
                            else:
                                print(f"Failed to fetch CSV: {csv_resp.status_code}")
                    except Exception as e:
                        print(f"Error fetching CSV: {e}")

                    # Check for "Scrape <url>" pattern
                    scrape_content = ""
                    try:
                        scrape_match = re.search(r"Scrape\s+([^\s]+)", inner_text)
                        if scrape_match:
                            scrape_path = scrape_match.group(1)
                            # Clean up punctuation if captured
                            scrape_path = scrape_path.rstrip(".,;)")
                            
                            print(f"Found scrape path: {scrape_path}")
                            
                            # Handle URL construction
                            if scrape_path.startswith("http"):
                                scrape_url = scrape_path
                            elif scrape_path.startswith("/"):
                                # Absolute path relative to domain
                                base_url = "/".join(page.url.split("/")[:3])
                                scrape_url = base_url + scrape_path
                            else:
                                # Relative to current URL
                                base_url = page.url.rsplit('/', 1)[0]
                                scrape_url = f"{base_url}/{scrape_path}"
                                
                            print(f"Fetching scrape URL: {scrape_url}")
                            # Use a new page for scraping to avoid messing up the main flow and to handle JS
                            scrape_page = await context.new_page()
                            try:
                                await scrape_page.goto(scrape_url)
                                await scrape_page.wait_for_load_state("networkidle")
                                # Get innerText to avoid HTML tags unless necessary
                                scrape_content = await scrape_page.evaluate("document.body.innerText")
                                print(f"Fetched scrape content ({len(scrape_content)} bytes): {scrape_content[:100]}...")
                            except Exception as e:
                                print(f"Error scraping with Playwright: {e}")
                            finally:
                                await scrape_page.close()

                    except Exception as e:
                        print(f"Error processing scrape instruction: {e}")

                    # Check for Audio tag
                    audio_url = ""
                    transcription = ""
                    try:
                        audio_src = await page.evaluate("""() => {
                            const audio = document.querySelector('audio');
                            return audio ? audio.src : null;
                        }""")
                        if audio_src:
                            print(f"Found audio source: {audio_src}")
                            audio_url = audio_src
                            transcription = transcribe_audio(audio_url)
                            print(f"Transcription: {transcription}")
                    except Exception as e:
                        print(f"Error finding/transcribing audio: {e}")

                    # Step 1: If CSV exists, generate code to analyze it
                    csv_analysis_result = None
                    if csv_local_file:
                        print("CSV detected. Asking LLM to generate analysis code...")
                        
                        # Build the context including audio transcription if available
                        code_context = f"FULL PAGE CONTENT (READ CAREFULLY):\n{inner_text}\n"
                        if transcription:
                            code_context += f"\nAUDIO TRANSCRIPTION (IMPORTANT - May contain additional instructions):\n{transcription}\n"
                        
                        code_gen_prompt = f"""
You are a Python code generator specialized in data analysis. A CSV file is located at: {csv_local_file}

{code_context}

YOUR TASK:
1. Carefully read and understand the EXACT question being asked
2. Identify what specific operation is required (sum, count, filter, average, etc.)
3. Pay attention to any thresholds, cutoffs, or conditions mentioned
4. Generate a complete, working Python script that answers the question precisely

CRITICAL REQUIREMENTS:
- Read the CSV using: df = pd.read_csv('{csv_local_file}')
- Examine the CSV structure first if needed (df.head(), df.columns, df.dtypes)
- **IMPORTANT**: If this is a retry attempt (attempt > 1), consider that the CSV might NOT have headers:
  * The first row might be DATA, not column names
  * Try reading with: df = pd.read_csv('{csv_local_file}', header=None)
  * This uses numeric column indices (0, 1, 2...) instead of named columns
  * Check df.head() to verify if the first row looks like data or headers
- Follow the EXACT instructions from the question (e.g., "count rows where column X > Y")
- Be careful with pandas syntax:
  * Use .values NOT .values() (it's a property, not a method)
  * Use .sum() NOT sum() on Series
  * Use proper indexing: df['column_name'] or df[df['col'] > value]
  * For headerless CSVs: df[0], df[1], df[df[0] > value]
- Handle any necessary data type conversions (e.g., str to int, str to float)
- Remove any NaN or invalid values if appropriate
- Print ONLY the final numeric answer (no text, no explanations)
- If the answer is a count, print an integer
- If the answer is a sum/average, print the number
- Do NOT print the entire series/dataframe

EXAMPLE PATTERNS:
- "Count rows where X > 100": len(df[df['X'] > 100]) or (df['X'] > 100).sum()
- "Sum of column Y where Z < 50": df[df['Z'] < 50]['Y'].sum()
- "Average of column A": df['A'].mean()
- For headerless: "Count rows where first column > 100": len(df[df[0] > 100]) or (df[0] > 100).sum()

CURRENT ATTEMPT NUMBER: {attempts}
{"NOTE: This is a retry - consider reading CSV with header=None if the data looks wrong!" if attempts > 1 else ""}

OUTPUT ONLY THE PYTHON CODE. NO MARKDOWN BLOCKS. NO EXPLANATIONS.
"""
                        
                        code_gen_messages = [{"role": "user", "content": code_gen_prompt}]
                        code_gen_response = chat_completion(code_gen_messages)
                        
                        if code_gen_response and 'choices' in code_gen_response:
                            generated_code = code_gen_response['choices'][0]['message']['content']
                            
                            # Clean up markdown if present
                            if "```python" in generated_code:
                                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
                            elif "```" in generated_code:
                                generated_code = generated_code.split("```")[1].split("```")[0].strip()
                            
                            # Execute the code
                            csv_analysis_result = execute_python_code(generated_code)
                            
                            # Clean up CSV file
                            if os.path.exists(csv_local_file):
                                os.remove(csv_local_file)
                        else:
                            print("Failed to generate code")
                            if os.path.exists(csv_local_file):
                                os.remove(csv_local_file)

                    extra_content = ""
                    if csv_href:
                        extra_content += f"\nCSV CONTENT LINK:\n{csv_href}\n"
                        if csv_analysis_result:
                            extra_content += f"\nCSV ANALYSIS RESULT:\n{csv_analysis_result}\n"
                        else:
                            extra_content += "\n(CSV was found but analysis failed)\n"
                            
                    if scrape_content:
                        extra_content += f"\nSCRAPED CONTENT (Priority for Answer):\n{scrape_content}\n"
                    if audio_url:
                        extra_content += f"\nAUDIO URL:\n{audio_url}\n"
                        if transcription:
                            extra_content += f"\nAUDIO TRANSCRIPTION:\n{transcription}\n"

                    # Construct prompt for LLM
                    prompt = f"""
                    You are an AI agent solving a data analysis quiz.
                    
                    PAGE TEXT:
                    {inner_text}
                    
                    {extra_content}
                    
                    INSTRUCTIONS:
                    1. Identify the question and solve it.
                    2. If "SCRAPED CONTENT" is provided, the answer is likely within it. Ignore HTML tags in the answer unless explicitly asked for.
                    3. If "CSV ANALYSIS RESULT" is provided, that is the answer to the CSV-based question. Use it directly.
                    4. If "AUDIO TRANSCRIPTION" is provided, it may contain additional instructions or the answer itself. Read it carefully and follow any instructions within it.
                    5. Identify the submission URL (where to POST the answer).
                    6. Create the JSON payload to submit.
                    
                    Use the following credentials for the payload:
                    Email: {email}
                    Secret: {secret}
                    Current URL: {current_url}
                    
                    The payload usually looks like:
                    {{
                        "email": "{email}",
                        "secret": "{secret}",
                        "url": "{current_url}",
                        "answer": <YOUR_ANSWER>
                    }}
                    
                    OUTPUT FORMAT:
                    Return a valid JSON object with EXACTLY this structure:
                    {{
                        "submission_url": "https://...",
                        "payload": {{ ... }}
                    }}
                    
                    Do not include any markdown formatting (like ```json). Just the raw JSON.
                    """
                    
                    messages = [{"role": "user", "content": prompt}]
                    print(messages)
                    
                    response = chat_completion(messages)
                    
                    if not response:
                        print("Failed to get response from AI")
                        continue
                        
                    # Extract content based on response structure
                    llm_output = ""
                    if 'choices' in response and len(response['choices']) > 0:
                        llm_output = response['choices'][0]['message']['content']

                    print(f"LLM Output: {llm_output}")
                    
                    # Parse JSON
                    try:
                        # Clean up markdown if present
                        if "```json" in llm_output:
                            llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                        elif "```" in llm_output:
                            llm_output = llm_output.split("```")[1].split("```")[0].strip()
                            
                        data = json.loads(llm_output)
                        submission_url = data.get("submission_url")
                        payload = data.get("payload")
                        
                        if not submission_url or not payload:
                            print("Invalid JSON structure from LLM")
                            continue

                        # Validate answer
                        if not str(payload.get("answer", "")).strip():
                            print("LLM returned empty answer. Retrying...")
                            continue
                            
                        print(f"Submitting to {submission_url} with payload: {payload}")
                        
                        # Submit
                        submit_response = requests.post(submission_url, json=payload)
                        print(f"Submission response: {submit_response.status_code} - {submit_response.text}")
                        
                        if submit_response.status_code == 200:
                            res_json = submit_response.json()
                            if res_json.get("correct"):
                                print("Answer correct!")
                                solved = True
                                next_url = res_json.get("url")
                                if next_url:
                                    current_url = next_url
                                else:
                                    print("Quiz completed!")
                                    current_url = None
                            else:
                                print(f"Answer incorrect: {res_json.get('reason')}")
                                # If we get a new URL even if wrong (as per prompt), take it
                                if res_json.get("url"):
                                    print("Received new URL despite incorrect answer, moving on.")
                                    current_url = res_json.get("url")
                                    solved = True # Treat as "done with this step"
                                else:
                                    # Retry same URL
                                    print("Retrying same URL...")
                        else:
                            print(f"Submission failed with status {submit_response.status_code}")
                            
                    except json.JSONDecodeError:
                        print("Failed to parse JSON from LLM output")
                    except Exception as e:
                        print(f"Error during solving/submission: {e}")
                
                if not solved:
                    print(f"Failed to solve {current_url} after {max_attempts} attempts or time out.")
                    break # Break outer loop if we can't solve it
                    
            except Exception as e:
                print(f"Error processing {current_url}: {e}")
                break
        
        await browser.close()

if __name__ == "__main__":
    # Test run
    # asyncio.run(solve_quiz_task("https://tds-llm-analysis.s-anand.net/demo-audio?email=test%40example.com&id=7017", "test@example.com", "test_secret"))
    asyncio.run(solve_quiz_task("https://tds-llm-analysis.s-anand.net/demo", "test@example.com", "test_secret"))
