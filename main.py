from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
import asyncio
from agent_chain import solve_quiz

app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# Read secret from environment variable
QUIZ_SECRET = os.getenv("QUIZ_SECRET", "default_insecure_secret")
if QUIZ_SECRET == "default_insecure_secret":
    print("WARNING: QUIZ_SECRET not set, using default insecure secret.")

async def run_agent_task(url: str, email: str, secret: str):
    print(f"Starting agent for {url}")
    try:
        # Running sync function in threadpool
        await asyncio.to_thread(solve_quiz, url, email, secret)
        print("Agent execution completed.")
    except Exception as e:
        print(f"Agent execution failed: {e}")

@app.post("/quiz")
async def quiz_endpoint(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    background_tasks.add_task(run_agent_task, request.url, request.email, request.secret)
    return {"message": "Quiz task received", "status": "processing"}

@app.get("/")
def read_root():
    return {"message": "LLM Quiz Solver is running"}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
