LLM Analysis Quiz Solver
========================

This project implements an automated quiz solver using FastAPI, Playwright, and AI Pipe.

Setup
-----
1. Install Dependencies:
   pip install -r requirements.txt
   playwright install

2. Configuration:
   - The AI Pipe token is hardcoded in `aipipe.py`.
   - The API secret is hardcoded in `main.py` (default: `my_super_secret_code_word`).

Running the Server
------------------
Start the FastAPI server:
uvicorn main:app --reload

The endpoint will be available at `http://localhost:8000/quiz`.

Testing
-------
You can test the solver using the provided `solver.py` script directly:
python llm-quiz/solver.py

Or send a POST request to the running server:
curl -X POST "http://localhost:8000/quiz" \
     -H "Content-Type: application/json" \
     -d '{
           "email": "your_email@example.com",
           "secret": "my_super_secret_code_word",
           "url": "https://tds-llm-analysis.s-anand.net/demo"
         }'

Project Structure
-----------------
- `main.py`: FastAPI application handling incoming quiz requests.
- `solver.py`: Core logic for visiting quiz pages, extracting questions, and submitting answers.
- `aipipe.py`: Client for the AI Pipe LLM API.
- `requirements.txt`: Python dependencies.
