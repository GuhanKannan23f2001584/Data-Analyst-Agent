import asyncio
from agent_chain import agent
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    print("Starting Custom Task Verification with LangChain Loop...")
    
    # Local Test Server URL
    url = "http://localhost:8002/task/scrape"
    email = "test@example.com"
    secret = "test_secret"
    
    print(f"Invoking agent with URL: {url}")
    
    try:
        await agent.run(url, email, secret)
        print("\nVerification completed.")
                    
    except Exception as e:
        print(f"\nVerification failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
