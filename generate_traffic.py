import requests
import json
import time

URL = "http://localhost:8000/query"

queries = [
    "What is the attention mechanism?",
    "Who wrote the paper on Transformers?",
    "Explain the difference between RAG and Graph RAG."
]

print(f"Sending {len(queries)} test queries to {URL}...")

for q in queries:
    print(f"\nQuery: {q}")
    try:
        response = requests.post(URL, json={"query": q})
        if response.status_code == 200:
            print("Response:", response.json().get("final_answer", "")[:100] + "...")
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Make sure the backend is running on port 8000!")
        break
    time.sleep(1)

print("\nDone! Check LangSmith Dashboard.")
