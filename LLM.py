from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

a = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

result = a.invoke("Hello, What is the Capital of India?")

print(result)
