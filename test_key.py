from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
key = os.getenv("GROQ_API_KEY")
print("Key starts with:", key[:10] if key else "NOT FOUND")

client = Groq(api_key=key)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("Response:", response.choices[0].message.content)