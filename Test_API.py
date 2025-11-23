#THIS IS JUST A TEST FILE THAT CHECKS IF API IS CALL IS WORKING AND MODEL IS GENERATING RESPONSE.
#THIS FILE IS MAINLY USED FOR DEBUGGING.

from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

#Check for any errors in API Key
if not api_key:
    raise RuntimeError("No API key found. Make sure it's in your .env file.")

client = OpenAI(api_key=api_key)

print("Sending test request...")

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Hello! Can you confirm the API is working? Just say YES OR NO"}
    ]
)

print("\nAPI Response:")
print(response.choices[0].message.content)

