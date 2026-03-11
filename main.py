from openai import OpenAI
from dotenv import load_dotenv
import os

"""
WEEK 1 

------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[
        {
            "role": "user",
            "content": "What is the capital city of Turkiye?"
        }
    ]
)

print(response.choices[0].message.content)

"""



