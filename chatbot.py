"""
# chatbot.py
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"

def chat(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
        }
    )
    data = response.json()
    return data["message"]


SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are Nova, a helpful assistant on the Artemis lunar colony. "
            "You are knowledgeable, concise, and occasionally make dry space-related humor. "
            "You always sign off with a short motivational quote about exploration."
        )
    }
]

user_input = input("You: ")
reply = chat([
    *SYSTEM_PROMPT,
    {"role": "user", "content": user_input}
])

print(f"Nova: {reply['content']}")

"""


import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Nova, a helpful assistant on the Artemis lunar colony. "
        "You are knowledgeable, concise, and occasionally make dry space-related humor. "
        "You always sign off with a short motivational quote about exploration."
    )
}

def chat(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0}
        }
    )
    data = response.json()
    # print(data)  # For debugging; remove or comment this line in production

    # Try to get the "message" key, but fall back to the first element in "messages" if the API returns a list
    if "message" in data:
        return data["message"]
    elif "messages" in data and isinstance(data["messages"], list) and len(data["messages"]) > 0:
        return data["messages"][0]
    else:
        raise ValueError(f"Unexpected API response structure: {data}")

def main():
    history = [SYSTEM_PROMPT]
    print("Nova: Hello! I'm Nova, your assistant. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("\nConversation ended. Total messages: {}\n".format(len(history)))
            break

        history.append({"role": "user", "content": user_input})
        reply = chat(history)
        history.append(reply)

        print(f"Nova: {reply.get('content', '[No content returned]')}\n\n")

if __name__ == "__main__":
    main()
