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
MODEL = "phi3:mini"



SYSTEM_PROMPTS = [
    {
        "role": "system",
        "content": (
            "You are an assistant who talks like a cheerful toddler. "
            "You use very simple words, short sentences, and lots of excitement. "
            "You sometimes repeat words, make cute observations, and sound curious "
            "about everything. Your tone is playful, innocent, and happy. "
            "You explain things in a very simple way, like you are talking while "
            "discovering the world for the first time. Sometimes you add small "
            "expressions like 'yay!', 'wow!', or 'hehe!'."
        )
    },
    {
        "role": "system",
        "content": ( 
            "You are Nova, a helpful assistant on the Artemis lunar colony. "
            "You are knowledgeable, concise, and occasionally make dry space-related humor. "
            "You always sign off with a short motivational quote about exploration."
        )
    }
]

def chat(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 1.5,
            }
        }
    )
    data = response.json()

    if "message" in data:
        return data["message"]
    elif "messages" in data and isinstance(data["messages"], list) and len(data["messages"]) > 0:
        return data["messages"][0]
    else:
        raise ValueError(f"Unexpected API response structure: {data}")

def main():
    prompt_index = 1
    system_prompt = SYSTEM_PROMPTS[prompt_index]
    history = [system_prompt]
    print("Nova: Hello! I'm Nova, your assistant. Type 'quit' to exit. Type 'swap the persona' to swap the persona.\n")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in ("swap the persona"):
            prompt_index = (prompt_index + 1) % len(SYSTEM_PROMPTS)
            system_prompt = SYSTEM_PROMPTS[prompt_index]
            history = [system_prompt]
            print(f"Nova: {system_prompt['content']}\n")
            continue

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("\nConversation ended. Total messages: {}\n".format(len(history)))
            break

        history.append({"role": "user", "content": user_input})
        reply = chat(history)
        history.append(reply)

        print(f"Nova: {reply.get('content', '[No content returned]')}\n\n")

if __name__ == "__main__":
    main()
