import json

import requests

from context_management import compact_history, estimate_tokens
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Nova, a helpful assistant on the Artemis lunar colony. "
        "You have access to tools. Use them when the user asks for "
        "calculations, the current time, or colony status. "
        "Do not make up data — use the tools to get real information."
    ),
}


def chat(messages, *, include_tools: bool = True):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": messages,
            **({"tools": TOOL_SCHEMAS} if include_tools else {}),
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]


def _parse_tool_arguments(raw_args):
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def execute_tool_calls(tool_calls):
    """Execute each tool call and return a list of tool result messages."""
    results = []
    for tc in tool_calls:
        function = tc.get("function", {})
        name = function.get("name")
        args = _parse_tool_arguments(function.get("arguments"))

        if name in TOOL_FUNCTIONS:
            print(f"\n[Tool] Calling {name}({args}) \n")
            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as ex:
                result = f"Error executing tool '{name}': {ex}"
        else:
            result = f"Unknown tool: {name}"

        results.append({"role": "tool", "content": str(result)})
    return results


def main():
    history = [SYSTEM_PROMPT]
    print("Nova: Hello! I can check colony status, do math, and tell the time.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})
        token_count = estimate_tokens(history)
        print(f" [Tokens: ~{token_count}]")
        history = compact_history(
            history,
            summarize_chat=lambda req: chat(req, include_tools=False),
        )

        # Agentic loop: keep going until the model gives a text reply
        while True:
            history = compact_history(
                history,
                summarize_chat=lambda req: chat(req, include_tools=False),
            )
            reply = chat(history)
            history.append(reply)

            if reply.get("tool_calls"):
                tool_results = execute_tool_calls(reply["tool_calls"])
                history.extend(tool_results)
                # Loop back so the model can process the results
            else:
                # No tool calls — this is the final response
                print(f"Nova: {reply.get('content', '')}\n")
                break


if __name__ == "__main__":
    main()
