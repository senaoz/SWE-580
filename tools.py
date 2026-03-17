import datetime
import math


def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: The timezone name (only UTC supported in this demo).

    Returns:
        The current date and time as a string.
    """
    if timezone != "UTC":
        return "Error: Only UTC timezone is supported in this demo."

    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"\n[Tool] Getting current time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math expression like '2 + 2' or 'sqrt(144)'.

    Returns:
        The result as a string.
    """
    allowed = {
        "sqrt": math.sqrt,
        "pow": math.pow,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as ex:
        return f"Error: {ex}"


def colony_status() -> str:
    """Get the current Artemis Base colony status report.

    Returns:
        A status report string.
    """
    return (
        "Artemis Base Status Report\n"
        "Crew count: 47 active, 1 on medical leave\n"
        "O2 level: 21.2% (nominal)\n"
        "Power: TMR-3 units at 92% capacity\n"
        "Next Earth comm window: 14:30 LAT\n"
        "Hydroponics harvest: 3 days remaining"
    )


# Registry: maps function names to callables
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "colony_status": colony_status,
}

# Schemas for the Ollama tools API
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name (only UTC supported).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression. Supports basic arithmetic and "
                "functions like sqrt, sin, cos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 + 2' or 'sqrt(144)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "colony_status",
            "description": (
                "Get the current Artemis Base colony status report including crew count, "
                "oxygen levels, power, and schedule."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
