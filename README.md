# SWE-580

## [HW 1 - RAG Q&A System](https://github.com/senaoz/SWE-580/tree/master/hw_1)

## Chatbots - ICP

- Basic chatbot: `python chatbot.py`
- Tool-using chatbot (math/time/status): `python chatbot_tools.py`

## Context compaction - ICP

Both chatbots estimate history size with a rough heuristic (~1 token per 4 characters) and compact older history into a summary once it grows past a threshold.

- `chatbot.py`: tune `TOKEN_THRESHOLD` and `KEEP_RECENT`
- `chatbot_tools.py`: defaults come from `context_management.py` (`token_threshold=2000`, `keep_recent_user_turns=4`)

Suggested manual test (15+ turns):
- Tell the bot: “My name is Jordan”, “I’m allergic to soy”, “My favorite number is 42”
- Chat for a while until you see `[Compacted ... messages into summary]`
- Ask again for the name/allergy/favorite number
