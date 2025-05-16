
import config
from collections import defaultdict, deque
from typing import List, Dict

# Simple in-memory storage for dialogue history per user
# Key: user_id, Value: deque of dialogue strings
user_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.MAX_HISTORY_LENGTH)) # Store slightly more than needed for prompt

def get_history(user_id: str) -> List[str]:
    """Retrieves the dialogue history for a user."""
    return list(user_histories[user_id])

def add_to_history(user_id: str, user_input: str, llm_response: str):
    """Adds a user input and NPC response to the user's history."""
    history = user_histories[user_id]

    history.append(f"User: {user_input}")
    history.append(f"LLM Respponce: {llm_response}")