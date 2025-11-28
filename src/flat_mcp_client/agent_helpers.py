from typing import Literal, Optional, Any
from collections.abc import Mapping
import string
import random
from dataclasses import dataclass

# USEFUL STRING LITERALS
ModelProvider = Literal["ollama", "vllm", "llama.cpp"]
TerminationCondition = Literal[
    "inference_call_completed",
    "nonempty_response_content",
    "no_further_tool_calls",
    "self_determined_termination"
]


# USEFUL DATACLASS
@dataclass
class ServedLLM:
    model_provider: ModelProvider = "ollama"
    model_endpoint: Optional[str] = None
    model_name: Optional[str] = None
    model_path: Optional[str] = None # option to specify path to local file in the case of llama.cpp


# HELPER FUNCTIONS

def generate_random_id():
    characters = string.ascii_letters + string.digits
    result = ''.join(random.choice(characters) for _ in range(9))
    return result


def is_substring_ignoring_case_and_special_characters(term: str, string: str) -> bool:
    from . import debug
    debug(f"Looking for substring [{term}] in string [{string}]...")
    cleaned_term = ''.join(c.lower() for c in term if c.isalnum())
    cleaned_string = ''.join(c.lower() for c in string if c.isalnum())
    return (cleaned_term in cleaned_string)


def deep_merge(d1, d2):
    """
    Performs a deep merge (like a deep | operator) of two dictionaries,
    returning a *new* dictionary without modifying the originals.
    Values from d2 overwrite values from d1 where keys overlap.
    """
    # Start with a shallow copy of d1 to ensure we don't modify the original
    merged = d1.copy()

    for key, value in d2.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            # If both values at this key are dictionaries, recurse
            merged[key] = deep_merge(merged[key], value)
        else:
            # Otherwise, overwrite the value from d2
            merged[key] = value

    return merged


def get_deep_value(dictionary: dict, *keys) -> Any:
    """
    Safely retrieves a value from an arbitrarily deep dictionary.

    Args:
        dictionary (dict): The dictionary to search.
        *keys: Variable number of arguments representing the sequence of nested keys.

    Returns:
        The value at the specified depth, or None if not found.
    """
    current_level = dictionary
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return None
    # If we made it through all keys, return the final value
    return current_level


def set_deep_value(dictionary: dict, value: Any, *keys) -> None:
    """
    Sets a value in a dictionary at an arbitrary, variable depth.

    Creates intermediate dictionaries if they do not already exist.

    Args:
        dictionary (dict): The target dictionary.
        value: The value to set at the final key location.
        *keys: Variable number of keys defining the path to the location.
    """
    current_level = dictionary
    # Iterate through all keys *except* the very last one
    for key in keys[:-1]:
        # Use setdefault to safely create an empty dictionary if 'key' is missing.
        # This guarantees 'current_level[key]' will be a dictionary that we can step into.
        if not isinstance(current_level, dict):
             raise TypeError(f"Cannot set value: Intermediate path element '{key}' is not a dictionary.")
        current_level = current_level.setdefault(key, {})
    # The last key in the sequence is where we place the actual 'value'
    current_level[keys[-1]] = value
