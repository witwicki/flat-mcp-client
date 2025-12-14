from collections.abc import Mapping
import string
import random


# HELPER FUNCTIONS

def generate_random_id() -> str:
    characters = string.ascii_letters + string.digits
    result = ''.join(random.choice(characters) for _ in range(9))
    return result


def is_substring_ignoring_case_and_special_characters(term: str, string: str) -> bool:
    from . import debug
    debug(f"Looking for substring [{term}] in string [{string}]...")
    cleaned_term = ''.join(c.lower() for c in term if c.isalnum())
    cleaned_string = ''.join(c.lower() for c in string if c.isalnum())
    return (cleaned_term in cleaned_string)


def deep_merge(d1: Mapping[str, object], d2: Mapping[str, object]) -> dict[str, object]:
    """
    Performs a deep merge (like a deep | operator) of two dictionaries,
    returning a *new* dictionary without modifying the originals.
    Values from d2 overwrite values from d1 where keys overlap.
    """
    # Start with a shallow copy of d1 to ensure we don't modify the original
    merged: dict[str, object] = dict(d1)

    for key, value in d2.items():
        existing_value = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing_value, Mapping):
            existing_mapping: Mapping[str, object] = existing_value  # pyright: ignore[reportUnknownVariableType]
            value_mapping: Mapping[str, object] = value  # pyright: ignore[reportUnknownVariableType]
            merged[key] = deep_merge(existing_mapping, value_mapping)
        else:
            # Otherwise, overwrite the value from d2
            merged[key] = value

    return merged


def get_deep_value(dictionary: Mapping[str, object], *keys: str) -> object | None:
    """
    Safely retrieves a value from an arbitrarily deep dictionary.

    Args:
        dictionary (dict): The dictionary to search.
        *keys: Variable number of arguments representing the sequence of nested keys.

    Returns:
        The value at the specified depth, or None if not found.
    """
    current_level: object | Mapping[str, object] = dictionary
    for key in keys:
        if not isinstance(current_level, Mapping):
            return None
        current_mapping = dict(current_level)  # pyright: ignore[reportUnknownArgumentType]
        next_level: object | None = current_mapping.get(key)
        if next_level is None:
            return None
        current_level = next_level
    return current_level


def set_deep_value(dictionary: dict[str, object], value: object, *keys: str) -> None:
    """
    Sets a value in a dictionary at an arbitrary, variable depth.

    Creates intermediate dictionaries if they do not already exist.

    Args:
        dictionary (dict): The target dictionary.
        value: The value to set at the final key location.
        *keys: Variable number of keys defining the path to the location.
    """
    current_level: dict[str, object] = dictionary
    # Iterate through all keys *except* the very last one
    for key in keys[:-1]:
        next_level = current_level.get(key)
        if isinstance(next_level, dict):
            current_level = next_level  # pyright: ignore[reportUnknownVariableType]
        elif next_level is None:
            new_level: dict[str, object] = {}
            current_level[key] = new_level
            current_level = new_level
        else:
            raise TypeError(f"Cannot set value: Intermediate path element '{key}' is not a dictionary.")
    # The last key in the sequence is where we place the actual 'value'
    current_level[keys[-1]] = value
