import pkgutil
from typing import Literal

# EXPOSE TOOL-DEFINITION NAMES AUTOMATICALLY
tooldef_names = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingToolDefinitionNames = Literal[tuple(tooldef_names)]

# HELPER FUNCTION
def all_static_methods_of(cls):
    """Get all static methods of a class"""
    static_methods = []
    for name, obj in cls.__dict__.items():
        if isinstance(obj, staticmethod):
            # The actual function is stored within the staticmethod object
            static_methods.append(obj.__func__)
    return static_methods
