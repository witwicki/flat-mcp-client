import pkgutil
from typing import Literal

# EXPOSE TOOL-DEFINITION NAMES AUTOMATICALLY
tooldef_names = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingToolDefinitionNames = Literal[tuple(tooldef_names)]
