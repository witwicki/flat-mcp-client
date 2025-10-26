# Tool Definitions

## How to

1. Define a class that inherits from `Toolbox`, including all necessary methods
2. Decorate the methods that serve as tool entry points with `@implements_tool`
3. Now you can use your newly defined toolbox by, e.g., naming it an argument to `--tool` on the CLI

## Automatic (or Manual) Tool Schema

JSON schema need not be defined explicitly.  By default, they get derived from the meta-data, including docstrings!
However, you may choose to define custom schema for your tools by storing them in a list `specific_tool_definitions` outside of the class.

## Benefits of this implementation

- Simplicity of creation (with automatic registry of tool functions and compilation of tool definitions)
- Toolboxes can inherit from one another
- Tools may be stateful (via instance variables and tool implementations as instance methods)!
