import pydantic

def generate_strict_json_schema(model: type[pydantic.BaseModel]) -> dict[str, object]:
    """Helper function to enforce strictness following OpenAI's Structured Outputs guidelines"""
    model.model_config = pydantic.ConfigDict(strict=True, extra="forbid")
    #schema_dictionary = model.model_json_schema()
    #return {"type": "json_schema", "strict": True, "json_schema": schema_dictionary}
    schema: dict[str, object] = model.model_json_schema()
    del schema["title"]
    schema["strict"] = True
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": schema,
            "strict": True
        }
    }


def just_json_schema(structured_output: dict[str, object] | None) -> dict[str, object] | None:
    """
    The ollama python API expects structured output as a plain json schema, whereas OpenAI ChatCompletion spec
    expects {"type": "json_schema", "strict": True, "json_schema": schema_dictionary}.  Let's accomodate either
    using this helper function.
    """
    if not structured_output:
        return None
    json_schema_obj: object | None = structured_output.get("json_schema")
    if isinstance(json_schema_obj, dict):
        json_schema_dict: dict[str, object] = json_schema_obj  # pyright: ignore[reportUnknownVariableType]
        nested_schema_obj: object | None = json_schema_dict.get("schema")
        if isinstance(nested_schema_obj, dict):
            nested_schema_dict: dict[str, object] = nested_schema_obj  # pyright: ignore[reportUnknownVariableType]
            return nested_schema_dict
        return json_schema_dict
    return structured_output
