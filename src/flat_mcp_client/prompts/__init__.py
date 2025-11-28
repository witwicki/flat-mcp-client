from typing import Type
import pydantic

def generate_strict_json_schema(model: Type[pydantic.BaseModel]) -> dict:
    """Helper function to enforce strictness following OpenAI's Structured Outputs guidelines"""
    model.model_config = pydantic.ConfigDict(strict=True, extra="forbid")
    #schema_dictionary = model.model_json_schema()
    #return {"type": "json_schema", "strict": True, "json_schema": schema_dictionary}
    schema = model.model_json_schema()
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


def just_json_schema(structured_output: dict | None) -> dict | None:
    """
    The ollama python API expects structured output as a plain json schema, whereas OpenAI ChatCompletion spec
    expects {"type": "json_schema", "strict": True, "json_schema": schema_dictionary}.  Let's accomodate either
    using this helper function.
    """
    schema = None
    if structured_output:
        if "json_schema" in structured_output.keys():
            schema = structured_output["json_schema"]
            if "schema" in schema.keys():
                schema = schema["schema"]
        else:
            schema = structured_output
    return schema
