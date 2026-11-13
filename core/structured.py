"""Structured output that survives local models answering in text instead of calling the schema tool."""
import re


def json_from(text, schema):
    """First {...} block in a text reply, validated against the schema; None if there is none."""
    if isinstance(text, list):
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    m = re.search(r"\{.*\}", text if isinstance(text, str) else "", re.S)
    if not m:
        return None
    try:
        return schema.model_validate_json(m.group(0))
    except Exception:
        return None


async def structured(model, schema, messages, attempts: int = 2, on_call=None):
    """Parsed schema object or None. Retries once with a reminder, then parses JSON from the text reply.
    on_call(): called before every model call (for call budgets); return False to stop."""
    msgs = list(messages)
    for _ in range(attempts):
        if on_call is not None and on_call() is False:
            return None
        out = await model.with_structured_output(schema, method="function_calling", include_raw=True).ainvoke(msgs)
        if isinstance(out, dict) and "parsed" in out:
            if out.get("parsed") is not None:
                return out["parsed"]
            parsed = json_from(getattr(out.get("raw"), "content", ""), schema)
            if parsed is not None:
                return parsed
        elif out is not None:          # models/fakes that return the object directly
            return out
        msgs = msgs + [("user", f"Answer only by calling the {schema.__name__} tool with valid arguments.")]
    return None
