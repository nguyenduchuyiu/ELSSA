GEMINI_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "response": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {}
                },
                "required": ["name", "arguments"]
            }
        },
        "is_final_answer": {"type": "boolean"}
    },
    "required": ["thought", "response", "tool_calls", "is_final_answer"]
}

OPENAI_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "response": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"}
                },
                "required": ["name", "arguments"]
            }
        },
        "is_final_answer": {"type": "boolean"}
    },
    "required": ["thought", "response", "tool_calls", "is_final_answer"]
}