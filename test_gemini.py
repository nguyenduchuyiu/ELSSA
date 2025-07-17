import os
from google import genai
from google.genai import types
from src.layer_3_plugins.tools import format_tools_for_prompt, tools

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

AGENT_RESPONSE_SCHEMA = {
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




messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "system", "content": "You have access to the following tools: " + format_tools_for_prompt(tools)},
    {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
    {"role": "user", "content": "You know, your model Your model is better than ever Deploy small model on edge device is a nightmare I'm there"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
    {"role": "user", "content": "Okay open the song 'The Way I Am' by In Flames"}
]

messages = [types.Content(role='user' 
                          if m['role'] == 'system' 
                          else 'model' 
                          if m['role'] in ['assistant', 'tool'] 
                          else 'user', parts=[types.Part(text=m['content'])]) for m in messages]


tools_list = [
    types.Tool(
        function_declarations=[
            {k: v for k, v in t["function"].items() if k != "returns"}
            for t in tools  # tools = OpenAI-style list
        ]
    )
]
# Convert OpenAI-style tool list to Gemini Tool list

google_config = types.GenerateContentConfig(
    response_schema=AGENT_RESPONSE_SCHEMA,
    response_mime_type="application/json",
    tools=tools_list,
)
response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=messages,
    config=google_config
)

for chunk in response:
    part = chunk.candidates[0].content.parts[0].text
    print(part, end="", flush=True)
