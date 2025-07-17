from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from src.layer_2_agentic_reasoning.llm_manager import LLMManager
import yaml
from src.layer_3_plugins.tools import tools, function_map, format_tools_for_prompt
import json

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# JSON Schema for Agent Response
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
                    "arguments": {"type": "object"}
                },
                "required": ["name", "arguments"]
            }
        },
        "is_final_answer": {"type": "boolean"}
    },
    "required": ["thought", "response", "tool_calls", "is_final_answer"]
}

# Initialize FastAPI app
app = FastAPI()

# Load model configuration
model_path = config["model_path"]
n_ctx = config["n_ctx"]
n_gpu_layers = config["n_gpu_layers"]
n_batch = config["n_batch"]

init_kwargs = {
    "n_ctx": n_ctx,
    "n_gpu_layers": n_gpu_layers,
    "n_batch": n_batch,
}

completion_kwargs = {
    "temperature": config["temperature"],
    "max_tokens": config["max_tokens"],
}

# Store config stop tokens separately
config_stop_tokens = config.get("stop", [])

manager = LLMManager(model_path, **init_kwargs)

def execute_tool_calls(tool_calls: list) -> list:
    """Execute tool calls and return results"""
    results = []
    for tool_call in tool_calls:
        try:
            # Extract tool info from different formats
            if isinstance(tool_call, dict):
                if "function" in tool_call:
                    # OpenAI format: {"type": "function", "function": {"name": "...", "arguments": "..."}}
                    tool_name = tool_call["function"]["name"]
                    tool_args_str = tool_call["function"]["arguments"]
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                elif "name" in tool_call:
                    # Simple format: {"name": "...", "arguments": {...}}
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("arguments", {})
                else:
                    continue
            else:
                continue
            
            if tool_name in function_map:
                print(f"🔧 Executing {tool_name} with args: {tool_args}")
                result = function_map[tool_name](**tool_args)
                results.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": f"call_{tool_name}_{len(results)}"
                })
            else:
                results.append({
                    "role": "tool",
                    "content": json.dumps({"success": False, "message": f"Unknown tool: {tool_name}", "error": "Tool not found"}),
                    "tool_call_id": f"call_{tool_name}_{len(results)}"
                })
        except Exception as e:
            print(f"❌ Error executing tool call: {e}")
            results.append({
                "role": "tool",
                "content": json.dumps({"success": False, "message": "Tool execution failed", "error": str(e)}),
                "tool_call_id": f"call_error_{len(results)}"
            })
    
    return results

def parse_agent_response(response_content: str, response_tool_calls: list) -> dict:
    """Parse response to extract agent payload from BaseModel JSON"""
    
    # Default structure
    agent_payload = {
        "thought": "",
        "response": "",
        "tool_calls": [],
        "is_final_answer": True
    }
    
    try:
        # Parse JSON response from BaseModel
        if response_content.strip():
            parsed_response = json.loads(response_content.strip())
            
            # Extract fields from structured response
            agent_payload["thought"] = parsed_response.get("thought", "")
            agent_payload["response"] = parsed_response.get("response", "")
            agent_payload["tool_calls"] = parsed_response.get("tool_calls", [])
            agent_payload["is_final_answer"] = parsed_response.get("is_final_answer", True)
            
            # Hard code simple logic: tool_calls = False, no tool calls = True
            # TODO: remove this hard code logic if we have a better way to determine if the turn is final or not
            if agent_payload["tool_calls"]:
                agent_payload["is_final_answer"] = False
            else:
                agent_payload["is_final_answer"] = True
            
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        agent_payload["response"] = response_content.strip() if response_content.strip() else "I'm sorry, I had trouble processing that request."
        agent_payload["is_final_answer"] = True
    
    return agent_payload

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("prompt", [])
    
    try:
        llm = manager.get()
    except RuntimeError:
        # Auto-load the LLM if not loaded
        try:
            manager.load()
            llm = manager.get()
        except Exception as e:
            return {"error": f"Failed to load LLM: {str(e)}"}, 400
    
    # Add system prompt if not present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {
            "role": "system",
            "content": config["system_prompt"]
        })
    
    response = llm.create_chat_completion(
        messages=messages, 
        tools=tools, 
        tool_choice="auto", 
        stop=config_stop_tokens,
        response_format={"type": "json_object", "schema": AGENT_RESPONSE_SCHEMA},
        **completion_kwargs
    )
    return response

@app.post("/chat-stream")
async def chat_stream(request: Request):
    try:
        llm = manager.get()
    except RuntimeError:
        # Auto-load the LLM if not loaded
        try:
            manager.load()
            llm = manager.get()
        except Exception as e:
            return {"error": f"Failed to load LLM: {str(e)}"}, 400

    data = await request.json()
    messages = data.get("prompt", [])
    
    manager.reset_interrupt()

    async def stream_gen():
        try:
            if not isinstance(messages, list):
                yield "Error: prompt must be a list of messages."
                return

            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    yield "Error: Invalid message format. Expected dict with 'role' and 'content'."
                    return

            # Add system prompt if not present
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {
                    "role": "system",
                    "content": config["system_prompt"]
                })
            
            # Add tool usage instructions
            messages.append({
                "role": "system",
                "content": format_tools_for_prompt(tools)
            })
            
            # Single turn processing
            response = llm.create_chat_completion(
                messages=messages,
                stream=False,
                tools=tools,
                tool_choice="auto",
                stop=config_stop_tokens,
                response_format={"type": "json_object", "schema": AGENT_RESPONSE_SCHEMA},
                **completion_kwargs
            )
            
            # Extract and parse response
            response_content = ""
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'message' in choice:
                    response_content = choice['message'].get('content', '')
                else:
                    response_content = choice.get('text', '')
            
            # Parse agent response
            agent_payload = parse_agent_response(response_content, [])
            
            # Return structured response
            yield json.dumps(agent_payload)

        except Exception as e:
            yield json.dumps({"error": str(e), "is_final_answer": True})

    return StreamingResponse(stream_gen(), media_type="application/json")

@app.post("/load")
async def load_model():
    try:
        manager.load()
        return {"message": "Model loaded successfully"}
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}, 500

@app.post("/unload")
async def unload_model():
    try:
        manager.unload()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        return {"error": f"Failed to unload model: {str(e)}"}, 500

@app.post("/interrupt")
async def interrupt_generation():
    manager.interrupt()
    return {"message": "Generation interrupted"}

@app.get("/status")
async def get_status():
    """Get current server status"""
    return {
        "handler": "enhanced-multi-turn-agent",
        "model_loaded": manager.is_loaded(),
        "tools_count": len(tools),
        "description": "Multi-turn agent with smart conversation flow"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)