from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from .llm_manager import LLMManager
import yaml
from test import tools, function_map
import json

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI()

# Load model configuration
model_path = config["model_path"]
n_ctx = config["n_ctx"]
n_gpu_layers = config["n_gpu_layers"]
n_batch = config["n_batch"]
# Initialize LLMManager with model configuration
init_kwargs = {
    "n_ctx": n_ctx,
    "n_gpu_layers": n_gpu_layers,
    "n_batch": n_batch,
}

completion_kwargs = {
    "temperature": config["temperature"],
    "max_tokens": config["max_tokens"],
    "stop": config["stop"]
}

manager = LLMManager(model_path, **init_kwargs)

@app.post("/chat")
async def chat(request: Request):
    try:
        llm = manager.get()
    except RuntimeError:
        # Auto-load the LLM if not loaded
        try:
            manager.load()
            llm = manager.get()
        except Exception as e:
            return {"error": f"Failed to load LLM: {str(e)}"}

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

            current_messages = messages.copy()
        
            while True:  # Loop for multi-turn conversation with tools
                tool_calls = {}
                partial_content = ""

                llm_stream = llm.create_chat_completion(
                    messages=current_messages,
                    stream=True,
                    tools=tools,
                    tool_choice="auto",
                    **completion_kwargs
                )
                
                for chunk in llm_stream:
                    if manager.should_interrupt():
                        return
                    
                    if 'choices' not in chunk or not chunk['choices']:
                        continue

                    delta = chunk['choices'][0].get('delta', {})
                    
                    # 🧠 Assistant content
                    if 'content' in delta and delta['content']:
                        partial_content += delta['content']
                        yield delta['content']

                    # 🧠 Tool call handling
                    if 'tool_calls' in delta:
                        for call in delta['tool_calls']:
                            call_id = call.get("id")
                            if call_id not in tool_calls:
                                tool_calls[call_id] = {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }

                            fn = call.get("function", {})
                            tool_calls[call_id]["function"]["name"] += fn.get("name", "")
                            tool_calls[call_id]["function"]["arguments"] += fn.get("arguments", "")

                # Add assistant message to conversation
                if partial_content.strip() or tool_calls:
                    assistant_msg = {
                        "role": "assistant", 
                        "content": partial_content.strip() if partial_content.strip() else None
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = list(tool_calls.values())
                    current_messages.append(assistant_msg)

                # 🛠️ Execute tools if any
                if not tool_calls:
                    break  # No tools to call, end conversation
                
                has_valid_tools = False
                for call_id, call in tool_calls.items():
                    name = call['function']['name']
                    args_str = call['function']['arguments']

                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        yield f"\n[⚠️] JSON decode error in tool arguments: {args_str}"
                        # Add error tool response
                        current_messages.append({
                            "tool_call_id": call_id,
                            "role": "tool",
                            "name": name,
                            "content": json.dumps({"error": "Invalid JSON arguments"})
                        })
                        continue

                    if name in function_map:
                        try:
                            # Yield thông tin khi gọi tool và truyền tham số
                            yield f"\n[🔧 Calling tool `{name}` with args]: {json.dumps(args, ensure_ascii=False)}"
                            
                            result = function_map[name](**args)
                            yield f"\n[✅ Tool `{name}` completed]: {result}"
                            
                            # Add tool response to conversation
                            current_messages.append({
                                "tool_call_id": call_id,
                                "role": "tool",
                                "name": name,
                                "content": json.dumps(result)
                            })
                            has_valid_tools = True
                        except Exception as e:
                            yield f"\n[⚠️] Tool `{name}` error: {str(e)}"
                            current_messages.append({
                                "tool_call_id": call_id,
                                "role": "tool",
                                "name": name,
                                "content": json.dumps({"error": str(e)})
                            })
                    else:
                        yield f"\n[⚠️] No function found for `{name}`"
                        current_messages.append({
                            "tool_call_id": call_id,
                            "role": "tool", 
                            "name": name,
                            "content": json.dumps({"error": f"Function {name} not found"})
                        })

                # If we executed any tools, continue conversation for assistant to respond
                if has_valid_tools:
                    yield f"\n\n"  # Add spacing before next response
                    continue
                else:
                    break

        except Exception as e:
            print(f"❌ Error in stream: {e}")
            yield f"\nError: {str(e)}"

    return StreamingResponse(stream_gen(), media_type="text/plain")

@app.post("/load")
def load_llm():
    manager.load()
    return {"status": "loaded"}

@app.post("/unload")
def unload_llm():
    manager.unload()
    return {"status": "unloaded"}

@app.post("/interrupt")
def interrupt_llm():
    manager.interrupt()
    return {"status": "interrupted"}

@app.get("/")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)