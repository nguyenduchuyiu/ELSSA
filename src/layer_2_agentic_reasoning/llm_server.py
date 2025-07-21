from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from .llm_manager import LLMManager
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI()

# Load model configuration
model_path = config["model_path"]
n_ctx = config["n_ctx"]
n_gpu_layers = config["n_gpu_layers"]
# Initialize LLMManager with model configuration
init_kwargs = {
    "n_ctx": n_ctx,
    "n_gpu_layers": n_gpu_layers,
}

completion_kwargs = {
    "temperature": config["temperature"],
    "max_tokens": config["max_tokens"],
    "stop": config["stop"]
}

manager = LLMManager(model_path, **init_kwargs)

@app.post("/chat_stream")
async def chat_stream(request: Request):
    try:
        llm = manager.get()
    except RuntimeError:
        return {"error": "LLM not loaded"}

    # Get JSON data directly from request
    data = await request.json()
    messages = data.get("prompt", [])
    
    manager.reset_interrupt()

    async def stream_gen():
        try:
            # Validate message format
            if not isinstance(messages, list):
                yield f"Error: prompt must be a list of messages."
                return
                
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    yield f"Error: Invalid message format. Expected dict with 'role' and 'content' keys."
                    return
            
            llm_stream = llm.create_chat_completion(
                messages=messages,
                stream=True,
                **completion_kwargs
            )
            for chunk in llm_stream:
                if manager.should_interrupt():
                    break
                # Extract content from chunk
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        yield delta['content']
        except Exception as e:
            print(f"❌ Error in streaming: {e}")
            yield f"Error: {str(e)}"

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