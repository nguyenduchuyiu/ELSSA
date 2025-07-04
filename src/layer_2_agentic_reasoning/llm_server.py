from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from .llm_manager import LLMManager
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = config["model_path"]
n_ctx = config["n_ctx"]
app = FastAPI()
kwargs = {
    "n_ctx": n_ctx,
}
manager = LLMManager(model_path, **kwargs)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: PromptRequest):
    try:
        llm = manager.get()
    except RuntimeError:
        return {"error": "LLM not loaded"}

    manager.reset_interrupt()

    async def stream_gen():
        llm_stream = llm.create_chat_completion(
            messages=[req.prompt],
            stream=True,
            temperature=0.7,
            max_tokens=128,
            stop=["<|user|>", "<|assistant|>", "< Mayer|>", "<3", "<_assistant_|>", "<_user_|>"]
        )
        async for chunk in llm_stream:
            yield chunk

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

@app.post("/interrupt")
async def interrupt_llm():
    manager.interrupt()
    return {"status": "interrupted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)