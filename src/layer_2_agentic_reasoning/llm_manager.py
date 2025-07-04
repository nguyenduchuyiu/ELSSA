from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama

class LLMManager:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.llm = None
        self._stop_generation = False
        self.kwargs = kwargs

    def load(self):
        if self.llm is None:
            print("🧠 [LLM] Loading into memory...")
            self.llm = Llama(model_path=self.model_path, **self.kwargs)
        else:
            print("✅ [LLM] Already loaded")

    def unload(self):
        if self.llm is not None:
            print("💤 [LLM] Unloading from memory...")
            self.llm = None

    def get(self):
        if self.llm is None:
            raise RuntimeError("LLM not loaded")
        return self.llm

    def interrupt(self):
        print("🛑 [LLM] Interrupt requested")
        self._stop_generation = True

    def should_interrupt(self):
        return self._stop_generation

    def reset_interrupt(self):
        self._stop_generation = False
