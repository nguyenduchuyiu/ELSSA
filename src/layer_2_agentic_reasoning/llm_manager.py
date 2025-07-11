from llama_cpp import Llama
# Import để register handler
from src.layer_3_plugins.streaming_function_calling import chatml_function_calling_streaming

class LLMManager:
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.llm = None
        self._stop_generation = False
        self.kwargs = kwargs

    def load(self):
        if self.llm is None:
            self.llm = Llama(
                model_path=self.model_path, 
                chat_format="chatml-function-calling-streaming",
                **self.kwargs
            )
        else:
            print("✅ [LLM] Already loaded")

    def unload(self):
        if self.llm is not None:
            self.llm = None

    def get(self):
        if self.llm is None:
            raise RuntimeError("LLM not loaded")
        return self.llm

    def interrupt(self):
        self._stop_generation = True

    def should_interrupt(self):
        return self._stop_generation

    def reset_interrupt(self):
        self._stop_generation = False
