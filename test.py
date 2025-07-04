import time
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner

prompt = [
    {"role": "System", "content": "You are a helpful assistant named ELSSA."},
    {"role": "User", "content": "I've created you to help me with my tasks."},
    {"role": "Assistant", "content": "I'm ELSSA, your helpful assistant. How can I assist you today?"},
    {"role": "User", "content": "What is the capital of Vietnam?"},
    {"role": "Assistant", "content": "That's an easy one, but I'd be happy to confirm. The capital of Vietnam is Hanoi!"},
    {"role": "User", "content": "Good, I'm going to ask you a few more questions. Tell me a fun fact about Hanoi."},
]

runner = LLMRunner()
runner.launch()
time.sleep(5)
for chunk in runner.chat(prompt):
    print(chunk, end="", flush=True)

runner.stop_server()
