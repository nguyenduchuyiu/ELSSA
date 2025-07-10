import asyncio
import os
import re
import time
import random
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager
from google import genai
from src.layer_2_agentic_reasoning.system_prompt import system_prompt, turing_test_questions
from test_tts import ELSSAMMSTTS


async def main():
    context_manager = ContextManager()

    await context_manager.start_new_session()
    

    
    prompt = [
        
        # {"role": "user", "content": random.choice(turing_test_questions)},
        {"role": "system", "content": "Please read the mathematical expression as if you are saying it aloud in natural language."},
        {"role": "system", "content": "Your answer should start with your thoughts process in tag <thoughts> and end with </thoughts>. After that, start with your answer in tag <answer> and end with </answer>. Example: <thoughts>I need to solve the equation x^2 + x = 10. I can use the quadratic formula to solve it.</thoughts> <answer>x = 2 or x = 5</answer>"},
        {"role": "user", "content": "x^2 + x = 10. => x = ?"}
    ]
    for p in prompt:
        await context_manager.add_message(p["role"], p["content"])

    conversation_context = await context_manager.get_conversation_context()

    # client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # model = "gemini-2.0-flash"
    # response = client.models.generate_content_stream(
    #     model=model,
    #     contents="Tell me a story about a cat.",
    # )
    # for chunk in response:
    #     # print(chunk.text, end="", flush=True)
    #     print(f"Chunk: {chunk.text}. Length: {len(chunk.text)}\n")
        
    runner = LLMRunner()
    runner.launch()
    tts = ELSSAMMSTTS()
    time.sleep(5)
    print(f"User: {conversation_context[1]['content']}")
    try:
        for chunk in runner.chat(conversation_context):
            print(chunk, end="\n", flush=True)
            # tts.say(chunk)
            # print(f"Chunk: {chunk}. Length: {len(chunk)}\n")
    except Exception as e:
        print(e)
        runner.stop_server()
    print("\n")
    await context_manager.end_current_session()
    runner.stop_server()
    
if __name__ == "__main__":
    asyncio.run(main())