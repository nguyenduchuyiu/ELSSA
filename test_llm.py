import asyncio
import os
import re
import time
import random
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager
from google import genai
from src.layer_2_agentic_reasoning.system_prompt import system_prompt, turing_test_questions


async def main():
    context_manager = ContextManager()

    await context_manager.start_new_session()
    

    
    prompt = [
        
        # {"role": "user", "content": random.choice(turing_test_questions)},
        {"role": "user", "content": "Find and open song Baby by Justin Bieber on YouTube"}
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
    time.sleep(5)
    print(f"User: {conversation_context[1]['content']}")
    try:
        answer = ""
        for chunk in runner.chat(conversation_context):
            # remove part in tool call {}, []
            # chunk = re.sub(r'\{.*?\}|\[.*?\]', '', chunk)
            answer += chunk
            print(chunk, end="", flush=True)
            # tts.say(chunk)
            # print(f"Chunk: {chunk}. Length: {len(chunk)}\n")
        # conversation_context = await context_manager.add_message("assistant", answer)
        # conversation_context = await context_manager.add_message("user", "search for a video about baby on youtube")
        # conversation_context = await context_manager.get_conversation_context()
        # print(f"User: {conversation_context[-1]['content']}")
        # for chunk in runner.chat(conversation_context):
        #     print(chunk, end="", flush=True)
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
    import gc
    gc.collect()