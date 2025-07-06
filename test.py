import asyncio
import os
import time
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager
from google import genai





async def main():
    context_manager = ContextManager()

    await context_manager.start_new_session()
    prompt = [
        {"role": "system", "content": "You are a helpful assistant named ELSSA. Answer concisely and to the point."},
        {"role": "user", "content": "Explain photon synthesis."},
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
    time.sleep(3)
    try:
        for chunk in runner.chat(conversation_context):
            print(chunk, end="", flush=True)
            # print(f"Chunk: {chunk}. Length: {len(chunk)}\n")
    except Exception as e:
        print(e)
        runner.stop_server()

if __name__ == "__main__":
    asyncio.run(main())