import asyncio
import os
import re
import time
import random
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager
from google import genai
from src.layer_2_agentic_reasoning.system_prompt import system_prompt, turing_test_questions
from src.layer_3_plugins.tools import tools

async def main():
    context_manager = ContextManager()

    await context_manager.start_new_session()
    

    
    prompt = [
        
        # {"role": "user", "content": random.choice(turing_test_questions)},
        {
            # "role": "user", "content": "Find and open song Photograpghy by Ed Sheeran on YouTube."
            "role": "user", "content": "Which tool do you have?"
         }
    ]
    for p in prompt:
        await context_manager.add_message(p["role"], p["content"])

    conversation_context = await context_manager.get_conversation_context()

    # client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # from google.genai import types
    # model = "gemini-2.0-flash"
    # tool = types.Tool(function_declarations=tools)
    # config = types.GenerateContentConfig(tools=[tool])

    # response = client.models.generate_content_stream(
    #     model=model,
    #     contents="Find and open song Photograpghy by Ed Sheeran on YouTube.",
    #     config=config,
    # )
    # for chunk in response:
    #     # print(chunk.text, end="", flush=True)
    #     print(f"Chunk: {chunk.text}. Length: {len(chunk.text)}\n")
        
    runner = LLMRunner()
    runner.launch()
    time.sleep(5)
    print(f"User: {conversation_context[1]['content']}")
    try:
        # answer = runner.chat(conversation_context)
        # print(answer)
        for chunk in runner.chat_stream(conversation_context[1]['content']):
                # Update conversation history
                conversation_history = chunk.get("conversation_history", [])
                
                chunk_type = chunk.get("type")
                
                if chunk_type == "response":
                    # Speak response immediately
                    response_text = chunk.get("message", "")
                    print(response_text)
                
                elif chunk_type == "tool_start":
                    # Optional: Could speak tool execution notification
                    tool_calls = chunk.get("tool_calls", [])
                    print(f"Tool calls: {tool_calls}")
                    # await self.speak_with_interrupt_support("Let me help you with that.")
                    pass
                
                elif chunk_type == "tool_complete":
                    # Optional: Could speak tool completion
                    pass
                
                elif chunk_type == "error":
                    error_msg = chunk.get("message", "An error occurred")
                    print(f"❌ {error_msg}")                
                elif chunk_type == "final":
                    print("✅ Conversation completed")
                    break
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