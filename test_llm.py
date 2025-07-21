import asyncio
import os
import re
import time
import random
import yaml
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager
from google import genai
from src.layer_2_agentic_reasoning.system_prompt import system_prompt, turing_test_questions
from src.layer_3_plugins.tools import tools

# Load config to get max_context_length
config = yaml.safe_load(open("config.yaml", "r"))

async def main():
    # Use config values for proper testing
    context_manager = ContextManager(
        context_dir="data/context",
        max_context_length=config["max_context_length"]
    )
    
    # Start a new session
    session_id = await context_manager.start_new_session()
    print(f"Started session: {session_id}")
    
    runner = LLMRunner()
    runner.start_server()
    runner.load_model()

    try:
        # Test multiple conversations to verify context management
        test_messages = [
            "Hello, how are you?",
            "What's your name?", 
            "What tools do you have?",
            "Can you search for a song on YouTube?",
            "Tell me a joke."
        ]
        
        for i, user_input in enumerate(test_messages, 1):
            print(f"\n🔄 Turn {i}: {user_input}")
            
            # Add user message
            await context_manager.add_message("user", user_input)
            
            # Get conversation context (should not contain system prompts)
            conversation_context = await context_manager.get_conversation_context()
            print(f"📝 Context length: {len(conversation_context)} messages")
            
            # Use the corrected flow
            for chunk in runner.chat_stream(conversation_context):
                conversation_history = chunk.get("conversation_history", [])
                
                chunk_type = chunk.get("type")
                
                if chunk_type == "response":
                    response_text = chunk.get("message", "")
                    print(f"🤖 {response_text}")
                
                elif chunk_type == "tool_start":
                    tool_calls = chunk.get("tool_calls", [])
                    print(f"🔧 Tool calls: {tool_calls}")
                
                elif chunk_type == "final":
                    # Save new messages to context manager
                    current_context = await context_manager.get_conversation_context()
                    current_length = len(current_context)
                    
                    for j, msg in enumerate(conversation_history[current_length:], current_length):
                        if not (msg["role"] == "user" and msg["content"] == user_input):
                            await context_manager.add_message(msg["role"], msg["content"])
                    break
    except Exception as e:
        print(e)
        runner.stop_server()
    
    print(f"\n📊 Final context length: {len(await context_manager.get_conversation_context())} messages")
    await context_manager.end_current_session()
    runner.stop_server()
    
if __name__ == "__main__":
    asyncio.run(main())
    import gc
    gc.collect()