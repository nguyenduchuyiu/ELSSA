import threading
import requests
import subprocess
import time
import json
import yaml
import os

from src.layer_3_plugins.tools import tools
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
from google import genai
from google.genai import types
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))



class LLMRunner:
    def __init__(self, module_path="src.layer_2_agentic_reasoning.llm_server", port=8000):
        self.module_path = module_path
        self.port = port
        self.proc = None
        self.llm_loaded = False
        self.conversation_history = []

    def _wait_until_server_ready(self, timeout=20):
        url = f"http://localhost:{self.port}"
        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get(url)
                print("✅ Server is ready.")
                return True
            except Exception:
                print("⏳ Waiting for server...", flush=True)
                time.sleep(1)
        raise TimeoutError("❌ Server not responding after timeout")

    def start_server(self):
        print("🚀 Starting LLM server...")
        self.proc = subprocess.Popen(
            ["python3", "-m", self.module_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_server_ready()

    def stop_server(self):
        if self.proc:
            print("🛑 Shutting down server...")
            self.proc.terminate()
            self.proc.wait()
            print("✅ Server stopped.")
            self.proc = None

    def load_model(self):
        r = requests.post(f"http://localhost:{self.port}/load")
        if r.status_code == 200:
            self.llm_loaded = True
            print("✅ LLM loaded")
        else:
            print(f"❌ LLM load failed: {r.json()['message']}")

    def unload_model(self):
        r = requests.post(f"http://localhost:{self.port}/unload")
        if r.status_code == 200:
            self.llm_loaded = False
            print("✅ LLM unloaded")
        else:
            print("❌ LLM unload failed")

    def chat(self, messages: list[dict]):
        """Single turn chat without streaming"""
        url = f"http://localhost:{self.port}/chat"
        data = {"prompt": messages}
        
        r = requests.post(url, json=data)
        r.raise_for_status()
        return r.json()

    def interrupt(self):
        print("✋ Sending interrupt signal...")
        r = requests.post(f"http://localhost:{self.port}/interrupt")
        print("Interrupt status:", r.json())

    def launch(self):
        def run():
            self.start_server()
            self._wait_until_server_ready()
            self.load_model()
        threading.Thread(target=run, daemon=True).start()
        
    def chat_stream(self, user_message: str, max_turns: int = 10):
        """Handle multi-turn conversation with agent - yields responses in real-time"""
        
        if self.proc is None:
            print("❌ Server not running")
            yield {"type": "error", "message": "Server not running", "conversation_history": []}
            return
        
        if not self.llm_loaded:
            print("❌ LLM not loaded. Loading...")
            self.load_model()
            print("✅ LLM loaded")
        
        # Add user message to history
        # print(f"🔍 DEBUG: User message type: {type(user_message)}, content: {repr(user_message)}")
        self.conversation_history.append({
            "role": "user", 
            "content": str(user_message)
        })
        
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            print(f"🔄 Agent turn {turn_count}")
            
            # Call server for single turn
            response = self._call_single_turn(self.conversation_history)
            
            if response.get("error"):
                error_msg = f"❌ Error: {str(response['error'])}"
                # print(error_msg)
                yield {"type": "error", "message": error_msg, "conversation_history": self.conversation_history.copy()}
                break
                
            # Get agent response
            agent_response = response.get("response", "")
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": str(agent_response)  # Ensure it's string
            })
            
            # Yield response immediately for TTS
            if agent_response:
                # print(f"🔍 DEBUG: Agent response type: {type(agent_response)}, content: {repr(agent_response)}")
                # print(str(agent_response))
                yield {
                    "type": "response", 
                    "message": str(agent_response),
                    "conversation_history": self.conversation_history.copy()
                }
            
            # Execute tools if any
            tool_calls = response.get("tool_calls", [])
            if tool_calls:
                # Yield tool execution notification
                yield {
                    "type": "tool_start",
                    "tool_calls": tool_calls,
                    "conversation_history": self.conversation_history.copy()
                }
                
                tool_results = self._execute_tools(tool_calls)
                self.conversation_history.extend(tool_results)
                
                # Show tool results
                for result in tool_results:
                    try:
                        result_data = json.loads(result['content'])
                        if result_data.get("success"):
                            # print(f"✅ {str(result_data.get('message', 'Tool executed successfully'))}")
                            pass
                        else:
                            # print(f"❌ {str(result_data.get('message', 'Tool execution failed'))}")
                            pass
                            if result_data.get("error"):
                                # print(f"   Error: {str(result_data['error'])}")
                                pass
                    except json.JSONDecodeError:
                        # print(f"📋 Tool result: {str(result['content'])}")
                        pass
                    except Exception as e:
                        # print(f"⚠️ Error parsing tool result: {str(e)}")
                        pass
                
                # Yield tool completion
                yield {
                    "type": "tool_complete",
                    "tool_results": tool_results,
                    "conversation_history": self.conversation_history.copy()
                }
            
            # Check if conversation is complete
            if response.get("is_final_answer", True):
                # print("✅ Conversation completed")
                pass
                yield {
                    "type": "final", 
                    "message": "Conversation completed",
                    "conversation_history": self.conversation_history.copy()
                }
                break
        
        # If max turns reached
        if turn_count >= max_turns:
            yield {
                "type": "final",
                "message": "Maximum turns reached",
                "conversation_history": self.conversation_history.copy()
            }



    def _call_single_turn(self, messages: list) -> dict:
        """Call server for single turn"""
        try:
            url = f"http://localhost:{self.port}/chat-stream"
            data = {"prompt": messages}
            
            if config.get("super_mode", False):
                print("🔍 DEBUG: Super mode")
                # Convert OpenAI-style tool list to Gemini Tool list
                AGENT_RESPONSE_SCHEMA = {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "response": {"type": "string"},
                        "tool_calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "arguments": {}
                                },
                                "required": ["name", "arguments"]
                            }
                        },
                        "is_final_answer": {"type": "boolean"}
                    },
                    "required": ["thought", "response", "tool_calls", "is_final_answer"]
                }

                tools_list = [
                    types.Tool(
                        function_declarations=[
                            {k: v for k, v in t["function"].items() if k != "returns"}
                            for t in tools  # tools = OpenAI-style list
                        ]
                    )
                ]
                messages = [types.Content(role='user' 
                                        if m['role'] == 'system' 
                                        else 'model' 
                                        if m['role'] in ['assistant', 'tool'] 
                                        else 'user', parts=[types.Part(text=m['content'])]) for m in messages]
                # Pass into config
                google_config = types.GenerateContentConfig(
                    response_schema=AGENT_RESPONSE_SCHEMA,
                    response_mime_type="application/json",
                    tools=tools_list,
                )
                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=messages,
                    config=google_config
                )
                full_response = ""
                for chunk in response:
                    part = chunk.candidates[0].content.parts[0].text
                    full_response += part
                print(f"🔍 DEBUG: Full response: {full_response}")
                return json.loads(full_response)
            else:
                response = requests.post(url, json=data, timeout=120)
                full_response = ""
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        # Ensure chunk is string
                        chunk_str = str(chunk) if not isinstance(chunk, str) else chunk
                        full_response += chunk_str
                return json.loads(full_response)
            
        except Exception as e:
            # print(f"🔍 DEBUG: Exception in _call_single_turn: {e}")
            return {"error": str(e), "is_final_answer": True}

    def _execute_tools(self, tool_calls: list) -> list:
        """Execute tool calls and return results"""
        from src.layer_3_plugins.tools import function_map
        
        results = []
        for tool_call in tool_calls:
            try:
                # print(f"🔍 DEBUG: Processing tool_call: {type(tool_call)} - {repr(tool_call)}")
                
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                # Fix: Parse tool_args if it's a string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                
                # print(f"🔍 DEBUG: tool_name: {type(tool_name)} - {repr(tool_name)}")
                # print(f"🔍 DEBUG: tool_args: {type(tool_args)} - {repr(tool_args)}")
                
                if tool_name in function_map:
                    # print(f"🔧 Executing {str(tool_name)} with args: {str(tool_args)}")
                    result = function_map[tool_name](**tool_args)
                    results.append({
                        "role": "tool",
                        "content": f"Tool {str(tool_name)} executed successfully with result: {json.dumps(result)}",
                        "tool_call_id": f"call_{str(tool_name)}_{len(results)}"
                    })
                else:
                    results.append({
                        "role": "tool",
                        "content": json.dumps({"success": False, "message": f"Unknown tool: {str(tool_name)}", "error": "Tool not found"}),
                        "tool_call_id": f"call_{str(tool_name)}_{len(results)}"
                    })
            except Exception as e:
                # print(f"🔍 DEBUG: Exception in _execute_tools: {e}")
                results.append({
                    "role": "tool",
                    "content": json.dumps({"success": False, "message": "Tool execution failed", "error": str(e)}),
                    "tool_call_id": f"call_error_{len(results)}"
                })
        
        return results

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = [] 
        