#!/usr/bin/env python3
"""
Test script để so sánh sentence-level streaming vs word-level streaming
"""

import asyncio
import time
import requests
import json


async def test_sentence_level_streaming():
    """Test sentence-level streaming behavior"""
    
    print("🧪 Testing Sentence-Level Streaming")
    print("=" * 50)
    
    test_queries = [
        "Tell me about quantum computing and its applications.",
        "Explain machine learning in simple terms with examples.",
        "What are the benefits of renewable energy? Give me details."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        # Track sentence boundaries
        sentence_count = 0
        current_sentence = ""
        
        test_data = {
            "prompt": [{"role": "user", "content": query}]
        }
        
        start_time = time.time()
        first_sentence_time = None
        
        try:
            response = requests.post(
                "http://localhost:8000/chat-stream",
                json=test_data,
                stream=True,
                timeout=30
            )
            
            print("📦 Streaming chunks:")
            chunk_count = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.strip():
                    try:
                        chunk_data = json.loads(line)
                        chunk_type = chunk_data.get("type")
                        
                        if chunk_type == "content_chunk":
                            content = chunk_data.get("content", "")
                            current_sentence += content
                            chunk_count += 1
                            
                            # Check if we hit a sentence boundary
                            if content and content[-1] in '.!?:;':
                                sentence_count += 1
                                
                                if first_sentence_time is None:
                                    first_sentence_time = time.time()
                                    latency = first_sentence_time - start_time
                                    print(f"🗣️  First sentence ({sentence_count}): '{current_sentence.strip()[:60]}...'")
                                    print(f"⚡ Time to first sentence: {latency:.2f}s")
                                else:
                                    print(f"🗣️  Sentence {sentence_count}: '{current_sentence.strip()[:60]}...'")
                                
                                current_sentence = ""
                            
                            # Show chunk progress (every 10 chunks)
                            if chunk_count % 20 == 0:
                                print(f"📝 Chunk {chunk_count}: Buffer='{current_sentence[:30]}...'")
                        
                        elif chunk_type == "final_response":
                            end_time = time.time()
                            total_time = end_time - start_time
                            
                            # Handle remaining sentence
                            if current_sentence.strip():
                                sentence_count += 1
                                print(f"🗣️  Final sentence ({sentence_count}): '{current_sentence.strip()[:60]}...'")
                            
                            print(f"✅ Total time: {total_time:.2f}s")
                            print(f"📊 Total sentences: {sentence_count}")
                            print(f"📦 Total chunks: {chunk_count}")
                            
                            if first_sentence_time:
                                print(f"⚡ Avg time per sentence: {total_time/sentence_count:.2f}s")
                            
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(1)  # Short pause between tests


def simulate_sentence_detection(text_stream):
    """Simulate sentence detection logic"""
    
    print("\n🔍 Sentence Detection Logic Test")
    print("=" * 40)
    
    # Sample streaming chunks (simulated)
    chunks = [
        "Quantum", " computing", " is", " a", " revolutionary", " technology", 
        ".", " It", " uses", " quantum", " bits", " or", " qubits", 
        ".", " These", " can", " exist", " in", " superposition", ",", 
        " which", " means", " they", " can", " be", " both", " 0", 
        " and", " 1", " simultaneously", ".", " This", " property", 
        " allows", " quantum", " computers", " to", " process", 
        " information", " exponentially", " faster", "!"
    ]
    
    sentence_buffer = ""
    sentence_count = 0
    
    for i, chunk in enumerate(chunks):
        sentence_buffer += chunk
        
        # Check sentence completion
        is_complete = False
        if chunk.strip() and chunk.strip()[-1] in '.!?:;':
            is_complete = True
        elif len(sentence_buffer) > 50 and chunk in [', and', ', but', ', so']:
            is_complete = True
        elif len(sentence_buffer) > 100:
            is_complete = True
        
        if is_complete:
            sentence_count += 1
            print(f"🗣️  Sentence {sentence_count}: '{sentence_buffer.strip()}'")
            sentence_buffer = ""
        else:
            print(f"📝 Chunk {i+1}: Buffer='{sentence_buffer[:40]}...' ({'Complete' if is_complete else 'Building'})")


async def main():
    """Main test function"""
    print("🧪 Sentence-Level Streaming Test")
    print("=" * 50)
    
    # Check if server is running
    try:
        requests.get("http://localhost:8000/status", timeout=5)
        print("✅ LLM Server is running")
    except:
        print("❌ LLM Server not running. Please start it first.")
        print("   Run: python -m src.layer_2_agentic_reasoning.llm_server")
        return
    
    print()
    
    # Test sentence detection logic
    simulate_sentence_detection([])
    
    # Test real streaming
    await test_sentence_level_streaming()
    
    print("\n🎉 Sentence-level streaming test completed!")
    print("💡 This should provide much smoother TTS compared to word-level chunks")


if __name__ == "__main__":
    asyncio.run(main()) 