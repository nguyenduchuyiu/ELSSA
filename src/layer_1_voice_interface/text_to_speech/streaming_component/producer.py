import asyncio

class StreamingProducer:
    """
    Component 1: Producer Task - Generate audio chunks from text stream
    """
    
    def __init__(self, tts_instance, sample_rate: int = 22050):
        self.tts = tts_instance
        self.sample_rate = sample_rate
        
    async def produce_audio_chunks(
        self,
        text_stream,
        synthesis_queue: asyncio.Queue,
        result: dict
    ):
        """Generate audio chunks from text stream sequentially and put them in synthesis queue"""
        print("🏭 Producer Task started (sequential mode)")
        
        chunk_count = 0
        
        try:
            # Process text chunks sequentially to avoid CoquiTTS cache race conditions
            for text_chunk in text_stream:
                if hasattr(self.tts, 'interrupt_manager') and self.tts.interrupt_manager.is_interrupted():
                    result['interrupted'] = True
                    break
                    
                result['text'] += text_chunk
                print(text_chunk, end="", flush=True)
                
                sentence = text_chunk.strip()
                if sentence:                        
                    try:
                        # Generate audio chunk synchronously to avoid race conditions
                        print(f"\n🔊 Generating chunk {chunk_count}: '{sentence[:50]}...'")
                        audio_chunk = await self.tts._generate_chunk_async(sentence)
                        
                        # Queue immediately when ready 
                        await synthesis_queue.put(audio_chunk)
                        print(f"🏭 Audio chunk {chunk_count} ready and queued")
                        chunk_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error generating audio chunk {chunk_count}: {e}")
                        # Continue with next chunk instead of failing completely
                        continue
                    
                await asyncio.sleep(0.001)  # Small yield to prevent blocking
            
        except Exception as e:
            print(f"⚠️ Error in producer task: {e}")
            result['interrupted'] = True
        finally:
            await synthesis_queue.put(None)
            print(f"🏭 Producer Task completed - generated {chunk_count} chunks")