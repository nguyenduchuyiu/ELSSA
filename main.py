import asyncio
import gc
import yaml
from enum import Enum
from typing import Optional
import sys
import os
import yaml 

config = yaml.safe_load(open("config.yaml", "r"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "libs")))
 

from src.layer_1_voice_interface.wake_word_handler import WakeWordHandler
from src.layer_1_voice_interface.speech_to_text import SpeechToText
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager
from src.utils.sound_player import play_wake_chime

if config['tts_engine'] == "openvoice":
    from src.layer_1_voice_interface.text_to_speech import OpenVoiceTTS as TextToSpeech
else:
    from src.layer_1_voice_interface.text_to_speech import CoquiTTS as TextToSpeech

class SystemState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    SPEAKING = "speaking"
    ACTIVE_LISTENING = "active_listening"


class ELSSAConfig:
    """Configuration management for ELSSA system"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.wake_audio_path = "assets/audio/elssa_online.wav"
        self.silence_timeout = self.config['silence_timeout']
        self.asr_timeout = self.config['asr_timeout']
        self.max_silence_retries = self.config['max_silence_retries']
        self.context_dir = "data/context"
        self.max_context_length = self.config['max_context_length']


class ELSSASystem:
    """Main ELSSA system class that manages all components and state transitions"""
    
    def __init__(self, config: ELSSAConfig):
        self.config = config
        self.current_state = SystemState.IDLE
        self.state_lock = asyncio.Lock()
        self._interrupt_detected = asyncio.Event()
        
        # Components
        self.wake_handler: Optional[WakeWordHandler] = None
        self.asr: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.llm_runner: Optional[LLMRunner] = None
        
        # Context management
        self.context_manager = ContextManager(
            context_dir=self.config.context_dir,
            max_context_length=self.config.max_context_length
        )
    
    async def cleanup_resources(self):
        """Clean up all system resources"""
        if self.asr:
            try:
                if self.asr.is_running:
                    await self.asr.stop_async()
                self.asr = None
                print("✅ ASR stopped")
            except Exception as e:
                print(f"⚠️ Error stopping ASR: {e}")
                self.asr = None
        
        if self.tts:
            try:
                self.tts.close()
                self.tts = None
                print("✅ TTS cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning TTS: {e}")
                self.tts = None

        if self.llm_runner:
            try:
                self.llm_runner.stop_server()
                self.llm_runner = None
                print("✅ LLM runner stopped")
            except Exception as e:
                print(f"⚠️ Error stopping LLM runner: {e}")
                self.llm_runner = None
        
        if self.wake_handler:
            try:
                self.wake_handler.stop()
                self.wake_handler = None
                print("✅ Wake handler stopped")
            except Exception as e:
                print(f"⚠️ Error stopping wake handler: {e}")
                self.wake_handler = None
        
        # End current conversation session
        if self.context_manager:
            try:
                await self.context_manager.end_current_session()
                print("✅ Context session ended")
            except Exception as e:
                print(f"⚠️ Error ending context session: {e}")
        
        gc.collect()
    
    async def transition_to_idle(self):
        """Transition to IDLE state"""
        async with self.state_lock:
            print("🔄 Transitioning to IDLE state...")
            self.current_state = SystemState.IDLE
            
            await self.cleanup_resources()
            
            # Start wake word detection
            try:
                self.wake_handler = WakeWordHandler()
                self.wake_handler.register_callback(self._on_wake_detected)
                self.wake_handler.start()
                print("👂 IDLE: Listening for wake word 'alexa'...")
            except Exception as e:
                print(f"❌ Error starting wake detection: {e}")
    
    async def transition_to_active(self):
        """Transition to ACTIVE state"""
        async with self.state_lock:
            print("🔄 Transitioning to ACTIVE state...")
            self.current_state = SystemState.ACTIVE
            
            # Stop wake word detection
            if self.wake_handler:
                self.wake_handler.stop()
                self.wake_handler = None
            
            # Initialize components
            self.tts = TextToSpeech()
            print("✅ TTS initialized")
            
            self.asr = SpeechToText(silence_threshold=self.config.silence_timeout)
            print("✅ ASR initialized")

            self.llm_runner = LLMRunner()
            self.llm_runner.launch()
            print("✅ LLM runner initialized")
            
            # Start new conversation session
            session_id = await self.context_manager.start_new_session()
            print(f"📝 Started conversation session: {session_id}")
                
            # Play wake acknowledgment
            play_wake_chime()
    
    async def transition_to_speaking(self):
        """Transition to SPEAKING state"""
        async with self.state_lock:
            print("🔄 Transitioning to SPEAKING state...")
            self.current_state = SystemState.SPEAKING

    async def transition_to_active_listening(self):
        """Transition to ACTIVE_LISTENING state"""
        async with self.state_lock:
            print("🔄 Transitioning to ACTIVE_LISTENING state...")
            self.current_state = SystemState.ACTIVE_LISTENING
            print("👂 ACTIVE_LISTENING: Ready to listen immediately after interrupt")
    
    def _on_interrupt_detected(self):
        """Callback for interrupt detection"""
        print("⚡ INTERRUPT DETECTED during TTS!")
        self._interrupt_detected.set()

    async def _on_wake_detected(self):
        """Callback for wake word detection"""
        if self.current_state == SystemState.IDLE:
            print("🔔 Wake word detected!")
            await self.transition_to_active()
            asyncio.create_task(self._active_conversation_loop())

    async def speak_with_interrupt_support(self, text: str) -> bool:
        """
        Speak with interrupt detection support
        Returns True if completed, False if interrupted
        """
        if not self.tts:
            print("⚠️ TTS not available")
            return False
        
        await self.transition_to_speaking()
        self._interrupt_detected.clear()
            
        result = await self.tts.speak_async(
            text, 
            play_audio=True,
            interruptible=True,
            interrupt_callback=self._on_interrupt_detected
        )
        
        if result['interrupted']:
            await self.transition_to_active_listening()
            return False
        else:
            async with self.state_lock:
                self.current_state = SystemState.ACTIVE
            return True

    async def _process_user_input_streaming(self, user_text: str) -> bool:
        """
        Process user input and stream response directly to TTS
        Returns True if completed, False if interrupted
        """
        if not self.llm_runner or not self.tts:
            fallback_response = "Sorry, I'm having trouble processing your request."
            return await self.speak_with_interrupt_support(fallback_response)
        
        # Add user message to context
        await self.context_manager.add_message("user", user_text)
        
        # Get conversation context for LLM
        context_messages = await self.context_manager.get_conversation_context()
        
        # Start streaming response
        stream_response = self.llm_runner.chat(context_messages)
        
        # Transition to speaking state
        await self.transition_to_speaking()
        self._interrupt_detected.clear()
        
        print("🤖 Streaming response: ", end="", flush=True)
        
        # Stream response directly to TTS
        full_response = ""
        try:
            # Use TTS streaming capability
            result = await self.tts.speak_stream_async(
                stream_response,
                interruptible=True,
                interrupt_callback=self._on_interrupt_detected
            )
            
            # Collect the full response for context saving
            full_response = result.get('text', '')
            
            if result['interrupted']:
                print("\n🔄 Speech was interrupted")
                # Still save partial response to context
                if full_response:
                    await self.context_manager.add_message("assistant", full_response)
                await self.transition_to_active_listening()
                return False
            else:
                print("\n✅ Speech completed")
                # Save full response to context
                await self.context_manager.add_message("assistant", full_response)
                async with self.state_lock:
                    self.current_state = SystemState.ACTIVE
                return True
                
        except Exception as e:
            raise e

    async def _process_user_input(self, user_text: str) -> str:
        """Process user input and generate response (legacy method for non-streaming)"""
        if not self.llm_runner:
            return "Sorry, I'm having trouble processing your request."
        
        # Add user message to context
        await self.context_manager.add_message("user", user_text)
        
        # Get conversation context for LLM
        context_messages = await self.context_manager.get_conversation_context()
        
        stream_response = self.llm_runner.chat(context_messages)
        
        response = ""
        for chunk in stream_response:
            response += chunk
            print(chunk, end="", flush=True)
        
        # Add assistant response to context
        await self.context_manager.add_message("assistant", response)
        
        return response

    async def _handle_conversation_turn(self) -> tuple[bool, str]:
        """
        Handle a single conversation turn
        Returns (has_input, user_text)
        """
        try:
            self.asr.reset_session()
            
            if not self.asr.is_running:
                self.asr.start()
            
            try:
                silence_detected, _ = await asyncio.wait_for(
                    self.asr.wait_for_silence_or_text_async(timeout=self.config.asr_timeout),
                    timeout=self.config.asr_timeout + 1.0
                )
            except asyncio.TimeoutError:
                silence_detected = False
            
            # Add small delay after silence detection to allow processing remaining audio chunks
            if silence_detected:
                await asyncio.sleep(0.5)  # Wait for any remaining chunks to be processed
            
            # Get final text from stop_async() which includes all processed chunks
            user_text = ""
            if self.asr.is_running:
                user_text = await self.asr.stop_async() 
            
            has_meaningful_input = user_text and len(user_text.strip()) > 0
            return has_meaningful_input, user_text
            
        except Exception as e:
            print(f"⚠️ Error in conversation turn: {e}")
            if self.asr and self.asr.is_running:
                await self.asr.stop_async()
            return False, ""

    async def _active_conversation_loop(self):
        """Main conversation loop for ACTIVE state"""
        silence_count = 0
        
        while (self.current_state in [SystemState.ACTIVE, SystemState.ACTIVE_LISTENING] 
               and silence_count < self.config.max_silence_retries):
            
            # Handle state transitions
            if self.current_state == SystemState.ACTIVE_LISTENING:
                print("🎤 ACTIVE_LISTENING: Listening immediately after interrupt...")
                async with self.state_lock:
                    self.current_state = SystemState.ACTIVE
            elif self.current_state == SystemState.ACTIVE:
                print(f"🎤 ACTIVE: Listening... (silence count: {silence_count}/{self.config.max_silence_retries})")
            else:
                break
            
            # Handle conversation turn
            has_input, user_text = await self._handle_conversation_turn()
            # has_input = True            
            # user_text = random.choice(qa_test_list)
            
            if has_input:
                silence_count = 0
                print(f"📝 User said: '{user_text}'")
                
                # Use streaming response directly to TTS
                completed = await self._process_user_input_streaming(user_text)
                if not completed:
                    continue
            else:
                silence_count += 1
                print(f"🔇 No input detected. Silence count: {silence_count}/{self.config.max_silence_retries}")
                
                if silence_count == 1:
                    completed = await self.speak_with_interrupt_support("Where did you go?")
                elif silence_count == 2:
                    completed = await self.speak_with_interrupt_support("Still can't hear you!")
                else:
                    break
                
                if not completed:
                    continue
        
        # Exit conversation
        if silence_count >= self.config.max_silence_retries:
            print("🛑 Max silence retries reached. Returning to IDLE.")
        else:
            print("🛑 Conversation ended. Returning to IDLE.")
        
        await self.transition_to_idle()

    async def start(self):
        """Start the ELSSA system"""
        print("🚀 Starting ELSSA system...")
        await self.transition_to_idle()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await self.cleanup_resources()
            print("👋 Goodbye!")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            await self.cleanup_resources()


async def main():
    """Main entry point"""
    config = ELSSAConfig()
    system = ELSSASystem(config)
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())
