import asyncio
import gc
import yaml
import json
import tempfile
import os
import sys
from enum import Enum
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load config
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
    THINKING = "thinking"
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

class ELSSAWebSocketSystem:
    """ELSSA system with WebSocket support"""
    
    def __init__(self, config: ELSSAConfig):
        self.config = config
        self.current_state = SystemState.IDLE
        self.state_lock = asyncio.Lock()
        self._interrupt_detected = asyncio.Event()
        self.websocket: Optional[WebSocket] = None
        
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
    
    async def send_message(self, message_type: str, data: dict):
        """Send message to WebSocket client"""
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps({
                    "type": message_type,
                    "data": data
                }))
            except Exception as e:
                print(f"Error sending WebSocket message: {e}")
    
    async def cleanup_resources(self):
        """Clean up all system resources"""
        if self.asr:
            try:
                if self.asr.is_running:
                    await self.asr.stop_async()
                self.asr = None
                await self.send_message("system_log", {"message": "✅ ASR stopped"})
            except Exception as e:
                await self.send_message("system_log", {"message": f"⚠️ Error stopping ASR: {e}"})
                self.asr = None
        
        if self.tts:
            try:
                self.tts.close()
                self.tts = None
                await self.send_message("system_log", {"message": "✅ TTS cleaned up"})
            except Exception as e:
                await self.send_message("system_log", {"message": f"⚠️ Error cleaning TTS: {e}"})
                self.tts = None

        if self.llm_runner:
            try:
                self.llm_runner.stop_server()
                self.llm_runner = None
                await self.send_message("system_log", {"message": "✅ LLM runner stopped"})
            except Exception as e:
                await self.send_message("system_log", {"message": f"⚠️ Error stopping LLM runner: {e}"})
                self.llm_runner = None
        
        if self.wake_handler:
            try:
                self.wake_handler.stop()
                self.wake_handler = None
                await self.send_message("system_log", {"message": "✅ Wake handler stopped"})
            except Exception as e:
                await self.send_message("system_log", {"message": f"⚠️ Error stopping wake handler: {e}"})
                self.wake_handler = None
        
        # End current conversation session
        if self.context_manager:
            try:
                await self.context_manager.end_current_session()
                await self.send_message("system_log", {"message": "✅ Context session ended"})
            except Exception as e:
                await self.send_message("system_log", {"message": f"⚠️ Error ending context session: {e}"})
        
        gc.collect()
    
    async def transition_to_idle(self):
        """Transition to IDLE state"""
        async with self.state_lock:
            await self.send_message("system_log", {"message": "🔄 Transitioning to IDLE state..."})
            self.current_state = SystemState.IDLE
            await self.send_message("state_change", {"state": "idle"})
            
            await self.cleanup_resources()
            
            # Start wake word detection
            try:
                self.wake_handler = WakeWordHandler()
                self.wake_handler.register_callback(self._on_wake_detected)
                self.wake_handler.start()
                await self.send_message("system_log", {"message": "👂 IDLE: Listening for wake word 'alexa'..."})
            except Exception as e:
                await self.send_message("system_log", {"message": f"❌ Error starting wake detection: {e}"})
    
    async def transition_to_active(self):
        """Transition to ACTIVE state"""
        async with self.state_lock:
            await self.send_message("system_log", {"message": "🔄 Transitioning to ACTIVE state..."})
            self.current_state = SystemState.ACTIVE
            await self.send_message("state_change", {"state": "active"})
            
            # Stop wake word detection
            if self.wake_handler:
                self.wake_handler.stop()
                self.wake_handler = None
            
            # Initialize components
            self.tts = TextToSpeech()
            await self.send_message("system_log", {"message": "✅ TTS initialized"})
            
            self.asr = SpeechToText(silence_threshold=self.config.silence_timeout)
            await self.send_message("system_log", {"message": "✅ ASR initialized"})

            self.llm_runner = LLMRunner()
            self.llm_runner.launch()
            await self.send_message("system_log", {"message": "✅ LLM runner initialized"})
            
            # Start new conversation session
            session_id = await self.context_manager.start_new_session()
            await self.send_message("system_log", {"message": f"📝 Started conversation session: {session_id}"})
                
            # Play wake acknowledgment
            play_wake_chime()
    
    async def transition_to_thinking(self):
        """Transition to THINKING state"""
        async with self.state_lock:
            await self.send_message("system_log", {"message": "🤔 Processing your request..."})
            self.current_state = SystemState.THINKING
            await self.send_message("state_change", {"state": "thinking"})

    async def transition_to_speaking(self):
        """Transition to SPEAKING state"""
        async with self.state_lock:
            await self.send_message("system_log", {"message": "🔄 Transitioning to SPEAKING state..."})
            self.current_state = SystemState.SPEAKING
            await self.send_message("state_change", {"state": "speaking"})

    async def transition_to_active_listening(self):
        """Transition to ACTIVE_LISTENING state"""
        async with self.state_lock:
            await self.send_message("system_log", {"message": "🔄 Transitioning to ACTIVE_LISTENING state..."})
            self.current_state = SystemState.ACTIVE_LISTENING
            await self.send_message("state_change", {"state": "active_listening"})
            await self.send_message("system_log", {"message": "👂 ACTIVE_LISTENING: Ready to listen immediately after interrupt"})
    
    def _on_interrupt_detected(self):
        """Callback for interrupt detection"""
        asyncio.create_task(self.send_message("system_log", {"message": "⚡ INTERRUPT DETECTED during TTS!"}))
        self._interrupt_detected.set()

    async def _on_wake_detected(self):
        """Callback for wake word detection"""
        if self.current_state == SystemState.IDLE:
            await self.send_message("system_log", {"message": "🔔 Wake word detected!"})
            await self.transition_to_active()
            asyncio.create_task(self._active_conversation_loop())

    async def speak_with_interrupt_support(self, text: str) -> bool:
        """
        Speak with interrupt detection support
        Returns True if completed, False if interrupted
        """
        if not self.tts:
            await self.send_message("system_log", {"message": "⚠️ TTS not available"})
            return False
        
        await self.transition_to_speaking()
        self._interrupt_detected.clear()
        
        # Send the message to frontend first
        await self.send_message("assistant_message", {"text": text})
            
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
                await self.send_message("state_change", {"state": "active"})
            return True

    def _accumulate_text_stream(self, text_stream):
        """Wrapper to accumulate text chunks until they are longer than 3 characters"""
        accumulated_text = ""
        
        for chunk in text_stream:
            accumulated_text += chunk
            
            if len(accumulated_text.strip()) > 3:
                yield accumulated_text
                accumulated_text = ""
        
        if accumulated_text.strip():
            yield accumulated_text

    async def _process_user_input_streaming(self, user_text: str) -> bool:
        """Process user input and stream response directly to TTS"""
        if not self.llm_runner or not self.tts:
            fallback_response = "Sorry, I'm having trouble processing your request."
            await self.send_message("assistant_message", {"text": fallback_response})
            return True
        
        # Transition to thinking state first
        await self.transition_to_thinking()
        
        # Add user message to context
        await self.context_manager.add_message("user", user_text)
        
        # Get conversation context for LLM
        context_messages = await self.context_manager.get_conversation_context()
        
        # Start streaming response
        stream_response = self.llm_runner.chat_stream(context_messages)
        
        # Wrap stream with text accumulation
        accumulated_stream = self._accumulate_text_stream(stream_response)
        
        # Transition to speaking state only when we start TTS
        await self.transition_to_speaking()
        self._interrupt_detected.clear()
        
        await self.send_message("system_log", {"message": "🤖 Streaming response:"})
        
        # Stream response directly to TTS and WebSocket
        full_response = ""
        try:
            # Collect chunks and send to both TTS and WebSocket
            response_chunks = []
            for chunk in accumulated_stream:
                response_chunks.append(chunk)
                full_response += chunk
                # Send partial response to frontend
                await self.send_message("assistant_response_chunk", {"text": chunk})
            
            # Use TTS streaming capability with accumulated text
            if response_chunks:
                result = await self.tts.speak_stream_async(
                    iter(response_chunks),
                    interruptible=True,
                    interrupt_callback=self._on_interrupt_detected
                )
                
                if result['interrupted']:
                    await self.send_message("system_log", {"message": "🔄 Speech was interrupted"})
                    if full_response:
                        await self.context_manager.add_message("assistant", full_response)
                    await self.transition_to_active_listening()
                    return False
                else:
                    await self.send_message("system_log", {"message": "✅ Speech completed"})
                    await self.context_manager.add_message("assistant", full_response)
                    async with self.state_lock:
                        self.current_state = SystemState.ACTIVE
                        await self.send_message("state_change", {"state": "active"})
                    return True
            
        except Exception as e:
            await self.send_message("system_log", {"message": f"❌ Error in streaming: {e}"})
            raise e

    async def _handle_conversation_turn(self) -> tuple[bool, str]:
        """Handle a single conversation turn"""
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
            
            if silence_detected:
                await asyncio.sleep(0.5)
            
            user_text = ""
            if self.asr.is_running:
                user_text = await self.asr.stop_async() 
            
            has_meaningful_input = user_text and len(user_text.strip()) > 0
            return has_meaningful_input, user_text
            
        except Exception as e:
            await self.send_message("system_log", {"message": f"⚠️ Error in conversation turn: {e}"})
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
                await self.send_message("system_log", {"message": "🎤 ACTIVE_LISTENING: Listening immediately after interrupt..."})
                async with self.state_lock:
                    self.current_state = SystemState.ACTIVE
                    await self.send_message("state_change", {"state": "active"})
            elif self.current_state == SystemState.ACTIVE:
                await self.send_message("system_log", {"message": f"🎤 ACTIVE: Listening... (silence count: {silence_count}/{self.config.max_silence_retries})"})
            else:
                break
            
            # Handle conversation turn
            has_input, user_text = await self._handle_conversation_turn()
            
            if has_input:
                silence_count = 0
                await self.send_message("user_message", {"text": user_text})
                await self.send_message("system_log", {"message": f"📝 User said: '{user_text}'"})
                
                # Use streaming response directly to TTS
                completed = await self._process_user_input_streaming(user_text)
                if not completed:
                    continue
            else:
                silence_count += 1
                await self.send_message("system_log", {"message": f"🔇 No input detected. Silence count: {silence_count}/{self.config.max_silence_retries}"})
                
                if silence_count >= self.config.max_silence_retries:
                    break
                
                completed = await self.speak_with_interrupt_support("Are you still there?")
                
                if not completed:
                    continue
        
        # Exit conversation
        if silence_count >= self.config.max_silence_retries:
            await self.speak_with_interrupt_support("Call me if you need anything!")
            await self.send_message("system_log", {"message": "🛑 Max silence retries reached. Returning to IDLE."})
        else:
            await self.send_message("system_log", {"message": "🛑 Conversation ended. Returning to IDLE."})
        
        await self.transition_to_idle()

    async def process_audio_file(self, audio_file_path: str):
        """Process uploaded audio file"""
        if self.current_state != SystemState.ACTIVE:
            await self.send_message("system_log", {"message": "⚠️ System not active. Cannot process audio."})
            return
        
        try:
            # In real implementation, use ASR to transcribe
            # For now, simulate ASR processing
            user_text = "Audio transcription would appear here"
            
            await self.send_message("user_message", {"text": user_text})
            await self.send_message("system_log", {"message": f"📝 User said: '{user_text}'"})
            
            # Process the transcribed text
            await self._process_user_input_streaming(user_text)
            
        except Exception as e:
            await self.send_message("system_log", {"message": f"❌ Error processing audio: {e}"})

# FastAPI app
app = FastAPI(title="ELSSA WebSocket App", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app_static"), name="static")

# Global ELSSA system instance
elssa_config = ELSSAConfig()
elssa_system = ELSSAWebSocketSystem(elssa_config)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    with open("app_static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    elssa_system.websocket = websocket
    
    try:
        # Start system in idle state
        await elssa_system.transition_to_idle()
        
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "wake":
                if elssa_system.current_state == SystemState.IDLE:
                    await elssa_system.transition_to_active()
                    asyncio.create_task(elssa_system._active_conversation_loop())
            
            elif message["type"] == "sleep":
                await elssa_system.transition_to_idle()
            
            elif message["type"] == "audio_data":
                # Handle audio data from frontend
                # In real implementation, process this audio data
                pass
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        elssa_system.websocket = None
        await elssa_system.cleanup_resources()
    except Exception as e:
        print(f"WebSocket error: {e}")
        elssa_system.websocket = None

if __name__ == "__main__":
    print("🚀 Starting ELSSA WebSocket App...")
    print("📱 Frontend will be available at: http://localhost:8001")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
