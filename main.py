from src.layer_1_voice_interface.wake_word_handler import WakeWordHandler
from src.layer_1_voice_interface.speech_to_text import SpeechToText
from src.layer_1_voice_interface.text_to_speech import TextToSpeech
from scipy.io.wavfile import read
import threading
import time
import asyncio
import numpy as np
import gc
from enum import Enum

# Path đến file .wav quick reply
WAKE_AUDIO_PATH = "assets/audio/elssa_online.wav"

# Wake word model
WAKE_WORD_MODEL = ["models/openwakeword/alexa_v0.1.tflite"]

# Session config
SILENCE_TIMEOUT = 5.0     
ASR_TIMEOUT = 60         # timeout cho mỗi ASR session
MAX_SILENCE_RETRIES = 3    # 3 lần không nói gì thì về idle

class SystemState(Enum):
    IDLE = "idle"              # Chỉ lắng nghe wake word
    ACTIVE = "active"          # Đang trong conversation session
    SPEAKING = "speaking"      # Đang phát TTS, có thể bị interrupt
    ACTIVE_LISTENING = "active_listening"  # Chuyển từ interrupt, lắng nghe ngay

# Global state
current_state = SystemState.IDLE
wake_handler = None
asr = None
tts = None
state_lock = asyncio.Lock()

# Interrupt handling
_interrupt_detected = asyncio.Event()

async def transition_to_idle():
    """Chuyển hệ thống về trạng thái IDLE"""
    global current_state, wake_handler, asr, tts
    
    async with state_lock:
        print("🔄 Transitioning to IDLE state...")
        current_state = SystemState.IDLE
        
        # Cleanup active state resources
        if asr is not None:
            try:
                if asr.is_running:
                    await asr.stop_async()
                asr = None
                print("✅ ASR stopped")
            except Exception as e:
                print(f"⚠️ Error stopping ASR: {e}")
                asr = None
        
        if tts is not None:
            try:
                tts.close()
                tts = None
                print("✅ TTS cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning TTS: {e}")
                tts = None
        
        # Force garbage collection
        gc.collect()
        print("🗑️ Memory cleaned")
        
        # Start wake word detection for IDLE state
        try:
            wake_handler = WakeWordHandler(wakeword_models=WAKE_WORD_MODEL)
            wake_handler.register_callback(on_wake_detected)
            wake_handler.start()
            print("👂 IDLE: Listening for wake word 'alexa'...")
        except Exception as e:
            print(f"❌ Error starting wake detection: {e}")

async def transition_to_active():
    """Chuyển hệ thống về trạng thái ACTIVE"""
    global current_state, wake_handler, asr, tts
    
    async with state_lock:
        print("🔄 Transitioning to ACTIVE state...")
        current_state = SystemState.ACTIVE
        
        # Stop wake word detection
        wake_handler.stop()
        wake_handler = None
        print("✅ Wake detection stopped")
        
        # Initialize TTS and ASR for active state
        tts = TextToSpeech()
        print("✅ TTS initialized for ACTIVE state")
    
        # Initialize ASR once and reuse throughout conversation
        asr = SpeechToText(silence_threshold=SILENCE_TIMEOUT)
        print("✅ ASR initialized for ACTIVE state")
            
        # Play wake acknowledgment
        sr, wav = read(WAKE_AUDIO_PATH)
        print(f"🔊 Wake audio - Sample rate: {sr}, Shape: {wav.shape}, Range: {wav.min()} to {wav.max()}")
        wav = wav.astype(np.float32) / 32768.0
        print(f"🔊 Wake audio normalized - Range: {wav.min():.6f} to {wav.max():.6f}")
        await tts.audio_manager.play_audio_async(wav, blocking=True)
        print("🔔 Wake acknowledgment played")
        
        print("🎯 ACTIVE: Ready for conversation")

async def transition_to_speaking():
    """Chuyển hệ thống về trạng thái SPEAKING"""
    global current_state
    
    async with state_lock:
        print("🔄 Transitioning to SPEAKING state...")
        current_state = SystemState.SPEAKING
        print("🗣️ SPEAKING: Now speaking, can be interrupted")

async def transition_to_active_listening():
    """Chuyển hệ thống về trạng thái ACTIVE_LISTENING sau interrupt"""
    global current_state
    
    async with state_lock:
        print("🔄 Transitioning to ACTIVE_LISTENING state...")
        current_state = SystemState.ACTIVE_LISTENING
        print("👂 ACTIVE_LISTENING: Ready to listen immediately after interrupt")

def on_interrupt_detected():
    """Callback khi detect wake word trong lúc đang nói (TTS)"""
    global _interrupt_detected
    print("⚡ INTERRUPT DETECTED during TTS!")
    _interrupt_detected.set()

async def speak_with_interrupt_support(text: str) -> bool:
    """
    Nói với hỗ trợ interrupt detection. 
    Returns True if completed, False if interrupted.
    """
    global tts, current_state
    
    if not tts:
        print("⚠️ TTS not available")
        return False
    
    # Transition to SPEAKING state
    await transition_to_speaking()
    
    # Clear interrupt flag
    _interrupt_detected.clear()
        
    # Speak with interrupt monitoring
    result = await tts.speak_async(
        text, 
        play_audio=True,
        interruptible=True,
        interrupt_callback=on_interrupt_detected
    )
    
    if result['interrupted']:
        print("🔄 Speech was interrupted")
        # Transition to active listening immediately
        await transition_to_active_listening()
        return False
    else:
        print("✅ Speech completed")
        # Return to active conversation state
        async with state_lock:
            current_state = SystemState.ACTIVE
        return True

async def on_wake_detected():
    """Callback khi wake word được phát hiện"""
    if current_state == SystemState.IDLE:
        print("🔔 Wake word detected!")
        await transition_to_active()
        # Start conversation task
        asyncio.create_task(active_conversation_loop())

async def active_conversation_loop():
    """Main conversation loop cho ACTIVE state"""
    global asr, tts, current_state
    
    silence_count = 0
    
    while current_state in [SystemState.ACTIVE, SystemState.ACTIVE_LISTENING] and silence_count < MAX_SILENCE_RETRIES:
        
        # Handle different states
        if current_state == SystemState.ACTIVE_LISTENING:
            # Jump straight to listening after interrupt
            print("🎤 ACTIVE_LISTENING: Listening immediately after interrupt...")
            async with state_lock:
                current_state = SystemState.ACTIVE
        elif current_state == SystemState.ACTIVE:
            print(f"🎤 ACTIVE: Listening... (silence count: {silence_count}/{MAX_SILENCE_RETRIES})")
        else:
            # If not in ACTIVE or ACTIVE_LISTENING state, break out
            break
        
        try:
            # Reset ASR session for new conversation turn
            asr.reset_session()
            
            # Start ASR (reuse existing instance)
            if not asr.is_running:
                asr.start()
            
            # Wait for user input with timeout
            try:
                silence_detected, user_text = await asyncio.wait_for(
                    asr.wait_for_silence_or_text_async(timeout=ASR_TIMEOUT),
                    timeout=ASR_TIMEOUT + 1.0  # Extra buffer for async version
                )
            except asyncio.TimeoutError:
                print("⏰ Conversation timeout")
                silence_detected, user_text = False, ""
            
            # Stop ASR (but don't destroy instance)
            if asr.is_running:
                await asr.stop_async()
            
            if user_text and len(user_text.strip()) > 0:
                # Got meaningful input - reset silence count
                silence_count = 0
                print(f"📝 User said: '{user_text}'")
                
                # Process and respond (dummy response for now)
                response = f"You said: {user_text.upper()}"
                print(f"🤖 Responding: '{response}'")
                
                # Speak with interrupt support
                completed = await speak_with_interrupt_support(response)
                
                if not completed:
                    # Speech was interrupted, continue loop in ACTIVE_LISTENING state
                    print("🔄 Continuing conversation after interrupt...")
                    continue
                
                # Continue conversation loop normally
                
            else:
                # No meaningful input - increment silence count
                silence_count += 1
                print(f"🔇 No input detected. Silence count: {silence_count}/{MAX_SILENCE_RETRIES}")
                
                if silence_count == 1:
                    completed = await speak_with_interrupt_support("Where did you go?")
                    if not completed:
                        continue
                elif silence_count == 2:
                    completed = await speak_with_interrupt_support("Still can't hear you...")
                    if not completed:
                        continue
                # On 3rd silence, loop will exit
                
        except Exception as e:
            print(f"⚠️ Error in conversation loop: {e}")
            if asr and asr.is_running:
                await asr.stop_async()
            silence_count += 1
    
    # Exit conversation - return to IDLE
    if silence_count >= MAX_SILENCE_RETRIES:
        print("🛑 Max silence retries reached. Returning to IDLE.")
    else:
        print("🛑 Conversation ended. Returning to IDLE.")
    
    await transition_to_idle()

async def main():
    """Main function - khởi tạo hệ thống ở trạng thái IDLE"""
    print("🚀 Starting ELSSA system...")
    
    try:
        # Start in IDLE state
        await transition_to_idle()
        
        # Keep main task alive
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Cleanup based on current state
        async with state_lock:
            if asr:
                await asr.stop_async()
            if tts:
                tts.close()
            if wake_handler:
                wake_handler.stop()
        
        print("👋 Goodbye!")
    
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        # Force cleanup
        try:
            if asr:
                await asr.stop_async()
            if tts:
                tts.close()
            if wake_handler:
                wake_handler.stop()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
