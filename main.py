from src.layer_1_voice_interface.wake_word_handler import WakeWordHandler
from src.layer_1_voice_interface.speech_to_text import SpeechToText
from src.layer_1_voice_interface.text_to_speech import TextToSpeech
from scipy.io.wavfile import read
import threading
import time
import numpy as np
import gc
from enum import Enum

# Path đến file .wav quick reply
WAKE_AUDIO_PATH = "assets/audio/elssa_online.wav"

# Session config
SILENCE_TIMEOUT = 3.0       # 3 giây silence detection
ASR_TIMEOUT = 10.0         # timeout cho mỗi ASR session
MAX_SILENCE_RETRIES = 3    # 3 lần không nói gì thì về idle

class SystemState(Enum):
    IDLE = "idle"      # Chỉ lắng nghe wake word
    ACTIVE = "active"  # Đang trong conversation session

# Global state
current_state = SystemState.IDLE
wake_handler = None
asr = None
tts = None
state_lock = threading.Lock()

def transition_to_idle():
    """Chuyển hệ thống về trạng thái IDLE"""
    global current_state, wake_handler, asr, tts
    
    with state_lock:
        print("🔄 Transitioning to IDLE state...")
        current_state = SystemState.IDLE
        
        # Cleanup active state resources
        if asr is not None:
            try:
                if asr.is_running:
                    asr.stop()
                asr = None
                print("✅ ASR cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning ASR: {e}")
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
            wake_models = ["models/openwakeword/alexa_v0.1.tflite"]
            wake_handler = WakeWordHandler(wake_models)
            wake_handler.register_callback(on_wake_detected)
            wake_handler.start()
            print("👂 IDLE: Listening for wake word 'alexa'...")
        except Exception as e:
            print(f"❌ Error starting wake detection: {e}")

def transition_to_active():
    """Chuyển hệ thống về trạng thái ACTIVE"""
    global current_state, wake_handler, asr, tts
    
    with state_lock:
        print("🔄 Transitioning to ACTIVE state...")
        current_state = SystemState.ACTIVE
        
        # Stop wake word detection
        if wake_handler is not None:
            try:
                wake_handler.stop()
                wake_handler = None
                print("✅ Wake detection stopped")
            except Exception as e:
                print(f"⚠️ Error stopping wake detection: {e}")
        
        # Initialize TTS for active state
        try:
            tts = TextToSpeech()
            print("✅ TTS initialized for ACTIVE state")
            
            # Play wake acknowledgment
            sr, wav = read(WAKE_AUDIO_PATH)
            wav = wav.astype(np.float32) / 32768.0
            tts.audio_manager.play_audio(wav, blocking=True)
            print("🔔 Wake acknowledgment played")
            
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")
            transition_to_idle()  # Fallback to idle
            return
        
        print("🎯 ACTIVE: Ready for conversation")

def on_wake_detected():
    """Callback khi wake word được phát hiện"""
    if current_state == SystemState.IDLE:
        print("🔔 Wake word detected!")
        transition_to_active()
        # Start conversation thread
        conversation_thread = threading.Thread(target=active_conversation_loop, daemon=True)
        conversation_thread.start()

def active_conversation_loop():
    """Main conversation loop cho ACTIVE state"""
    global asr, tts
    
    silence_count = 0
    
    while current_state == SystemState.ACTIVE and silence_count < MAX_SILENCE_RETRIES:
        print(f"🎤 ACTIVE: Listening... (silence count: {silence_count}/{MAX_SILENCE_RETRIES})")
        
        try:
            # Start ASR session
            asr = SpeechToText(silence_threshold=SILENCE_TIMEOUT)
            asr.start()
            
            # Wait for user input
            silence_detected, user_text = asr.wait_for_silence_or_text(timeout=ASR_TIMEOUT)
            asr.stop()
            asr = None
            
            if user_text and len(user_text.strip()) > 0:
                # Got meaningful input - reset silence count
                silence_count = 0
                print(f"📝 User said: '{user_text}'")
                
                # Process and respond (dummy response for now)
                response = f"You said: {user_text.upper()}"
                print(f"🤖 Responding: '{response}'")
                
                if tts:
                    tts.speak(response)
                
                # Continue conversation loop
                
            else:
                # No meaningful input - increment silence count
                silence_count += 1
                print(f"🔇 No input detected. Silence count: {silence_count}/{MAX_SILENCE_RETRIES}")
                
                if silence_count == 1:
                    if tts:
                        tts.speak("Where did you go?")
                elif silence_count == 2:
                    if tts:
                        tts.speak("Still can't hear you...")
                # On 3rd silence, loop will exit
                
        except Exception as e:
            print(f"⚠️ Error in conversation loop: {e}")
            if asr:
                asr.stop()
                asr = None
            silence_count += 1
    
    # Exit conversation - return to IDLE
    if silence_count >= MAX_SILENCE_RETRIES:
        print("🛑 Max silence retries reached. Returning to IDLE.")
    else:
        print("🛑 Conversation ended. Returning to IDLE.")
    
    transition_to_idle()

def main():
    """Main function - khởi tạo hệ thống ở trạng thái IDLE"""
    print("🚀 Starting ELSSA system...")
    
    try:
        # Start in IDLE state
        transition_to_idle()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Optional: print state every 30 seconds for debugging
            # if int(time.time()) % 30 == 0:
            #     print(f"📊 Current state: {current_state.value}")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Cleanup based on current state
        with state_lock:
            if asr:
                asr.stop()
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
                asr.stop()
            if tts:
                tts.close()
            if wake_handler:
                wake_handler.stop()
        except:
            pass

if __name__ == "__main__":
    main()
