import sounddevice as sd
import scipy.io.wavfile
import os

def play_chime(chime_path):
    try:
        if not os.path.exists(chime_path):
            print(f"Chime file not found at {chime_path}")
            return
        
        sr, audio = scipy.io.wavfile.read(chime_path)
        if audio.dtype != 'float32':
            audio = audio.astype('float32') / 32767
        sd.play(audio, sr)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Error playing {chime_path} chime: {e}")

def play_listening_chime():
    """
    Play a soft and gentle 'ting ting' chime that signals the system is listening.
    Inspired by voice assistant UX design.
    """
    chime_path = os.path.join("assets", "audio", "listening_chime.wav")
    play_chime(chime_path)

        
def play_wake_chime():
    chime_path = os.path.join("assets", "audio", "elssa_online.wav")
    play_chime(chime_path)


def play_processing_chime():
    chime_path = os.path.join("assets", "audio", "processing_chime.wav")
    play_chime(chime_path)

        
           
