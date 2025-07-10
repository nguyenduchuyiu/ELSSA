import sys
from pathlib import Path

# Use absolute paths for local development
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "libs"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer_1_voice_interface.tts_client import TTSClient



try:
    from src.layer_1_voice_interface.tts_server import app
    print('✅ TTS server import successful')

    # Check OpenVoice dependencies
    from openvoice import se_extractor
    from openvoice.api import BaseSpeakerTTS, ToneColorConverter
    print('✅ OpenVoice dependencies available')

    # Check required model and asset files
    required_files = [
        '/home/huy/Project/ELSSA/models/openvoice/checkpoints/base_speakers/EN/config.json',
        '/home/huy/Project/ELSSA/models/openvoice/checkpoints/base_speakers/EN/checkpoint.pth',
        '/home/huy/Project/ELSSA/models/openvoice/checkpoints/base_speakers/EN/en_default_se.pth',
        '/home/huy/Project/ELSSA/models/openvoice/checkpoints/converter/config.json',
        '/home/huy/Project/ELSSA/models/openvoice/checkpoints/converter/checkpoint.pth',
        '/home/huy/Project/ELSSA/assets/audio/ref_voice.mp3'
    ]

    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print('❌ Missing files:')
        for f in missing:
            print(f'  - {f}')
    else:
        print('✅ All required files exist')

    # Run the TTS server
    tts = TTSClient()
    tts.launch()  # Start server and load models
    time.sleep(3)  # Wait for server startup
    print("✅ TTS loaded successfully")

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()