import sounddevice as sd

def check_supported_sample_rates():
    rates_to_test = [8000, 16000, 22050, 32000, 44100, 48000, 96000]
    devices = sd.query_devices()
    
    for idx, device in enumerate(devices):
        name = device['name']
        max_input_channels = device['max_input_channels']
        max_output_channels = device['max_output_channels']
        print(f"\n🔍 Device {idx}: {name}")
        
        if max_input_channels > 0:
            print("  🎙 Supported input sample rates:")
            for rate in rates_to_test:
                try:
                    sd.check_input_settings(device=idx, samplerate=rate)
                    print(f"    ✅ {rate} Hz")
                except:
                    print(f"    ❌ {rate} Hz")
        
        if max_output_channels > 0:
            print("  🔊 Supported output sample rates:")
            for rate in rates_to_test:
                try:
                    sd.check_output_settings(device=idx, samplerate=rate)
                    print(f"    ✅ {rate} Hz")
                except:
                    print(f"    ❌ {rate} Hz")

if __name__ == "__main__":
    check_supported_sample_rates()
