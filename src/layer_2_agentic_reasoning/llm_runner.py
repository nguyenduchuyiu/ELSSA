import threading
import requests
import subprocess
import time

class LLMRunner:
    def __init__(self, module_path="src.layer_2_agentic_reasoning.llm_server", port=8000):
        self.module_path = module_path
        self.port = port
        self.proc = None
        self.llm_loaded = False

    def _format_chatml_prompt(self, messages):
        formatted = ""
        for msg in messages:
            role = msg["role"].lower()
            content = msg["content"].strip()
            formatted += f"<|{role}|>\n{content}\n"
        formatted += "<|assistant|>\n"
        return formatted

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
        print("📦 Loading LLM...")
        r = requests.post(f"http://localhost:{self.port}/load")
        if r.status_code == 200:
            self.llm_loaded = True
            print("✅ LLM loaded")
        else:
            print("❌ LLM load failed")

    def unload_model(self):
        print("🧹 Unloading LLM...")
        r = requests.post(f"http://localhost:{self.port}/unload")
        if r.status_code == 200:
            self.llm_loaded = False
            print("✅ LLM unloaded")
        else:
            print("❌ LLM unload failed")

    def chat(self, messages):
        if self.proc is None:
            print("❌ Server not running")
            return
        
        if not self.llm_loaded:
            print("❌ LLM not loaded")
            return

        url = f"http://localhost:{self.port}/chat"
        data = {"prompt": self._format_chatml_prompt(messages)}
        print("📤 Sending prompt...")
        with requests.post(url, json=data, stream=True, timeout=120) as response:
            print(f"✅ Status code: {response.status_code}")
            print("🗣 Response:")
            for chunk in response.iter_content(decode_unicode=True, chunk_size=20):
                if chunk:
                    yield chunk

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
        