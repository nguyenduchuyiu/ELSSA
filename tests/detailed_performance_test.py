#!/usr/bin/env python3
"""
Detailed ELSSA Performance Test
Đo memory thực tế từng giai đoạn với baseline + matplotlib charts
"""

import asyncio
import time
import gc
import psutil
import platform
import yaml
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import os
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "libs"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

from src.layer_1_voice_interface.wake_word_handler import WakeWordHandler
from src.layer_1_voice_interface.text_to_speech.speech_to_text import SpeechToText
from src.layer_1_voice_interface.tts_client import TTSClient
from src.layer_2_agentic_reasoning.llm_runner import LLMRunner
from src.layer_2_agentic_reasoning.context_manager import ContextManager


@dataclass
class DetailedMetrics:
    """Chi tiết metrics cho từng thời điểm"""
    timestamp: float
    relative_time: float
    phase: str
    active_components: List[str]
    
    # CPU Metrics
    cpu_percent: float
    cpu_count: int
    cpu_freq: float
    load_avg: Tuple[float, float, float]
    
    # Memory Metrics (System)
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    
    # Memory Metrics (Process)
    process_rss_mb: float  # Resident Set Size
    process_vms_mb: float  # Virtual Memory Size
    process_percent: float
    
    # GPU Metrics
    gpu_count: int = 0
    gpu_memory_used_mb: List[float] = field(default_factory=list)
    gpu_memory_total_mb: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)


@dataclass
class ComponentProfile:
    """Profile chi tiết của từng component"""
    name: str
    phase: str
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    
    # Memory Impact
    ram_before_gb: float = 0.0
    ram_after_gb: float = 0.0
    ram_delta_mb: float = 0.0
    
    process_before_mb: float = 0.0
    process_after_mb: float = 0.0
    process_delta_mb: float = 0.0
    
    # GPU Impact
    gpu_before_mb: List[float] = field(default_factory=list)
    gpu_after_mb: List[float] = field(default_factory=list)
    gpu_delta_mb: List[float] = field(default_factory=list)
    
    # CPU Impact
    cpu_before: float = 0.0
    cpu_after: float = 0.0


@dataclass
class PhaseMarker:
    """Marker cho từng giai đoạn"""
    name: str
    start_time: float
    end_time: float = 0.0
    description: str = ""
    color: str = "blue"


class DetailedPerformanceMonitor:
    """Monitor chi tiết performance"""
    
    def __init__(self):
        self.metrics: List[DetailedMetrics] = []
        self.component_profiles: List[ComponentProfile] = []
        self.phase_markers: List[PhaseMarker] = []
        
        self.start_time = time.time()
        self.baseline_duration = 10.0  # 10 giây đo baseline
        self.monitoring = False
        self.monitor_task = None
        
        self.process = psutil.Process()
        self.current_phase = "STARTING"
        self.active_components: List[str] = []
        
        # Baseline metrics
        self.baseline_metrics = None
        
        # GPU setup
        self.gpu_available = GPU_AVAILABLE
        self.nvml_available = NVML_AVAILABLE
        
        if self.gpu_available:
            try:
                self.gpus = GPUtil.getGPUs()
                print(f"🎮 Found {len(self.gpus)} GPU(s)")
                for i, gpu in enumerate(self.gpus):
                    print(f"  GPU {i}: {gpu.name} ({gpu.memoryTotal}MB VRAM)")
            except Exception as e:
                print(f"⚠️ GPU detection error: {e}")
                self.gpu_available = False
                self.gpus = []
        else:
            self.gpus = []
    
    def get_current_metrics(self) -> DetailedMetrics:
        """Lấy metrics hiện tại"""
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        try:
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        except:
            cpu_freq = 0
        
        try:
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        except:
            load_avg = (0, 0, 0)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_used_gb = memory.used / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        ram_percent = memory.percent
        
        # Process memory
        try:
            proc_memory = self.process.memory_info()
            process_rss_mb = proc_memory.rss / (1024**2)
            process_vms_mb = proc_memory.vms / (1024**2)
            process_percent = self.process.memory_percent()
        except:
            process_rss_mb = process_vms_mb = process_percent = 0
        
        # GPU metrics
        gpu_count = 0
        gpu_memory_used_mb = []
        gpu_memory_total_mb = []
        gpu_utilization = []
        gpu_temperature = []
        
        if self.gpu_available and self.gpus:
            try:
                current_gpus = GPUtil.getGPUs()
                gpu_count = len(current_gpus)
                
                for gpu in current_gpus:
                    gpu_memory_used_mb.append(gpu.memoryUsed)
                    gpu_memory_total_mb.append(gpu.memoryTotal)
                    gpu_utilization.append(gpu.load * 100)
                    gpu_temperature.append(gpu.temperature)
                    
            except Exception as e:
                print(f"⚠️ GPU monitoring error: {e}")
        
        return DetailedMetrics(
            timestamp=current_time,
            relative_time=relative_time,
            phase=self.current_phase,
            active_components=self.active_components.copy(),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            load_avg=load_avg,
            ram_total_gb=ram_total_gb,
            ram_used_gb=ram_used_gb,
            ram_available_gb=ram_available_gb,
            ram_percent=ram_percent,
            process_rss_mb=process_rss_mb,
            process_vms_mb=process_vms_mb,
            process_percent=process_percent,
            gpu_count=gpu_count,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature
        )
    
    def start_monitoring(self):
        """Bắt đầu monitoring"""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        print(f"🔍 Bắt đầu monitoring chi tiết (GPU: {self.gpu_available})")
    
    async def stop_monitoring(self):
        """Dừng monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Loop monitoring liên tục"""
        try:
            while self.monitoring:
                metrics = self.get_current_metrics()
                self.metrics.append(metrics)
                await asyncio.sleep(0.1)  # 100ms intervals
        except asyncio.CancelledError:
            pass
    
    def set_phase(self, phase_name: str, description: str = "", color: str = "blue"):
        """Thiết lập giai đoạn mới"""
        current_time = time.time() - self.start_time
        
        # Kết thúc phase trước
        if self.phase_markers and self.phase_markers[-1].end_time == 0:
            self.phase_markers[-1].end_time = current_time
        
        # Bắt đầu phase mới
        self.current_phase = phase_name
        marker = PhaseMarker(
            name=phase_name,
            start_time=current_time,
            description=description,
            color=color
        )
        self.phase_markers.append(marker)
        
        print(f"📍 Phase: {phase_name} - {description}")
    
    def start_component_profiling(self, component_name: str) -> ComponentProfile:
        """Bắt đầu profile component"""
        current_metrics = self.get_current_metrics()
        
        profile = ComponentProfile(
            name=component_name,
            phase=self.current_phase,
            start_time=current_metrics.relative_time,
            ram_before_gb=current_metrics.ram_used_gb,
            process_before_mb=current_metrics.process_rss_mb,
            gpu_before_mb=current_metrics.gpu_memory_used_mb.copy(),
            cpu_before=current_metrics.cpu_percent
        )
        
        if component_name not in self.active_components:
            self.active_components.append(component_name)
        
        print(f"🔬 Profiling {component_name}: RAM {profile.ram_before_gb:.1f}GB, "
              f"Process {profile.process_before_mb:.1f}MB")
        
        return profile
    
    def end_component_profiling(self, profile: ComponentProfile):
        """Kết thúc profile component"""
        current_metrics = self.get_current_metrics()
        
        profile.end_time = current_metrics.relative_time
        profile.duration = profile.end_time - profile.start_time
        profile.ram_after_gb = current_metrics.ram_used_gb
        profile.process_after_mb = current_metrics.process_rss_mb
        profile.gpu_after_mb = current_metrics.gpu_memory_used_mb.copy()
        profile.cpu_after = current_metrics.cpu_percent
        
        # Tính delta
        profile.ram_delta_mb = (profile.ram_after_gb - profile.ram_before_gb) * 1024
        profile.process_delta_mb = profile.process_after_mb - profile.process_before_mb
        
        if profile.gpu_before_mb and profile.gpu_after_mb:
            profile.gpu_delta_mb = [
                after - before for after, before in 
                zip(profile.gpu_after_mb, profile.gpu_before_mb)
            ]
        
        self.component_profiles.append(profile)
        
        print(f"📊 {profile.name} completed: {profile.duration:.2f}s, "
              f"RAM +{profile.ram_delta_mb:.1f}MB, "
              f"Process +{profile.process_delta_mb:.1f}MB"
              + (f", GPU +{profile.gpu_delta_mb}MB" if profile.gpu_delta_mb else ""))
    
    async def measure_baseline(self):
        """Đo baseline hệ thống trong 10 giây"""
        print(f"📏 Đo baseline hệ thống trong {self.baseline_duration}s...")
        self.set_phase("BASELINE", "Hệ thống không có ELSSA", "gray")
        
        baseline_samples = []
        for i in range(int(self.baseline_duration * 10)):  # 10 samples/second
            baseline_samples.append(self.get_current_metrics())
            await asyncio.sleep(0.1)
        
        # Tính baseline trung bình
        avg_ram = sum(m.ram_used_gb for m in baseline_samples) / len(baseline_samples)
        avg_process = sum(m.process_rss_mb for m in baseline_samples) / len(baseline_samples)
        avg_cpu = sum(m.cpu_percent for m in baseline_samples) / len(baseline_samples)
        
        self.baseline_metrics = {
            "ram_gb": avg_ram,
            "process_mb": avg_process,
            "cpu_percent": avg_cpu,
            "samples": len(baseline_samples)
        }
        
        print(f"📏 Baseline established: RAM {avg_ram:.2f}GB, "
              f"Process {avg_process:.1f}MB, CPU {avg_cpu:.1f}%")


class DetailedELSSATest:
    """Test ELSSA với measurements chi tiết"""
    
    def __init__(self):
        self.monitor = DetailedPerformanceMonitor()
        self.config = self._load_config()
        
        # Components
        self.wake_handler: Optional[WakeWordHandler] = None
        self.asr: Optional[SpeechToText] = None
        self.tts: Optional[TTSClient] = None
        self.llm_runner: Optional[LLMRunner] = None
        self.context_manager: Optional[ContextManager] = None
        
        # Test parameters
        self.test_query = "What is the capital of Vietnam?"
        
        # Events
        self.wake_detected = asyncio.Event()
    
    def _load_config(self):
        """Load configuration"""
        try:
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            return {"silence_timeout": 3, "n_ctx": 2048}
    
    def print_system_info(self):
        """In thông tin hệ thống"""
        print("💻 THÔNG TIN HỆ THỐNG")
        print("=" * 60)
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"CPU: {platform.processor()}")
        print(f"CPU Cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical")
        
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total / (1024**3):.1f} GB total")
        print(f"RAM Available: {memory.available / (1024**3):.1f} GB")
        
        if self.monitor.gpu_available:
            print(f"GPU: {len(self.monitor.gpus)} device(s)")
            for i, gpu in enumerate(self.monitor.gpus):
                print(f"  GPU {i}: {gpu.name} ({gpu.memoryTotal}MB VRAM)")
        else:
            print("GPU: Không có")
        print()
    
    async def test_phase_1_baseline(self):
        """Giai đoạn 1: Chỉ có hệ thống, chưa bật ELSSA"""
        await self.monitor.measure_baseline()
    
    async def test_phase_2_wakeword_only(self):
        """Giai đoạn 2: Chỉ bật WakeWordHandler"""
        self.monitor.set_phase("WAKE_ONLY", "Chỉ có WakeWordHandler", "green")
        
        # Profile WakeWordHandler
        wake_profile = self.monitor.start_component_profiling("WakeWordHandler")
        
        self.wake_handler = WakeWordHandler()
        self.wake_handler.register_callback(self._on_wake_detected)
        self.wake_handler.start()
        
        self.monitor.end_component_profiling(wake_profile)
        
        # Chạy trong 5 giây
        print("👂 WakeWordHandler đang chạy trong 5s...")
        await asyncio.sleep(5.0)
        
        print("✅ Phase 2 hoàn thành")
    
    async def _on_wake_detected(self):
        """Callback wake word"""
        self.wake_detected.set()
    
    async def test_phase_3_load_components(self):
        """Giai đoạn 3: Load ASR, TTS, LLM (không chạy)"""
        self.monitor.set_phase("LOAD_COMPONENTS", "Load ASR, TTS, LLM", "blue")
        
        # Dừng WakeWordHandler
        if self.wake_handler:
            self.wake_handler.stop()
            self.wake_handler = None
        
        # 1. Load ContextManager
        print("\n🔧 Loading ContextManager...")
        context_profile = self.monitor.start_component_profiling("ContextManager_Load")
        self.context_manager = ContextManager(
            context_dir="data/test_context",
            max_context_length=10
        )
        await self.context_manager.start_new_session()
        self.monitor.end_component_profiling(context_profile)
        
        # 2. Load LLMRunner
        print("\n🤖 Loading LLMRunner...")
        llm_profile = self.monitor.start_component_profiling("LLMRunner_Load")
        self.llm_runner = LLMRunner()
        self.llm_runner.launch()
        await asyncio.sleep(3)  # Chờ model load
        self.monitor.end_component_profiling(llm_profile)
        
        # 3. Load TTS with error handling
        print("\n🔊 Loading TTS...")
        tts_profile = self.monitor.start_component_profiling("TTS_Load")
        try:
            self.tts = TTSClient()
            self.tts.launch()  # Start server and load models
            await asyncio.sleep(3)  # Wait for server startup
            print("✅ TTS loaded successfully")
        except Exception as e:
            print(f"⚠️ TTS loading failed: {e}")
            self.tts = None  # Mark as unavailable
        self.monitor.end_component_profiling(tts_profile)
        
        # 4. Load ASR
        print("\n🎤 Loading ASR...")
        asr_profile = self.monitor.start_component_profiling("ASR_Load")
        self.asr = SpeechToText(silence_threshold=self.config.get('silence_timeout', 3))
        self.monitor.end_component_profiling(asr_profile)
        
        print("✅ Phase 3 hoàn thành - Components loaded (TTS may have failed)")
    
    async def test_phase_4_asr_start(self):
        """Giai đoạn 4: Start ASR"""
        self.monitor.set_phase("ASR_START", "Start ASR listening", "yellow")
        
        print("\n🎤 Phase 4: Starting ASR...")
        asr_start_profile = self.monitor.start_component_profiling("ASR_Start")
        
        if self.asr:
            self.asr.start()
            await asyncio.sleep(2.0)  # Simulate listening time
            if self.asr.is_running:
                await self.asr.stop_async()
        
        self.monitor.end_component_profiling(asr_start_profile)
        print("✅ Phase 4 hoàn thành - ASR đã chạy")
    
    async def test_phase_5_llm_process(self):
        """Giai đoạn 5: LLM Processing"""
        self.monitor.set_phase("LLM_PROCESS", "LLM processing", "purple")
        
        print("\n🤖 Phase 5: LLM Processing...")
        llm_process_profile = self.monitor.start_component_profiling("LLM_Process")
        
        await self.context_manager.add_message("user", self.test_query)
        context_messages = await self.context_manager.get_conversation_context()
        
        stream_response = self.llm_runner.chat(context_messages)
        full_response = ""
        for chunk in stream_response:
            full_response += chunk
        
        await self.context_manager.add_message("assistant", full_response)
        self.monitor.end_component_profiling(llm_process_profile)
        
        print(f"🤖 LLM Response: {full_response[:100]}...")
        print("✅ Phase 5 hoàn thành - LLM đã xử lý")
        
        return full_response
    
    async def test_phase_6_tts_synthesis(self, response_text: str):
        """Giai đoạn 6: TTS Synthesis"""
        self.monitor.set_phase("TTS_SYNTHESIS", "TTS synthesis", "orange")
        
        print("\n🔊 Phase 6: TTS Synthesis...")
        tts_synthesis_profile = self.monitor.start_component_profiling("TTS_Synthesis")
        
        if self.tts is None:
            print("⚠️ TTS not available, skipping synthesis")
            self.monitor.end_component_profiling(tts_synthesis_profile)
            return {"completed": False, "skipped": True}
        
        try:
            result = await self.tts.speak_async(
                response_text,
                play_audio=True,
                interruptible=False
            )
            
            self.monitor.end_component_profiling(tts_synthesis_profile)
            print("✅ Phase 6 hoàn thành - TTS đã tổng hợp")
            return result
            
        except Exception as e:
            print(f"⚠️ TTS synthesis failed: {e}")
            self.monitor.end_component_profiling(tts_synthesis_profile)
            return {"completed": False, "error": str(e)}

    async def test_individual_components_separate(self):
        """Test từng component riêng biệt - TÁCH BIỆT hoàn toàn"""
        print("\n" + "="*60)
        print("🔄 SHUTDOWN HỆ THỐNG VÀ TEST RIÊNG TỪNG COMPONENT")
        print("="*60)
        
        # Shutdown tất cả
        await self.cleanup_components()
        await asyncio.sleep(2.0)  # Đợi cleanup hoàn tất
        gc.collect()
        
        # ===== TEST RIÊNG WAKEWORDHANDLER =====
        self.monitor.set_phase("INDIVIDUAL_WAKE", "Test riêng WakeWordHandler", "lightgreen")
        print("\n🎯 Test riêng: WakeWordHandler")
        
        individual_wake_profile = self.monitor.start_component_profiling("WakeWord_Individual")
        wake_handler_individual = WakeWordHandler()
        wake_handler_individual.start()
        await asyncio.sleep(3.0)  # Test trong 3s
        wake_handler_individual.stop()
        self.monitor.end_component_profiling(individual_wake_profile)
        
        # Enhanced cleanup
        del wake_handler_individual
        gc.collect()
        await asyncio.sleep(1.0)  # Đợi cleanup hoàn tất
        
        # ===== TEST RIÊNG ASR =====
        self.monitor.set_phase("INDIVIDUAL_ASR", "Test riêng ASR", "lightblue")
        print("\n🎯 Test riêng: ASR")
        
        individual_asr_profile = self.monitor.start_component_profiling("ASR_Individual")
        asr_individual = SpeechToText(silence_threshold=3)
        asr_individual.start()
        await asyncio.sleep(2.0)  # Simulate listening
        if asr_individual.is_running:
            await asr_individual.stop_async()
        self.monitor.end_component_profiling(individual_asr_profile)
        
        # Enhanced ASR cleanup
        await self._enhanced_asr_cleanup(asr_individual)
        del asr_individual
        gc.collect()
        await asyncio.sleep(1.0)
        
        # ===== TEST RIÊNG TTS =====
        self.monitor.set_phase("INDIVIDUAL_TTS", "Test riêng TTS", "gold")
        print("\n🎯 Test riêng: TTS")
        
        individual_tts_profile = self.monitor.start_component_profiling("TTS_Individual")
        try:
            tts_individual = TTSClient()
            tts_individual.launch()  # Start isolated server
            await asyncio.sleep(3)  # Wait for server startup
            
            # Try synthesis
            await tts_individual.speak_async("Hello, this is a test.", play_audio=True, interruptible=False)
            print("✅ TTS individual test completed successfully")
            
        except Exception as e:
            print(f"⚠️ TTS individual test failed: {e}")
            tts_individual = None
            
        self.monitor.end_component_profiling(individual_tts_profile)
        
        # Enhanced TTS cleanup
        if tts_individual:
            await self._enhanced_individual_tts_cleanup(tts_individual)
            del tts_individual
        gc.collect()
        await asyncio.sleep(1.0)
        
        # ===== TEST RIÊNG LLM =====
        self.monitor.set_phase("INDIVIDUAL_LLM", "Test riêng LLM", "plum")
        print("\n🎯 Test riêng: LLM")
        
        individual_llm_profile = self.monitor.start_component_profiling("LLM_Individual")
        
        # Setup context riêng
        context_individual = ContextManager(context_dir="data/test_context", max_context_length=5)
        await context_individual.start_new_session()
        
        llm_individual = LLMRunner()
        llm_individual.launch()
        await asyncio.sleep(2)  # Wait for model load
        
        # Process simple query
        await context_individual.add_message("user", "What is 2+2?")
        context_messages = await context_individual.get_conversation_context()
        
        stream_response = llm_individual.chat(context_messages)
        response = ""
        for chunk in stream_response:
            response += chunk
        
        await context_individual.add_message("assistant", response)
        await context_individual.end_current_session()
        
        # Enhanced LLM cleanup
        await asyncio.to_thread(llm_individual.stop_server)
        self.monitor.end_component_profiling(individual_llm_profile)
        
        del context_individual, llm_individual
        gc.collect()
        await asyncio.sleep(1.0)
        
        # Final comprehensive cleanup after all individual tests
        await self._final_individual_cleanup()
        
        print("\n✅ Hoàn thành test riêng từng component với enhanced cleanup")

    async def test_phase_3_full_pipeline(self):
        """REMOVED - Replaced with separate phases"""
        pass

    async def test_full_conversation_flow(self):
        """REMOVED - Replaced with separate phases"""
        pass

    async def test_individual_component_usage(self):
        """REMOVED - Replaced with separate individual tests"""
        pass

    async def cleanup_components(self):
        """Cleanup tất cả components với enhanced memory management"""
        self.monitor.set_phase("CLEANUP", "Dọn dẹp components", "black")
        
        print("\n🧹 Enhanced cleaning up components...")
        
        cleanup_tasks = []
        
        # Stop wake word handler
        if self.wake_handler:
            try:
                self.wake_handler.stop()
            except:
                pass
        
        # Enhanced ASR cleanup
        if self.asr and self.asr.is_running:
            cleanup_tasks.append(self.asr.stop_async())
        
        # Enhanced TTS cleanup with error handling
        if self.tts:
            cleanup_tasks.append(self._enhanced_tts_cleanup())
        
        # Enhanced LLM cleanup
        if self.llm_runner:
            cleanup_tasks.append(asyncio.create_task(asyncio.to_thread(self.llm_runner.stop_server)))
        
        # Context manager cleanup
        if self.context_manager:
            cleanup_tasks.append(self.context_manager.end_current_session())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Force comprehensive cleanup
        await self._force_memory_cleanup()
        
        print("✅ Enhanced cleanup completed")
        
        # End final phase
        if self.monitor.phase_markers and self.monitor.phase_markers[-1].end_time == 0:
            current_time = time.time() - self.monitor.start_time
            self.monitor.phase_markers[-1].end_time = current_time
    
    async def _enhanced_tts_cleanup(self):
        """Enhanced TTS cleanup with process termination"""
        try:
            print("🧹 Enhanced TTS cleanup starting...")
            
            # Close TTS client (terminates subprocess)
            if self.tts:
                await asyncio.to_thread(self.tts.close)
            
            print("✅ Enhanced TTS cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in enhanced TTS cleanup: {e}")
    
    def _cleanup_gpu_memory(self):
        """Cleanup GPU memory using PyTorch"""
        try:
            import torch
            if torch.cuda.is_available():
                print("🎮 Clearing CUDA cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"🎮 GPU memory freed - Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        except Exception as e:
            print(f"⚠️ Error cleaning GPU memory: {e}")
    
    async def _force_memory_cleanup(self):
        """Force comprehensive memory cleanup"""
        try:
            print("🧹 Force memory cleanup...")
            
            # Clear all component references
            self.wake_handler = None
            self.asr = None
            self.tts = None
            self.llm_runner = None
            self.context_manager = None
            
            # Enhanced audio buffer cleanup
            try:
                # Clear any remaining audio arrays in numpy's cache
                import numpy as np
                # Force cleanup of numpy temporary arrays
                
                # Clear Python's internal caches
                import sys
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                
                print("🧹 Audio buffer caches cleared")
                
            except Exception as e:
                print(f"⚠️ Error clearing audio caches: {e}")
            
            # Force garbage collection multiple times
            for i in range(3):
                collected = gc.collect()
                print(f"🧹 GC cycle {i+1}: collected {collected} objects")
                await asyncio.sleep(0.2)
            
            # Additional GPU cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    # Force synchronization to complete cleanup
                    torch.cuda.synchronize()
                    print("🎮 Enhanced GPU cleanup completed")
            except:
                pass
            
            # Small delay to allow cleanup
            await asyncio.sleep(1.0)
            
            print("✅ Force memory cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in force cleanup: {e}")
    
    def generate_detailed_charts(self):
        """Tạo charts chi tiết với matplotlib"""
        if not self.monitor.metrics:
            print("⚠️ Không có data để vẽ charts")
            return
        
        # Setup figure với nhiều subplots
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('ELSSA Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        times = [m.relative_time for m in self.monitor.metrics]
        
        # Chart 1: System RAM Usage
        ax1 = axes[0, 0]
        ram_used = [m.ram_used_gb for m in self.monitor.metrics]
        
        if self.monitor.baseline_metrics:
            baseline_ram = self.monitor.baseline_metrics['ram_gb']
            ram_delta = [ram - baseline_ram for ram in ram_used]
            ax1.plot(times, ram_delta, 'b-', linewidth=2, label=f'RAM Delta from Baseline')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        else:
            ax1.plot(times, ram_used, 'b-', linewidth=2, label='System RAM Used (GB)')
        
        ax1.set_ylabel('Memory (GB)')
        ax1.set_title('System RAM Usage Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Process Memory Usage
        ax2 = axes[0, 1]
        process_rss = [m.process_rss_mb for m in self.monitor.metrics]
        
        if self.monitor.baseline_metrics:
            baseline_process = self.monitor.baseline_metrics['process_mb']
            process_delta = [rss - baseline_process for rss in process_rss]
            ax2.plot(times, process_delta, 'r-', linewidth=2, label=f'Process RSS Delta (MB)')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        else:
            ax2.plot(times, process_rss, 'r-', linewidth=2, label='Process RSS (MB)')
        
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Process Memory Usage Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: CPU Usage
        ax3 = axes[1, 0]
        cpu_usage = [m.cpu_percent for m in self.monitor.metrics]
        ax3.plot(times, cpu_usage, 'g-', linewidth=2, label='CPU Usage (%)')
        
        if self.monitor.baseline_metrics:
            baseline_cpu = self.monitor.baseline_metrics['cpu_percent']
            ax3.axhline(y=baseline_cpu, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_cpu:.1f}%)')
        
        ax3.set_ylabel('CPU Usage (%)')
        ax3.set_title('CPU Usage Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: GPU Memory (if available)
        ax4 = axes[1, 1]
        if self.monitor.gpu_available and any(m.gpu_memory_used_mb for m in self.monitor.metrics):
            gpu_memory = [m.gpu_memory_used_mb[0] if m.gpu_memory_used_mb else 0 for m in self.monitor.metrics]
            ax4.plot(times, gpu_memory, 'm-', linewidth=2, label='GPU VRAM (MB)')
            ax4.set_ylabel('GPU Memory (MB)')
            ax4.set_title('GPU VRAM Usage Over Time')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('GPU VRAM Usage (N/A)')
        ax4.grid(True, alpha=0.3)
        
        # Chart 5: Component Timeline
        ax5 = axes[2, 0]
        if self.monitor.component_profiles:
            component_names = list(set(p.name for p in self.monitor.component_profiles))
            y_positions = range(len(component_names))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))
            name_to_y = {name: i for i, name in enumerate(component_names)}
            
            legend_handles = []
            legend_labels = []
            
            for profile in self.monitor.component_profiles:
                y_pos = name_to_y[profile.name]
                color = colors[y_pos]
                
                # Create bar without text annotation
                bar = ax5.barh(y_pos, profile.duration, left=profile.start_time, 
                        height=0.6, color=color, alpha=0.7)
                
                # Add to legend only once per component
                if profile.name not in [label.split(' (')[0] for label in legend_labels]:
                    legend_handles.append(bar[0])
                    legend_labels.append(f"{profile.name} (+{profile.ram_delta_mb:.0f}MB)")
            
            ax5.set_yticks(y_positions)
            ax5.set_yticklabels(component_names, fontsize=8)
            ax5.set_xlabel('Time (seconds)')
            ax5.set_title('Component Initialization Timeline')
            
            # Add legend below the chart
            ax5.legend(legend_handles, legend_labels, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'No Component Data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Component Timeline (No Data)')
        ax5.grid(True, alpha=0.3)
        
        # Chart 6: Phase Timeline
        ax6 = axes[2, 1]
        if self.monitor.phase_markers:
            phase_colors = {
                'BASELINE': 'gray',
                'WAKE_ONLY': 'green', 
                'LOAD_COMPONENTS': 'blue',
                'ASR_START': 'yellow',
                'LLM_PROCESS': 'purple',
                'TTS_SYNTHESIS': 'orange',
                'INDIVIDUAL_WAKE': 'lightgreen',
                'INDIVIDUAL_ASR': 'lightblue', 
                'INDIVIDUAL_TTS': 'gold',
                'INDIVIDUAL_LLM': 'plum',
                'CLEANUP': 'black'
            }
            
            legend_handles = []
            legend_labels = []
            
            for i, phase in enumerate(self.monitor.phase_markers):
                if phase.end_time > 0:
                    duration = phase.end_time - phase.start_time
                    color = phase_colors.get(phase.name, phase.color)
                    
                    # Create bar without text inside
                    bar = ax6.barh(i, duration, left=phase.start_time, 
                            height=0.6, color=color, alpha=0.7)
                    
                    # Add to legend
                    legend_handles.append(bar[0])
                    legend_labels.append(f"{phase.name} ({duration:.1f}s)")
            
            ax6.set_yticks(range(len(self.monitor.phase_markers)))
            phase_labels = [f"Phase {i+1}" for i in range(len(self.monitor.phase_markers))]
            ax6.set_yticklabels(phase_labels, fontsize=8)
            ax6.set_xlabel('Time (seconds)')
            ax6.set_title('Test Phase Timeline (6 Phases + Individual Tests)')
            
            # Add legend below the chart
            ax6.legend(legend_handles, legend_labels, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
        else:
            ax6.text(0.5, 0.5, 'No Phase Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Phase Timeline (No Data)')
        ax6.grid(True, alpha=0.3)
        
        # Add phase markers to all charts with updated colors
        for ax in [ax1, ax2, ax3, ax4]:
            for phase in self.monitor.phase_markers:
                if phase.end_time > 0:
                    ax.axvspan(phase.start_time, phase.end_time, alpha=0.1, 
                              color=phase_colors.get(phase.name, phase.color))
                    ax.axvline(x=phase.start_time, color=phase_colors.get(phase.name, phase.color), 
                              linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        
        # Adjust layout to make space for legends
        plt.subplots_adjust(bottom=0.15)
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_performance.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed chart saved: {filename}")
        
        try:
            plt.show()
        except:
            print("⚠️ Cannot display chart (no GUI), saved to file instead")
    
    def generate_detailed_report(self):
        """Tạo báo cáo chi tiết"""
        total_duration = time.time() - self.monitor.start_time
        
        # Phân tích memory
        memory_analysis = {}
        if self.monitor.metrics and self.monitor.baseline_metrics:
            ram_values = [m.ram_used_gb for m in self.monitor.metrics]
            process_values = [m.process_rss_mb for m in self.monitor.metrics]
            cpu_values = [m.cpu_percent for m in self.monitor.metrics]
            
            baseline_ram = self.monitor.baseline_metrics['ram_gb']
            baseline_process = self.monitor.baseline_metrics['process_mb']
            baseline_cpu = self.monitor.baseline_metrics['cpu_percent']
            
            memory_analysis = {
                "baseline": {
                    "system_ram_gb": baseline_ram,
                    "process_ram_mb": baseline_process,
                    "cpu_percent": baseline_cpu
                },
                "peak": {
                    "system_ram_gb": max(ram_values),
                    "process_ram_mb": max(process_values),
                    "cpu_percent": max(cpu_values)
                },
                "delta_from_baseline": {
                    "system_ram_gb": max(ram_values) - baseline_ram,
                    "process_ram_mb": max(process_values) - baseline_process,
                    "cpu_percent": max(cpu_values) - baseline_cpu
                },
                "average": {
                    "system_ram_gb": sum(ram_values) / len(ram_values),
                    "process_ram_mb": sum(process_values) / len(process_values),
                    "cpu_percent": sum(cpu_values) / len(cpu_values)
                }
            }
        
        # Phân tích GPU
        gpu_analysis = {"available": False}
        if self.monitor.gpu_available and any(m.gpu_memory_used_mb for m in self.monitor.metrics):
            gpu_values = [m.gpu_memory_used_mb[0] if m.gpu_memory_used_mb else 0 for m in self.monitor.metrics]
            gpu_analysis = {
                "available": True,
                "peak_vram_mb": max(gpu_values),
                "average_vram_mb": sum(gpu_values) / len(gpu_values),
                "gpu_count": len(self.monitor.gpus)
            }
        
        # Phân tích components
        component_analysis = {}
        for profile in self.monitor.component_profiles:
            if profile.name not in component_analysis:
                component_analysis[profile.name] = []
            
            component_analysis[profile.name].append({
                "phase": profile.phase,
                "duration": profile.duration,
                "ram_delta_mb": profile.ram_delta_mb,
                "process_delta_mb": profile.process_delta_mb,
                "gpu_delta_mb": profile.gpu_delta_mb,
                "cpu_before": profile.cpu_before,
                "cpu_after": profile.cpu_after
            })
        
        # Phân tích phases
        phase_analysis = []
        for phase in self.monitor.phase_markers:
            if phase.end_time > 0:
                phase_analysis.append({
                    "name": phase.name,
                    "description": phase.description,
                    "start_time": phase.start_time,
                    "end_time": phase.end_time,
                    "duration": phase.end_time - phase.start_time
                })
        
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "type": "Detailed ELSSA Performance Test",
                "total_duration": total_duration,
                "baseline_duration": self.monitor.baseline_duration
            },
            "system_info": {
                "os": f"{platform.system()} {platform.release()}",
                "cpu_cores": psutil.cpu_count(),
                "cpu_physical": psutil.cpu_count(logical=False),
                "total_ram_gb": psutil.virtual_memory().total / (1024**3),
                "gpu_available": self.monitor.gpu_available,
                "gpu_count": len(self.monitor.gpus) if self.monitor.gpus else 0
            },
            "memory_analysis": memory_analysis,
            "gpu_analysis": gpu_analysis,
            "component_analysis": component_analysis,
            "phase_analysis": phase_analysis,
            "insights": self._generate_insights()
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"detailed_performance_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report, report_file
    
    def _generate_insights(self):
        """Tạo insights từ data"""
        insights = []
        
        if self.monitor.baseline_metrics and self.monitor.metrics:
            ram_values = [m.ram_used_gb for m in self.monitor.metrics]
            baseline_ram = self.monitor.baseline_metrics['ram_gb']
            max_ram_delta = max(ram_values) - baseline_ram
            
            if max_ram_delta > 1.0:
                insights.append(f"High RAM usage: +{max_ram_delta:.1f}GB above baseline")
            
            process_values = [m.process_rss_mb for m in self.monitor.metrics]
            baseline_process = self.monitor.baseline_metrics['process_mb']
            max_process_delta = max(process_values) - baseline_process
            
            if max_process_delta > 500:
                insights.append(f"High process memory: +{max_process_delta:.0f}MB above baseline")
        
        # Component insights
        memory_heavy_components = []
        for profile in self.monitor.component_profiles:
            if profile.ram_delta_mb > 100:
                memory_heavy_components.append(f"{profile.name} (+{profile.ram_delta_mb:.0f}MB)")
        
        if memory_heavy_components:
            insights.append(f"Memory-heavy components: {', '.join(memory_heavy_components)}")
        
        return insights
    
    def print_summary(self, report: dict, report_file: str):
        """In tóm tắt kết quả"""
        print("\n" + "=" * 80)
        print("📊 KẾT QUẢ DETAILED PERFORMANCE TEST")
        print("=" * 80)
        print(f"📄 Report file: {report_file}")
        print(f"⏱️ Total duration: {report['test_info']['total_duration']:.2f}s")
        print(f"📏 Baseline duration: {report['test_info']['baseline_duration']:.1f}s")
        
        memory_analysis = report.get('memory_analysis', {})
        if memory_analysis:
            baseline = memory_analysis.get('baseline', {})
            peak = memory_analysis.get('peak', {})
            delta = memory_analysis.get('delta_from_baseline', {})
            
            print(f"\n💾 MEMORY ANALYSIS:")
            print(f"  Baseline: RAM {baseline.get('system_ram_gb', 0):.2f}GB, "
                  f"Process {baseline.get('process_ram_mb', 0):.1f}MB")
            print(f"  Peak:     RAM {peak.get('system_ram_gb', 0):.2f}GB, "
                  f"Process {peak.get('process_ram_mb', 0):.1f}MB")
            print(f"  Delta:    RAM +{delta.get('system_ram_gb', 0):.2f}GB, "
                  f"Process +{delta.get('process_ram_mb', 0):.1f}MB")
        
        gpu_analysis = report.get('gpu_analysis', {})
        if gpu_analysis.get('available'):
            print(f"\n🎮 GPU ANALYSIS:")
            print(f"  Peak VRAM: {gpu_analysis.get('peak_vram_mb', 0):.0f}MB")
            print(f"  Average VRAM: {gpu_analysis.get('average_vram_mb', 0):.0f}MB")
        
        print(f"\n🎯 COMPONENT SUMMARY:")
        component_analysis = report.get('component_analysis', {})
        for name, profiles in component_analysis.items():
            total_ram = sum(p.get('ram_delta_mb', 0) for p in profiles)
            total_process = sum(p.get('process_delta_mb', 0) for p in profiles)
            avg_duration = sum(p.get('duration', 0) for p in profiles) / len(profiles)
            print(f"  • {name}: {avg_duration:.2f}s avg, "
                  f"RAM +{total_ram:.0f}MB, Process +{total_process:.0f}MB")
        
        print(f"\n💡 INSIGHTS:")
        for insight in report.get('insights', []):
            print(f"  • {insight}")
        
        print(f"\n🏁 Test completed!")
    
    async def run_detailed_test(self):
        """Chạy test chi tiết theo 6 giai đoạn + individual tests"""
        print("🧪 DETAILED ELSSA PERFORMANCE TEST - 6 PHASES")
        print("=" * 80)
        
        self.print_system_info()
        
        try:
            # Start monitoring
            self.monitor.start_monitoring()
            
            # ===== 6 GIAI ĐOẠN CHÍNH =====
            print("\n🎯 BẮT ĐẦU 6 GIAI ĐOẠN CHÍNH")
            print("="*50)
            
            # Phase 1: Baseline (System only)
            print("\n📍 PHASE 1: Hệ thống không có ELSSA")
            await self.test_phase_1_baseline()
            
            # Phase 2: WakeWord only
            print("\n📍 PHASE 2: Chỉ có WakeWordHandler")
            await self.test_phase_2_wakeword_only()
            
            # Phase 3: Load components
            print("\n📍 PHASE 3: Load ASR, TTS, LLM")
            await self.test_phase_3_load_components()
            
            # Phase 4: Start ASR
            print("\n📍 PHASE 4: Start ASR")
            await self.test_phase_4_asr_start()
            
            # Phase 5: LLM Process
            print("\n📍 PHASE 5: LLM Processing")
            response_text = await self.test_phase_5_llm_process()
            
            # Phase 6: TTS Synthesis
            print("\n📍 PHASE 6: TTS Synthesis")
            await self.test_phase_6_tts_synthesis(response_text)
            
            print("\n✅ HOÀN THÀNH 6 GIAI ĐOẠN CHÍNH")
            
            # ===== INDIVIDUAL COMPONENT TESTS =====
            await self.test_individual_components_separate()
            
        except Exception as e:
            print(f"💥 Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup_components()
            await self.monitor.stop_monitoring()
            
            # Generate reports and charts
            report, report_file = self.generate_detailed_report()
            self.generate_detailed_charts()
            
            # Print summary
            self.print_summary(report, report_file)

    async def _enhanced_asr_cleanup(self, asr_instance):
        """Enhanced ASR cleanup to free Whisper model memory"""
        try:
            print("🧹 Enhanced ASR cleanup starting...")
            
            # Stop ASR if still running
            if asr_instance.is_running:
                await asr_instance.stop_async()
            
            # Clear Whisper model reference
            if hasattr(asr_instance, 'asr'):
                del asr_instance.asr
            
            # Clear audio buffers
            if hasattr(asr_instance, 'audio_buffer'):
                asr_instance.audio_buffer.clear()
            if hasattr(asr_instance, '_all_text'):
                asr_instance._all_text.clear()
            
            print("✅ Enhanced ASR cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in enhanced ASR cleanup: {e}")
    
    async def _enhanced_individual_tts_cleanup(self, tts_instance):
        """Enhanced TTS cleanup for individual tests"""
        try:
            print("🧹 Enhanced individual TTS cleanup starting...")
            
            # Close TTS client (terminates subprocess)
            await asyncio.to_thread(tts_instance.close)
            
            print("✅ Enhanced individual TTS cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in enhanced individual TTS cleanup: {e}")
    
    async def _final_individual_cleanup(self):
        """Final comprehensive cleanup after all individual tests"""
        try:
            print("🧹 Final individual test cleanup...")
            
            # Force garbage collection multiple times
            for i in range(3):
                gc.collect()
                await asyncio.sleep(0.5)
            
            # GPU cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("🎮 Final GPU memory cleanup completed")
            except:
                pass
            
            print("✅ Final individual test cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in final individual cleanup: {e}")


async def main():
    """Main function"""
    test = DetailedELSSATest()
    await test.run_detailed_test()


if __name__ == "__main__":
    asyncio.run(main()) 