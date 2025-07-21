#!/usr/bin/env python3
"""
Script đơn giản để monitor resource usage khi chạy main.py
Đo memory, CPU, thời gian và lưu dữ liệu để vẽ biểu đồ
"""

import psutil
import subprocess
import time
import json
import os
import sys
from datetime import datetime


class ResourceMonitor:
    def __init__(self):
        self.baseline_memory = 0
        self.data = []
        self.start_time = 0
        self.process = None
        
    def get_baseline_memory(self):
        """Đo memory baseline trước khi chạy chương trình"""
        # Force garbage collection để có baseline chính xác
        import gc
        gc.collect()
        
        self.baseline_memory = psutil.virtual_memory().used
        print(f"📊 Baseline memory: {self.baseline_memory / 1024 / 1024:.2f} MB")
        
    def get_current_stats(self):
        """Lấy thông số hiện tại của hệ thống"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Memory do chương trình chiếm dụng (trừ baseline)
        program_memory = memory.used - self.baseline_memory
        
        stats = {
            'timestamp': time.time() - self.start_time,
            'total_memory_mb': memory.used / 1024 / 1024,
            'program_memory_mb': program_memory / 1024 / 1024,
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'available_memory_mb': memory.available / 1024 / 1024
        }
        
        # Nếu có process của main.py, lấy thêm thông tin riêng của nó
        if self.process and self.process.poll() is None:
            try:
                proc = psutil.Process(self.process.pid)
                proc_memory = proc.memory_info()
                stats['process_memory_mb'] = proc_memory.rss / 1024 / 1024
                stats['process_cpu_percent'] = proc.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        return stats
    
    def save_data(self, filename='resource_data.json'):
        """Lưu dữ liệu ra file JSON"""
        with open(filename, 'w') as f:
            json.dump({
                'baseline_memory_mb': self.baseline_memory / 1024 / 1024,
                'total_duration': time.time() - self.start_time,
                'measurements': self.data
            }, f, indent=2)
        print(f"💾 Đã lưu dữ liệu vào {filename}")
    
    def run_monitor(self, monitor_interval=1.0):
        """Chạy monitoring cho main.py"""
        print("🚀 Bắt đầu monitoring main.py...")
        
        # Đo baseline memory
        self.get_baseline_memory()
        
        # Bắt đầu đo thời gian
        self.start_time = time.time()
        
        # Khởi chạy main.py
        print("🎯 Đang khởi chạy main.py...")
        self.process = subprocess.Popen([sys.executable, 'main.py'])
        
        try:
            # Monitor loop
            while self.process.poll() is None:
                stats = self.get_current_stats()
                self.data.append(stats)
                
                # In thông tin real-time
                print(f"⏱️  {stats['timestamp']:.1f}s | "
                      f"🧠 Program: {stats['program_memory_mb']:.1f}MB | "
                      f"💻 CPU: {stats['cpu_percent']:.1f}%", end='\r')
                
                time.sleep(monitor_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Đang dừng monitoring...")
            self.process.terminate()
            self.process.wait()
            
        finally:
            # Lấy thống kê cuối cùng
            if self.process.poll() is not None:
                final_stats = self.get_current_stats()
                self.data.append(final_stats)
            
            # Lưu dữ liệu
            self.save_data()
            
            # In tóm tắt
            self.print_summary()
    
    def print_summary(self):
        """In tóm tắt kết quả"""
        if not self.data:
            return
            
        duration = self.data[-1]['timestamp']
        max_program_memory = max(d['program_memory_mb'] for d in self.data)
        avg_cpu = sum(d['cpu_percent'] for d in self.data) / len(self.data)
        max_cpu = max(d['cpu_percent'] for d in self.data)
        
        print(f"\n" + "="*50)
        print(f"📈 TÓM TẮT KẾT QUẢ MONITORING")
        print(f"="*50)
        print(f"⏱️  Thời gian chạy: {duration:.2f} giây")
        print(f"🧠 Memory tối đa (chương trình): {max_program_memory:.2f} MB")
        print(f"💻 CPU trung bình: {avg_cpu:.1f}%")
        print(f"💻 CPU tối đa: {max_cpu:.1f}%")
        print(f"📊 Số lần đo: {len(self.data)}")
        print(f"💾 Dữ liệu đã lưu: resource_data.json")


def main():
    print("🔍 ELSSA Resource Monitor")
    print("=" * 30)
    
    if not os.path.exists('main.py'):
        print("❌ Không tìm thấy main.py trong thư mục hiện tại!")
        return
    
    monitor = ResourceMonitor()
    
    try:
        monitor.run_monitor(monitor_interval=0.5)  # Đo mỗi 0.5 giây
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
    
    print("\n✅ Monitoring hoàn tất!")


if __name__ == "__main__":
    main() 