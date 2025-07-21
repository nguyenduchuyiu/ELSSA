#!/usr/bin/env python3
"""
Simple script to plot data from monitoring
"""

import json
import matplotlib.pyplot as plt
import os


def plot_resource_data(filename='resource_data.json'):
    """Plots data from monitoring"""
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    # Read data
    with open(filename, 'r') as f:
        data = json.load(f)
    
    measurements = data['measurements']
    if not measurements:
        print("❌ No data to plot")
        return
    
    # Prepare data
    timestamps = [m['timestamp'] for m in measurements]
    program_memory = [m['program_memory_mb'] for m in measurements]
    cpu_percent = [m['cpu_percent'] for m in measurements]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('ELSSA Resource Usage Monitor', fontsize=16, fontweight='bold')
    
    # Memory plot
    ax1.plot(timestamps, program_memory, 'b-', linewidth=2, label='Program Memory')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Memory Usage')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # CPU plot
    ax2.plot(timestamps, cpu_percent, 'r-', linewidth=2, label='CPU Usage')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('CPU (%)')
    ax2.set_title('CPU Usage')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add summary information
    max_memory = max(program_memory)
    avg_cpu = sum(cpu_percent) / len(cpu_percent)
    duration = data['total_duration']
    
    # Text box with summary information
    summary_text = f"""Summary:
• Run time: {duration:.1f}s
• Max memory: {max_memory:.1f}MB
• Avg CPU: {avg_cpu:.1f}%
• Baseline memory: {data['baseline_memory_mb']:.1f}MB"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    output_file = 'resource_usage_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {output_file}")
    
    # Show plot
    plt.show()


def main():
    print("📊 Plotting Resource Usage")
    print("=" * 30)
    
    try:
        plot_resource_data()
    except ImportError:
        print("❌ Please install matplotlib: uv pip install matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()