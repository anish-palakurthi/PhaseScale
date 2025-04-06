#!/usr/bin/env python3

import os
import time
import json
import subprocess
import psutil
from datetime import datetime
import argparse
import signal
import sys

class SystemMonitor:
    def __init__(self, output_dir, interval=1):
        self.output_dir = output_dir
        self.interval = interval
        self.running = True
        self.timestamp_format = "%Y%m%d_%H%M%S"
        self.session_id = datetime.now().strftime(self.timestamp_format)
        self.metrics_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.metrics_dir, exist_ok=True)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        print(f"Received signal {sig}, shutting down...")
        self.running = False
        
    def get_cpu_frequencies(self):
        """Get current CPU frequencies for all cores"""
        freqs = {}
        for cpu in range(psutil.cpu_count()):
            freq_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            if os.path.exists(freq_file):
                with open(freq_file, 'r') as f:
                    freqs[f"cpu{cpu}"] = int(f.read().strip())
        return freqs
        
    def get_cpu_governors(self):
        """Get current CPU governors for all cores"""
        governors = {}
        for cpu in range(psutil.cpu_count()):
            gov_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            if os.path.exists(gov_file):
                with open(gov_file, 'r') as f:
                    governors[f"cpu{cpu}"] = f.read().strip()
        return governors
        
    def get_vm_stats(self):
        """Get stats for all running VMs"""
        vm_stats = {}
        try:
            # Get list of running VMs
            result = subprocess.run(["virsh", "list", "--name"], 
                                   capture_output=True, text=True, check=False)
            vm_names = result.stdout.strip().split('\n')
            vm_names = [vm for vm in vm_names if vm]  # Filter out empty lines
            
            for vm in vm_names:
                # Get CPU stats for VM
                cpu_result = subprocess.run(["virsh", "domstats", vm, "--cpu"], 
                                         capture_output=True, text=True, check=False)
                
                # Get memory stats for VM
                mem_result = subprocess.run(["virsh", "domstats", vm, "--memory"], 
                                         capture_output=True, text=True, check=False)
                
                vm_stats[vm] = {
                    "cpu": cpu_result.stdout.strip(),
                    "memory": mem_result.stdout.strip()
                }
        except Exception as e:
            vm_stats["error"] = str(e)
        
        return vm_stats
    
    def get_system_metrics(self):
        """Collect all system metrics"""
        timestamp = datetime.now()
        
        metrics = {
            "timestamp": timestamp.strftime(self.timestamp_format),
            "cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
            "cpu_times": [x._asdict() for x in psutil.cpu_times(percpu=True)],
            "memory": psutil.virtual_memory()._asdict(),
            "cpu_freq": self.get_cpu_frequencies(),
            "cpu_governors": self.get_cpu_governors(),
            "vm_stats": self.get_vm_stats()
        }
        
        return metrics
    
    def start_monitoring(self):
        """Start the monitoring loop"""
        print(f"Starting system monitoring. Press Ctrl+C to stop.")
        print(f"Metrics will be saved to {self.metrics_dir}")
        
        count = 0
        while self.running:
            start_time = time.time()
            
            # Collect metrics
            metrics = self.get_system_metrics()
            
            # Save metrics to file
            filename = f"metrics_{metrics['timestamp']}.json"
            filepath = os.path.join(self.metrics_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            count += 1
            if count % 10 == 0:
                print(f"Collected {count} samples...")
            
            # Sleep for the remainder of the interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)
            
        print(f"Monitoring stopped. Collected {count} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System metrics collector")
    parser.add_argument("--output-dir", default="./data/metrics", 
                        help="Directory to store metrics")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Collection interval in seconds")
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.output_dir, args.interval)
    monitor.start_monitoring()
