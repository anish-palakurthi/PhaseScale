#!/usr/bin/env python3

import os
import time
import json
import signal
import logging
import argparse
import subprocess
import psutil
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metrics_collector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MetricsCollector")

class MetricsCollector:
    def __init__(self, output_dir="./metrics", interval_system=1.0, interval_detailed=10.0):
        """Initialize metrics collector with different collection intervals"""
        self.output_dir = output_dir
        self.interval_system = interval_system  # For system-wide metrics
        self.interval_detailed = interval_detailed  # For detailed/expensive metrics
        self.running = True
        self.timestamp_format = "%Y%m%d_%H%M%S"
        
        # Create session directory
        self.session_id = datetime.now().strftime(self.timestamp_format)
        self.metrics_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Setup worker threads
        self.system_collector_thread = None
        self.detailed_collector_thread = None
        
        logger.info(f"Initialized MetricsCollector with output directory: {self.metrics_dir}")
        logger.info(f"System metrics interval: {interval_system}s, Detailed metrics interval: {interval_detailed}s")
    
    def signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
    
    def collect_cpu_metrics(self):
        """Collect CPU-related metrics"""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
            "cpu_times": [x._asdict() for x in psutil.cpu_times(percpu=True)],
            "cpu_stats": psutil.cpu_stats()._asdict(),
            "cpu_freq": self._get_cpu_frequencies(),
            "cpu_governors": self._get_cpu_governors(),
            "cpu_load": os.getloadavg(),
        }
        
        # Collect CPU temperature if available
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    metrics["cpu_temp"] = temps
            except:
                pass
        
        return metrics
    
    def collect_memory_metrics(self):
        """Collect memory-related metrics"""
        metrics = {
            "virtual_memory": psutil.virtual_memory()._asdict(),
            "swap_memory": psutil.swap_memory()._asdict(),
        }
        return metrics
    
    def collect_io_metrics(self):
        """Collect I/O-related metrics"""
        metrics = {
            "disk_io": psutil.disk_io_counters(perdisk=True),
            "net_io": psutil.net_io_counters(pernic=True),
        }
        return metrics
    
    def collect_power_metrics(self):
        """Collect power consumption metrics via RAPL if available"""
        power_metrics = {}
        
        # Check for RAPL interface
        rapl_dir = "/sys/class/powercap/intel-rapl"
        if os.path.exists(rapl_dir):
            for domain in os.listdir(rapl_dir):
                if domain.startswith("intel-rapl:"):
                    domain_path = os.path.join(rapl_dir, domain)
                    energy_file = os.path.join(domain_path, "energy_uj")
                    
                    if os.path.exists(energy_file):
                        try:
                            with open(energy_file, 'r') as f:
                                power_metrics[domain] = int(f.read().strip())
                        except:
                            pass
        
        return power_metrics
    
    def collect_vm_metrics(self):
        """Collect VM-related metrics"""
        vm_metrics = {}
        
        try:
            # Get list of running VMs
            result = subprocess.run(["virsh", "list", "--name"], 
                                  capture_output=True, text=True, check=False)
            vm_names = result.stdout.strip().split('\n')
            vm_names = [vm for vm in vm_names if vm]  # Filter out empty lines
            
            for vm in vm_names:
                # Get various VM stats
                stats = {}
                
                # CPU stats
                cpu_result = subprocess.run(["virsh", "domstats", vm, "--cpu"], 
                                        capture_output=True, text=True, check=False)
                if cpu_result.returncode == 0:
                    stats["cpu"] = cpu_result.stdout.strip()
                
                # Memory stats
                mem_result = subprocess.run(["virsh", "domstats", vm, "--memory"], 
                                        capture_output=True, text=True, check=False)
                if mem_result.returncode == 0:
                    stats["memory"] = mem_result.stdout.strip()
                
                # Balloon stats (memory allocation)
                balloon_result = subprocess.run(["virsh", "domstats", vm, "--balloon"], 
                                            capture_output=True, text=True, check=False)
                if balloon_result.returncode == 0:
                    stats["balloon"] = balloon_result.stdout.strip()
                
                # Block device stats
                block_result = subprocess.run(["virsh", "domstats", vm, "--block"], 
                                          capture_output=True, text=True, check=False)
                if block_result.returncode == 0:
                    stats["block"] = block_result.stdout.strip()
                
                # Network interface stats
                net_result = subprocess.run(["virsh", "domstats", vm, "--interface"], 
                                        capture_output=True, text=True, check=False)
                if net_result.returncode == 0:
                    stats["network"] = net_result.stdout.strip()
                
                vm_metrics[vm] = stats
        except Exception as e:
            logger.error(f"Error collecting VM metrics: {str(e)}")
            vm_metrics["error"] = str(e)
        
        return vm_metrics
    
    def collect_perf_counters(self):
        """Collect hardware performance counters"""
        counters = {}
        
        try:
            # Run perf stat for a brief period to collect basic counters
            # Note: Requires 'perf' to be installed and available
            perf_events = "cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses"
            cmd = f"perf stat -e {perf_events} -a sleep 0.5"
            
            result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, text=True)
            output = result.stderr
            
            # Parse perf output
            for line in output.split('\n'):
                if any(event in line for event in perf_events.split(',')):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        value = parts[0].replace(',', '')
                        event = parts[-1]
                        try:
                            counters[event] = float(value)
                        except ValueError:
                            pass
        except Exception as e:
            logger.error(f"Error collecting performance counters: {str(e)}")
            counters["error"] = str(e)
        
        return counters
    
    def _get_cpu_frequencies(self):
        """Get current CPU frequencies for all cores"""
        freqs = {}
        for cpu in range(psutil.cpu_count()):
            freq_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            if os.path.exists(freq_file):
                try:
                    with open(freq_file, 'r') as f:
                        freqs[f"cpu{cpu}"] = int(f.read().strip())
                except:
                    pass
        return freqs
    
    def _get_cpu_governors(self):
        """Get current CPU governors for all cores"""
        governors = {}
        for cpu in range(psutil.cpu_count()):
            gov_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            if os.path.exists(gov_file):
                try:
                    with open(gov_file, 'r') as f:
                        governors[f"cpu{cpu}"] = f.read().strip()
                except:
                    pass
        return governors
    
    def collect_system_metrics(self):
        """Collect basic system-wide metrics (fast collection)"""
        timestamp = datetime.now()
        
        metrics = {
            "timestamp": timestamp.strftime(self.timestamp_format),
            "timestamp_epoch": time.time(),
            "cpu": self.collect_cpu_metrics(),
            "memory": self.collect_memory_metrics(),
            "io": self.collect_io_metrics(),
        }
        
        return metrics
    
    def collect_detailed_metrics(self):
        """Collect detailed metrics (slower collection)"""
        timestamp = datetime.now()
        
        metrics = {
            "timestamp": timestamp.strftime(self.timestamp_format),
            "timestamp_epoch": time.time(),
            "power": self.collect_power_metrics(),
            "vms": self.collect_vm_metrics(),
            "perf_counters": self.collect_perf_counters(),
        }
        
        return metrics
    
    def system_metrics_worker(self):
        """Worker function for collecting system metrics"""
        count = 0
        while self.running:
            start_time = time.time()
            
            # Collect basic metrics
            metrics = self.collect_system_metrics()
            
            # Save metrics to file
            filename = f"system_metrics_{metrics['timestamp']}.json"
            filepath = os.path.join(self.metrics_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(metrics, f)
            
            count += 1
            if count % 60 == 0:  # Log every ~minute
                logger.info(f"Collected {count} system metric samples...")
            
            # Sleep for the remainder of the interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval_system - elapsed)
            time.sleep(sleep_time)
    
    def detailed_metrics_worker(self):
        """Worker function for collecting detailed metrics"""
        count = 0
        while self.running:
            start_time = time.time()
            
            # Collect detailed metrics
            metrics = self.collect_detailed_metrics()
            
            # Save metrics to file
            filename = f"detailed_metrics_{metrics['timestamp']}.json"
            filepath = os.path.join(self.metrics_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(metrics, f)
            
            count += 1
            logger.info(f"Collected detailed metrics sample #{count}")
            
            # Sleep for the remainder of the interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval_detailed - elapsed)
            time.sleep(sleep_time)
    
    def start_collecting(self):
        """Start the metrics collection process"""
        logger.info(f"Starting metrics collection. Output directory: {self.metrics_dir}")
        
        # Start system metrics collector thread
        self.system_collector_thread = threading.Thread(target=self.system_metrics_worker)
        self.system_collector_thread.daemon = True
        self.system_collector_thread.start()
        
        # Start detailed metrics collector thread
        self.detailed_collector_thread = threading.Thread(target=self.detailed_metrics_worker)
        self.detailed_collector_thread.daemon = True
        self.detailed_collector_thread.start()
        
        logger.info("Metrics collection started. Press Ctrl+C to stop.")
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
            self.running = False
        
        # Wait for threads to finish
        self.system_collector_thread.join()
        self.detailed_collector_thread.join()
        
        logger.info("Metrics collection stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System metrics collector")
    parser.add_argument("--output-dir", default="./data/metrics", help="Directory to store metrics")
    parser.add_argument("--system-interval", type=float, default=1.0, help="System metrics collection interval in seconds")
    parser.add_argument("--detailed-interval", type=float, default=10.0, help="Detailed metrics collection interval in seconds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start the collector
    collector = MetricsCollector(
        output_dir=args.output_dir,
        interval_system=args.system_interval,
        interval_detailed=args.detailed_interval
    )
    
    collector.start_collecting()
