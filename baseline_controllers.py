#!/usr/bin/env python3

import os
import json
import time
import logging
import argparse
import psutil
from cpu_control import CPUController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("baseline_controllers.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BaselineControllers")

class OndemandController:
    """Baseline controller using Linux's ondemand governor"""
    
    def __init__(self, output_dir="./ondemand_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CPU controller
        self.cpu_controller = CPUController()
        self.cpu_count = self.cpu_controller.cpu_count
        
        # Initialize logging
        self.stats_file = open(os.path.join(output_dir, "ondemand_stats.csv"), "w")
        self.stats_file.write("timestamp,cpu_id,frequency_mhz,usage_percent\n")
        
        logger.info(f"Ondemand controller initialized for {self.cpu_count} CPUs")
    
    def collect_metrics(self):
        """Collect current CPU metrics"""
        cpu_data = []
        
        # Get CPU info from psutil
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        
        for i in range(len(cpu_percent)):
            cpu_entry = {
                "cpu_id": i,
                "usage_percent": cpu_percent[i],
                "frequency_mhz": cpu_freq[i].current if i < len(cpu_freq) and cpu_freq[i] else 1200.0
            }
            cpu_data.append(cpu_entry)
        
        return cpu_data
    
    def run(self, run_time=3600):
        """Run the ondemand governor and collect stats"""
        logger.info(f"Starting ondemand controller for {run_time} seconds")
        
        # Set ondemand governor for all cores
        self.cpu_controller.set_governor("ondemand")
        logger.info("Set all CPUs to ondemand governor")
        
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < run_time:
                step += 1
                
                # Collect metrics
                cpu_data = self.collect_metrics()
                
                # Log metrics
                timestamp = time.time()
                for cpu_entry in cpu_data:
                    self.stats_file.write(f"{timestamp},{cpu_entry['cpu_id']},{cpu_entry['frequency_mhz']},{cpu_entry['usage_percent']}\n")
                self.stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step}, Elapsed: {time.time() - start_time:.1f}s")
                
                # Wait before next sampling
                time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        
        finally:
            self.stats_file.close()
            logger.info(f"Run completed, total steps: {step}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stats_file.close()
        logger.info("Controller cleaned up")


class StaticController:
    """Baseline controller using static frequency settings"""
    
    def __init__(self, frequency, output_dir="./static_results"):
        self.frequency = frequency  # in KHz
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CPU controller
        self.cpu_controller = CPUController()
        self.cpu_count = self.cpu_controller.cpu_count
        
        # Initialize logging
        self.stats_file = open(os.path.join(output_dir, f"static_{frequency//1000}mhz_stats.csv"), "w")
        self.stats_file.write("timestamp,cpu_id,frequency_mhz,usage_percent\n")
        
        logger.info(f"Static controller initialized for {self.cpu_count} CPUs at {frequency//1000} MHz")
    
    def collect_metrics(self):
        """Collect current CPU metrics"""
        cpu_data = []
        
        # Get CPU info from psutil
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        
        for i in range(len(cpu_percent)):
            cpu_entry = {
                "cpu_id": i,
                "usage_percent": cpu_percent[i],
                "frequency_mhz": cpu_freq[i].current if i < len(cpu_freq) and cpu_freq[i] else self.frequency//1000
            }
            cpu_data.append(cpu_entry)
        
        return cpu_data
    
    def run(self, run_time=3600):
        """Run the static frequency controller and collect stats"""
        logger.info(f"Starting static controller at {self.frequency//1000} MHz for {run_time} seconds")
        
        # Set userspace governor and fixed frequency for all cores
        self.cpu_controller.set_governor("userspace")
        
        # Set fixed frequency for all cores
        for cpu in range(self.cpu_count):
            self.cpu_controller.set_frequency(self.frequency, [cpu])
        
        logger.info(f"Set all CPUs to {self.frequency//1000} MHz")
        
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < run_time:
                step += 1
                
                # Collect metrics
                cpu_data = self.collect_metrics()
                
                # Log metrics
                timestamp = time.time()
                for cpu_entry in cpu_data:
                    self.stats_file.write(f"{timestamp},{cpu_entry['cpu_id']},{cpu_entry['frequency_mhz']},{cpu_entry['usage_percent']}\n")
                self.stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step}, Elapsed: {time.time() - start_time:.1f}s")
                
                # Wait before next sampling
                time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        
        finally:
            self.stats_file.close()
            logger.info(f"Run completed, total steps: {step}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stats_file.close()
        # Reset to default governor
        self.cpu_controller.set_governor("ondemand")
        logger.info("Controller cleaned up, CPU governor reset to ondemand")


class ReactiveController:
    """Baseline controller that reactively sets frequencies based on current load"""
    
    def __init__(self, output_dir="./reactive_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CPU controller
        self.cpu_controller = CPUController()
        self.cpu_count = self.cpu_controller.cpu_count
        
        # Get available frequency steps
        min_freq, max_freq = self.cpu_controller.get_min_max_freq()
        self.frequency_steps = [
            min_freq,  # Lowest frequency (e.g., 1200000 KHz)
            min_freq + (max_freq - min_freq) // 4,  # 25% step
            min_freq + (max_freq - min_freq) // 2,  # 50% step
            min_freq + 3 * (max_freq - min_freq) // 4,  # 75% step
            max_freq   # Highest frequency (e.g., 2600000 KHz)
        ]
        
        # Initialize logging
        self.stats_file = open(os.path.join(output_dir, "reactive_stats.csv"), "w")
        self.stats_file.write("timestamp,cpu_id,frequency_mhz,usage_percent\n")
        
        logger.info(f"Reactive controller initialized for {self.cpu_count} CPUs")
        logger.info(f"Available frequency steps: {[f//1000 for f in self.frequency_steps]} MHz")
    
    def collect_metrics(self):
        """Collect current CPU metrics"""
        cpu_data = []
        
        # Get CPU info from psutil
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        
        for i in range(len(cpu_percent)):
            cpu_entry = {
                "cpu_id": i,
                "usage_percent": cpu_percent[i],
                "frequency_mhz": cpu_freq[i].current if i < len(cpu_freq) and cpu_freq[i] else 1200.0
            }
            cpu_data.append(cpu_entry)
        
        return cpu_data
    
    def determine_frequency(self, usage):
        """Determine appropriate frequency based on CPU usage"""
        if usage < 20:
            return self.frequency_steps[0]  # Lowest for idle cores
        elif usage < 40:
            return self.frequency_steps[1]  # 25% step for light load
        elif usage < 60:
            return self.frequency_steps[2]  # 50% step for moderate load
        elif usage < 80:
            return self.frequency_steps[3]  # 75% step for heavy load
        else:
            return self.frequency_steps[4]  # Highest for very heavy load
    
    def run(self, run_time=3600):
        """Run the reactive controller and collect stats"""
        logger.info(f"Starting reactive controller for {run_time} seconds")
        
        # Set userspace governor for all cores
        self.cpu_controller.set_governor("userspace")
        logger.info("Set all CPUs to userspace governor")
        
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < run_time:
                step += 1
                
                # Collect metrics
                cpu_data = self.collect_metrics()
                
                # Set frequencies based on usage
                for cpu_entry in cpu_data:
                    cpu_id = cpu_entry["cpu_id"]
                    usage = cpu_entry["usage_percent"]
                    freq = self.determine_frequency(usage)
                    
                    # Apply new frequency
                    self.cpu_controller.set_frequency(freq, [cpu_id])
                    
                    # Update frequency in our data
                    cpu_entry["frequency_mhz"] = freq // 1000
                
                # Log metrics
                timestamp = time.time()
                for cpu_entry in cpu_data:
                    self.stats_file.write(f"{timestamp},{cpu_entry['cpu_id']},{cpu_entry['frequency_mhz']},{cpu_entry['usage_percent']}\n")
                self.stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step}, Elapsed: {time.time() - start_time:.1f}s")
                
                # Wait before next adjustment
                time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        
        finally:
            self.stats_file.close()
            logger.info(f"Run completed, total steps: {step}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stats_file.close()
        # Reset to default governor
        self.cpu_controller.set_governor("ondemand")
        logger.info("Controller cleaned up, CPU governor reset to ondemand")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline CPU Frequency Controllers")
    subparsers = parser.add_subparsers(dest="controller", help="Controller type")
    
    # Ondemand controller
    ondemand_parser = subparsers.add_parser("ondemand", help="Use Linux's ondemand governor")
    ondemand_parser.add_argument("--run-time", type=int, default=3600, help="Run time in seconds")
    ondemand_parser.add_argument("--output-dir", default="./ondemand_results", help="Output directory")
    
    # Static controller
    static_parser = subparsers.add_parser("static", help="Use static frequency")
    static_parser.add_argument("frequency", type=int, help="Frequency in MHz")
    static_parser.add_argument("--run-time", type=int, default=3600, help="Run time in seconds")
    static_parser.add_argument("--output-dir", default="./static_results", help="Output directory")
    
    # Reactive controller
    reactive_parser = subparsers.add_parser("reactive", help="Use reactive frequency control")
    reactive_parser.add_argument("--run-time", type=int, default=3600, help="Run time in seconds")
    reactive_parser.add_argument("--output-dir", default="./reactive_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        if args.controller == "ondemand":
            controller = OndemandController(output_dir=args.output_dir)
            controller.run(run_time=args.run_time)
        
        elif args.controller == "static":
            controller = StaticController(
                frequency=args.frequency * 1000,  # Convert MHz to KHz
                output_dir=args.output_dir
            )
            controller.run(run_time=args.run_time)
        
        elif args.controller == "reactive":
            controller = ReactiveController(output_dir=args.output_dir)
            controller.run(run_time=args.run_time)
        
        else:
            parser.print_help()
    
    finally:
        if 'controller' in locals():
            controller.cleanup()
