#!/usr/bin/env python3

import os
import sys
import time
import json
import argparse
import logging
import subprocess
import threading
from per_core_dqn_controller import PerCoreDQNController
from baseline_controllers import OndemandController, StaticController, ReactiveController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dqn_with_vms.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DQN_with_VMs")

class VMPerformanceMonitor:
    """Monitors VM performance metrics and feeds them to the frequency controller"""
    
    def __init__(self, vm_names, monitor_interval=5):
        self.vm_names = vm_names
        self.monitor_interval = monitor_interval
        self.metrics = {}
        self.running = False
        self.monitor_thread = None
        logger.info(f"VM Performance Monitor initialized for VMs: {', '.join(vm_names)}")
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("VM performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("VM performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                for vm_name in self.vm_names:
                    # Get VM metrics using virsh domstats
                    self._collect_vm_metrics(vm_name)
                
                # Wait for next collection cycle
                time.sleep(self.monitor_interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1)
    
    def _collect_vm_metrics(self, vm_name):
        """Collect metrics for a specific VM"""
        try:
            # Run virsh domstats to get various VM stats
            cmd = ["virsh", "domstats", vm_name, "--cpu", "--memory", "--balloon", "--block", "--net"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the output
            metrics = self._parse_domstats(result.stdout, vm_name)
            
            # Store in metrics dictionary
            self.metrics[vm_name] = metrics
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error collecting metrics for VM {vm_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error collecting metrics for VM {vm_name}: {str(e)}")
    
    def _parse_domstats(self, output, vm_name):
        """Parse virsh domstats output into a structured format"""
        metrics = {
            "cpu": {},
            "memory": {},
            "block": {},
            "net": {}
        }
        
        # Simple parsing of key-value pairs
        current_section = None
        for line in output.strip().split("\n"):
            if line.startswith(vm_name):
                # Main VM section header
                continue
            
            if ".cpu." in line:
                current_section = "cpu"
            elif ".memory." in line:
                current_section = "memory"
            elif ".block." in line:
                current_section = "block"
            elif ".net." in line:
                current_section = "net"
            
            if current_section:
                try:
                    # Extract key-value
                    parts = line.strip().split("=")
                    if len(parts) == 2:
                        key = parts[0].split(".")[-1]  # Get the last part of the key
                        value = parts[1]
                        
                        # Try to convert to numeric if possible
                        try:
                            value = float(value)
                            if value.is_integer():
                                value = int(value)
                        except:
                            pass
                        
                        metrics[current_section][key] = value
                except:
                    pass
        
        return metrics
    
    def get_aggregated_metrics(self):
        """Get aggregated performance metrics across all VMs"""
        if not self.metrics:
            return {}
        
        aggregated = {
            "throughput": 0,
            "iops": 0,
            "latency": 0,
            "cpu_usage": 0
        }
        
        vm_count = len(self.metrics)
        
        for vm_name, vm_metrics in self.metrics.items():
            # Sum up network throughput (if available)
            for net_key, net_value in vm_metrics.get("net", {}).items():
                if "rx_bytes" in net_key or "tx_bytes" in net_key:
                    aggregated["throughput"] += net_value
            
            # Sum up disk I/O
            for block_key, block_value in vm_metrics.get("block", {}).items():
                if "rd_ops" in block_key or "wr_ops" in block_key:
                    aggregated["iops"] += block_value
            
            # CPU usage
            if "usage" in vm_metrics.get("cpu", {}):
                aggregated["cpu_usage"] += vm_metrics["cpu"]["usage"]
        
        # Average CPU usage across VMs
        if vm_count > 0:
            aggregated["cpu_usage"] = aggregated["cpu_usage"] / vm_count
        
        return aggregated


def run_experiment(controller_type, model_path=None, vm_count=2, vm_base_name="myvm", run_time=1800):
    """Run an experiment with VMs and the specified CPU controller"""
    
    logger.info(f"Starting experiment with controller: {controller_type}")
    
    # 1. First, check if VMs are running and create VM list
    vm_names = [f"{vm_base_name}{i+1}" for i in range(vm_count)]
    running_vms = []
    
    for vm in vm_names:
        try:
            # Check if VM exists
            result = subprocess.run(["virsh", "domstate", vm], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0 and "running" in result.stdout:
                running_vms.append(vm)
                logger.info(f"VM {vm} is running")
            else:
                logger.warning(f"VM {vm} not running or not found")
        except Exception as e:
            logger.error(f"Error checking VM {vm}: {str(e)}")
    
    if not running_vms:
        logger.error("No running VMs found! Please start VMs first.")
        return
    
    # 2. Initialize the VM performance monitor
    monitor = VMPerformanceMonitor(running_vms)
    monitor.start_monitoring()
    
    # 3. Initialize the appropriate controller
    controller = None
    
    try:
        if controller_type == "dqn":
            if not model_path:
                logger.error("Model path required for DQN controller")
                return
            
            controller = PerCoreDQNController()
            
            # If in inference mode, load the model
            controller.agent.load(model_path)
            controller.agent.epsilon = 0.01  # Low exploration for inference
            
            # Start the controller inference loop in a separate thread
            controller_thread = threading.Thread(
                target=controller.run,
                kwargs={"model_path": model_path, "run_time": run_time}
            )
            controller_thread.daemon = True
            controller_thread.start()
            
            # Periodically feed VM metrics to the controller
            start_time = time.time()
            while time.time() - start_time < run_time:
                # Get aggregated VM metrics
                vm_metrics = monitor.get_aggregated_metrics()
                
                # Update controller with these metrics
                controller.update_performance_metrics(vm_metrics)
                
                # Log current state
                if vm_metrics:
                    logger.info(f"VM Metrics: throughput={vm_metrics.get('throughput', 0)}, " +
                               f"iops={vm_metrics.get('iops', 0)}, " +
                               f"cpu_usage={vm_metrics.get('cpu_usage', 0)}")
                
                # Wait before next update
                time.sleep(5)
            
            controller_thread.join(timeout=10)
            
        elif controller_type == "ondemand":
            controller = OndemandController()
            controller.run(run_time=run_time)
            
        elif controller_type == "static":
            # Static at 2600 MHz (or whatever max frequency is)
            controller = StaticController(frequency=2600000)
            controller.run(run_time=run_time)
            
        elif controller_type == "reactive":
            controller = ReactiveController()
            controller.run(run_time=run_time)
            
        else:
            logger.error(f"Unknown controller type: {controller_type}")
            return
        
    except KeyboardInterrupt:
        logger.info("Experiment stopped by user")
    
    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}")
    
    finally:
        # Clean up resources
        logger.info("Cleaning up...")
        
        if monitor:
            monitor.stop_monitoring()
        
        if controller:
            controller.cleanup()
        
        logger.info("Experiment complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with VMs and CPU frequency controller")
    parser.add_argument("controller", choices=["dqn", "ondemand", "static", "reactive"], 
                        help="Controller type to use")
    parser.add_argument("--model", help="Path to DQN model file (required for DQN controller)")
    parser.add_argument("--vms", type=int, default=2, help="Number of VMs to use")
    parser.add_argument("--vm-name", default="myvm", help="Base name for VMs")
    parser.add_argument("--run-time", type=int, default=1800, help="Run time in seconds")
    
    args = parser.parse_args()
    
    # Check if running with sudo
    if os.geteuid() != 0:
        logger.error("This script must be run with sudo privileges")
        sys.exit(1)
    
    run_experiment(
        controller_type=args.controller,
        model_path=args.model,
        vm_count=args.vms,
        vm_base_name=args.vm_name,
        run_time=args.run_time
    )
