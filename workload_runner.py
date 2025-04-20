#!/usr/bin/env python3

import os
import sys
import yaml
import time
import argparse
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("workload_runner.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("WorkloadRunner")

class WorkloadRunner:
    def __init__(self, config_file, output_dir="./workload_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded workload configuration from {config_file}")
    
    def run_workload(self, workload_type, workload_name):
        """Run a specific workload by type and name"""
        workload_path = f"{workload_type}.{workload_name}"
        workload_parts = workload_path.split('.')
        
        if len(workload_parts) != 2:
            logger.error(f"Invalid workload path: {workload_path}")
            return False
            
        workload_type, workload_name = workload_parts
        
        # Find the workload in config
        if workload_type not in self.config['workloads']:
            logger.error(f"Workload type {workload_type} not found in configuration")
            return False
            
        workload_list = self.config['workloads'][workload_type]
        workload = None
        
        for wl in workload_list:
            if wl['name'] == workload_name:
                workload = wl
                break
        
        if not workload:
            logger.error(f"Workload {workload_name} not found in {workload_type}")
            return False
        
        # Log the start of workload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting workload {workload_path} at {timestamp}")
        
        # Create log file for workload output
        log_file = f"{self.output_dir}/{workload_type}_{workload_name}_{timestamp}.log"
        
        # Run the workload
        if 'command' in workload:
            cmd = workload['command']
            try:
                with open(log_file, 'w') as f:
                    logger.info(f"Running command: {cmd}")
                    process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
                    
                    # Wait for completion or timeout
                    start_time = time.time()
                    while process.poll() is None:
                        if 'duration' in workload and (time.time() - start_time > workload['duration']):
                            process.terminate()
                            logger.info(f"Terminated workload {workload_path} after duration {workload['duration']}s")
                            break
                        time.sleep(1)
                    
                    if process.poll() is None:
                        process.wait(timeout=10)
                    
                    return_code = process.returncode
                    logger.info(f"Workload {workload_path} completed with return code {return_code}")
                    return return_code == 0
            except Exception as e:
                logger.error(f"Error running workload {workload_path}: {str(e)}")
                return False
        
        elif 'commands' in workload:
            processes = []
            try:
                for i, cmd in enumerate(workload['commands']):
                    cmd_log_file = f"{log_file}.{i}"
                    logger.info(f"Running command: {cmd}")
                    with open(cmd_log_file, 'w') as f:
                        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
                        processes.append(process)
                
                # Wait for completion or timeout
                start_time = time.time()
                while any(p.poll() is None for p in processes):
                    if 'duration' in workload and (time.time() - start_time > workload['duration']):
                        for p in processes:
                            if p.poll() is None:
                                p.terminate()
                        logger.info(f"Terminated workload {workload_path} after duration {workload['duration']}s")
                        break
                    time.sleep(1)
                
                # Ensure all processes are terminated
                for p in processes:
                    if p.poll() is None:
                        p.wait(timeout=10)
                
                return True
            except Exception as e:
                logger.error(f"Error running workload {workload_path}: {str(e)}")
                return False
        else:
            logger.error(f"No command specified for workload {workload_path}")
            return False
    
    def run_sequence(self, sequence_name):
        """Run a predefined sequence of workloads"""
        if 'sequences' not in self.config or sequence_name not in self.config['sequences']:
            logger.error(f"Sequence {sequence_name} not found in configuration")
            return False
        
        sequence = self.config['sequences'][sequence_name]
        logger.info(f"Starting workload sequence: {sequence_name}")
        
        for step in sequence:
            if 'workload' not in step:
                logger.error(f"Missing workload in sequence step: {step}")
                continue
            
            workload_path = step['workload']
            duration = step.get('duration', None)
            
            # Extract workload type and name
            workload_parts = workload_path.split('.')
            if len(workload_parts) != 2:
                logger.error(f"Invalid workload path in sequence: {workload_path}")
                continue
            
            workload_type, workload_name = workload_parts
            
            # Find the workload
            if workload_type not in self.config['workloads']:
                logger.error(f"Workload type {workload_type} not found in configuration")
                continue
            
            found = False
            for wl in self.config['workloads'][workload_type]:
                if wl['name'] == workload_name:
                    found = True
                    
                    # Override duration if specified in sequence
                    if duration is not None:
                        wl_copy = wl.copy()
                        wl_copy['duration'] = duration
                        
                        # Temporarily modify config to use custom duration
                        temp_config = self.config.copy()
                        temp_config['workloads'] = self.config['workloads'].copy()
                        temp_config['workloads'][workload_type] = [wl_copy]
                        old_config = self.config
                        self.config = temp_config
                        
                        self.run_workload(workload_type, workload_name)
                        
                        # Restore original config
                        self.config = old_config
                    else:
                        self.run_workload(workload_type, workload_name)
                    break
            
            if not found:
                logger.error(f"Workload {workload_name} not found in {workload_type}")
        
        logger.info(f"Completed workload sequence: {sequence_name}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predefined workloads or sequences")
    parser.add_argument("--config", default="workload_config.yaml", help="Workload configuration file")
    parser.add_argument("--output-dir", default="./workload_data", help="Directory to store workload outputs")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Run a single workload
    workload_parser = subparsers.add_parser("workload", help="Run a single workload")
    workload_parser.add_argument("workload_path", help="Workload path (type.name)")
    
    # Run a sequence
    sequence_parser = subparsers.add_parser("sequence", help="Run a workload sequence")
    sequence_parser.add_argument("sequence_name", help="Name of the sequence to run")
    
    args = parser.parse_args()
    
    runner = WorkloadRunner(args.config, args.output_dir)
    
    if args.command == "workload":
        workload_parts = args.workload_path.split('.')
        if len(workload_parts) != 2:
            logger.error(f"Invalid workload path: {args.workload_path}")
            sys.exit(1)
        
        workload_type, workload_name = workload_parts
        success = runner.run_workload(workload_type, workload_name)
        sys.exit(0 if success else 1)
    
    elif args.command == "sequence":
        success = runner.run_sequence(args.sequence_name)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)
