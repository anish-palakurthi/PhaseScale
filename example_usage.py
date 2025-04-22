#!/usr/bin/env python3
"""
Example script demonstrating how to use the CPUDQN class
"""

import json
import time
import argparse
import subprocess
from cpu_dqn import CPUDQN

def get_cpu_metrics():
    """Collect CPU metrics using your existing host_metrics_daemon output"""
    # You can replace this with direct calls to your metrics collector
    # For demonstration, we'll use a sample file
    try:
        with open("metrics.jsonl", "r") as f:
            last_line = f.readlines()[-1]
        return json.loads(last_line)
    except:
        # Fallback to example data if file doesn't exist
        with open("example_metrics.json", "r") as f:
            return json.load(f)

def set_cpu_frequencies(frequencies):
    """Apply CPU frequencies using your existing CPU controller"""
    # In a real scenario, call your cpu_control.py methods
    print(f"Setting frequencies: {frequencies}")
    
    # Example of how you would actually set frequencies
    try:
        for cpu_id, freq_mhz in enumerate(frequencies):
            # Convert MHz to KHz for kernel interface
            freq_khz = freq_mhz * 1000
            subprocess.run(
                f"echo {freq_khz} | sudo tee /sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_setspeed",
                shell=True, check=False
            )
    except Exception as e:
        print(f"Error setting frequencies: {e}")
        # Continue anyway for demonstration

def calculate_reward(old_metrics, new_metrics):
    """Calculate reward based on energy efficiency and performance"""
    # Extract energy usage if available
    energy_reward = 0
    if 'rapl_energy' in new_metrics:
        old_energy = sum(old_metrics.get('rapl_energy', {}).get(pkg, {}).get('current_uj', 0) 
                         for pkg in ['package-0', 'package-1'])
        new_energy = sum(new_metrics.get('rapl_energy', {}).get(pkg, {}).get('current_uj', 0) 
                         for pkg in ['package-0', 'package-1'])
        energy_delta = new_energy - old_energy
        
        # Lower energy use = higher reward
        # Normalize to reasonable range
        energy_reward = -energy_delta / 1000000  # Negative because lower energy is better
    
    # Performance component based on high-utilization cores
    performance_reward = 0
    high_util_cores = [cpu for cpu in new_metrics['percpu'] if cpu['usage_percent'] > 80]
    for core in high_util_cores:
        # Reward higher frequencies for busy cores
        performance_reward += core['frequency_mhz'] / 1000  # Normalize
    
    # Combined reward (weighted sum)
    # Adjust weights based on your priorities
    reward = (0.7 * energy_reward) + (0.3 * performance_reward)
    
    return reward

def main():
    parser = argparse.ArgumentParser(description="CPUDQN Example")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run", action="store_true", help="Run with trained model")
    parser.add_argument("--model", default="./trained_dqn.h5", help="Model path")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    args = parser.parse_args()
    
    # Count CPUs in your system
    try:
        # Try to get actual CPU count
        import os
        cpu_count = len(os.listdir('/sys/devices/system/cpu'))
        if not 16 <= cpu_count <= 128:  # Sanity check
            cpu_count = 32  # Fallback to default
    except:
        cpu_count = 32  # Default value
    
    print(f"Detected {cpu_count} CPU cores")
    
    # Initialize the DQN
    if args.run:
        # Load trained model for inference
        dqn = CPUDQN(n_cpus=cpu_count, model_path=args.model)
        print(f"Loaded trained model from {args.model}")
    else:
        # Fresh model for training
        dqn = CPUDQN(n_cpus=cpu_count)
        print("Initialized new model for training")
    
    # Main loop
    last_metrics = None
    
    for step in range(args.steps):
        print(f"\nStep {step+1}/{args.steps}")
        
        # Get current metrics
        metrics = get_cpu_metrics()
        
        # Get frequency recommendations
        frequencies = dqn.predict(metrics)
        print(f"DQN recommends frequencies: {frequencies[:5]}... (showing first 5)")
        
        # Apply frequencies
        set_cpu_frequencies(frequencies)
        
        # If we have previous metrics, calculate reward
        if last_metrics is not None:
            reward = calculate_reward(last_metrics, metrics)
            print(f"Reward: {reward:.4f}")
            
            # Provide feedback to DQN (only in training mode)
            if args.train:
                dqn.reward(reward)
        
        # Save metrics for next iteration
        last_metrics = metrics
        
        # Wait before next iteration
        time.sleep(1)
    
    # Save model if in training mode
    if args.train:
        dqn.save(args.model)
        print(f"\nTraining complete. Model saved to {args.model}")
    else:
        print("\nInference complete.")

if __name__ == "__main__":
    main()
