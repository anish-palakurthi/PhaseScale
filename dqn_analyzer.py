#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

def analyze_training_data(stats_file, output_dir):
    """Analyze DQN training statistics and generate visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training stats
    print(f"Loading training stats from {stats_file}...")
    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        print(f"Error loading stats file: {str(e)}")
        return
    
    print(f"Loaded {len(df)} training steps")
    
    # Plot reward per episode
    plt.figure(figsize=(12, 6))
    episode_rewards = df.groupby('episode')['reward'].sum()
    plt.plot(episode_rewards.index, episode_rewards.values)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rewards_per_episode.png'))
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    epsilon_values = df.groupby('episode')['epsilon'].first()
    plt.plot(epsilon_values.index, epsilon_values.values)
    plt.title('Epsilon Decay During Training')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'epsilon_decay.png'))
    
    # If it's per-core DQN, try to analyze frequency choices
    if 'frequencies' in df.columns:
        try:
            # Sample frequencies from the last episode
            last_episode = df['episode'].max()
            last_episode_data = df[df['episode'] == last_episode]
            
            # Parse frequency string into list of actual frequencies
            last_step = last_episode_data.iloc[-1]
            freq_list = [float(f) for f in last_step['frequencies'].split(',')]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(freq_list)), freq_list)
            plt.title(f'CPU Frequencies in Last Step of Training (Episode {last_episode})')
            plt.xlabel('CPU Core')
            plt.ylabel('Frequency (MHz)')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(output_dir, 'final_frequencies.png'))
        except Exception as e:
            print(f"Error analyzing frequency data: {str(e)}")
    
    print(f"Analysis complete. Visualizations saved to {output_dir}")

def analyze_inference_data(inference_file, output_dir):
    """Analyze DQN inference results and generate visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load inference stats
    print(f"Loading inference data from {inference_file}...")
    try:
        df = pd.read_csv(inference_file)
    except Exception as e:
        print(f"Error loading inference file: {str(e)}")
        return
    
    print(f"Loaded {len(df)} data points")
    
    # If it's per-core data (has cpu_id column)
    if 'cpu_id' in df.columns:
        # Plot frequency vs. time for each core (sampling a few cores)
        cores_to_plot = min(8, df['cpu_id'].max() + 1)  # Plot max 8 cores
        
        plt.figure(figsize=(12, 8))
        for core in range(cores_to_plot):
            core_data = df[df['cpu_id'] == core]
            plt.plot(core_data['timestamp'] - core_data['timestamp'].min(), 
                     core_data['frequency_mhz'], 
                     label=f'Core {core}')
        
        plt.title('CPU Frequency Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (MHz)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'frequency_vs_time.png'))
        
        # Plot usage vs. frequency for each core
        plt.figure(figsize=(12, 8))
        for core in range(cores_to_plot):
            core_data = df[df['cpu_id'] == core]
            plt.scatter(core_data['usage_percent'], 
                       core_data['frequency_mhz'], 
                       alpha=0.5,
                       label=f'Core {core}')
        
        plt.title('CPU Usage vs. Frequency')
        plt.xlabel('CPU Usage (%)')
        plt.ylabel('Frequency (MHz)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'usage_vs_frequency.png'))
        
        # Create heatmap of frequency distribution per core
        plt.figure(figsize=(14, 8))
        frequency_distribution = {}
        
        # Get unique cores and frequencies
        unique_cores = sorted(df['cpu_id'].unique())
        unique_freqs = sorted(df['frequency_mhz'].unique())
        
        # Create 2D histogram
        heatmap_data = np.zeros((len(unique_cores), len(unique_freqs)))
        
        for i, core in enumerate(unique_cores):
            core_data = df[df['cpu_id'] == core]
            for j, freq in enumerate(unique_freqs):
                heatmap_data[i, j] = len(core_data[core_data['frequency_mhz'] == freq])
        
        # Normalize by row
        row_sums = heatmap_data.sum(axis=1, keepdims=True)
        heatmap_data = heatmap_data / row_sums * 100
        
        plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Percentage of Time (%)')
        plt.title('Frequency Distribution by CPU Core')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('CPU Core')
        
        # Set x-axis ticks to actual frequency values
        plt.xticks(range(len(unique_freqs)), [f"{int(f)}" for f in unique_freqs], rotation=45)
        plt.yticks(range(len(unique_cores)), [f"Core {c}" for c in unique_cores])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_heatmap.png'))
    
    # If it's action-based data
    elif 'action' in df.columns:
        # Plot action histogram
        plt.figure(figsize=(12, 6))
        action_counts = df['action'].value_counts().sort_index()
        plt.bar(action_counts.index, action_counts.values)
        plt.title('Frequency of Actions')
        plt.xlabel('Action ID')
        plt.ylabel('Count')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, 'action_histogram.png'))
    
    print(f"Analysis complete. Visualizations saved to {output_dir}")

def compare_to_baseline(dqn_metrics, baseline_metrics, output_dir):
    """Compare DQN performance against baseline"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading DQN metrics from {dqn_metrics}...")
    try:
        dqn_df = pd.read_csv(dqn_metrics)
    except Exception as e:
        print(f"Error loading DQN metrics: {str(e)}")
        return
    
    print(f"Loading baseline metrics from {baseline_metrics}...")
    try:
        base_df = pd.read_csv(baseline_metrics)
    except Exception as e:
        print(f"Error loading baseline metrics: {str(e)}")
        return
    
    # Plot CPU frequency comparison (assuming we have cpu_id and frequency_mhz columns)
    if 'cpu_id' in dqn_df.columns and 'cpu_id' in base_df.columns:
        # Sample a few cores for comparison
        cores_to_plot = min(4, dqn_df['cpu_id'].max() + 1, base_df['cpu_id'].max() + 1)
        
        for core in range(cores_to_plot):
            plt.figure(figsize=(12, 6))
            
            # DQN data for this core
            dqn_core = dqn_df[dqn_df['cpu_id'] == core]
            dqn_time = dqn_core['timestamp'] - dqn_core['timestamp'].min()
            
            # Baseline data for this core
            base_core = base_df[base_df['cpu_id'] == core]
            base_time = base_core['timestamp'] - base_core['timestamp'].min()
            
            # Plot both
            plt.plot(dqn_time, dqn_core['frequency_mhz'], label='DQN', color='blue')
            plt.plot(base_time, base_core['frequency_mhz'], label='Baseline', color='red')
            
            plt.title(f'Core {core} Frequency: DQN vs. Baseline')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency (MHz)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'core_{core}_comparison.png'))
    
    # Calculate and compare average frequencies
    if 'frequency_mhz' in dqn_df.columns and 'frequency_mhz' in base_df.columns:
        # Calculate stats
        dqn_avg = dqn_df['frequency_mhz'].mean()
        base_avg = base_df['frequency_mhz'].mean()
        
        dqn_max = dqn_df['frequency_mhz'].max()
        base_max = base_df['frequency_mhz'].max()
        
        dqn_min = dqn_df['frequency_mhz'].min()
        base_min = base_df['frequency_mhz'].min()
        
        # Calculate energy savings (approximation)
        # Assuming energy ~ frequency^2
        energy_savings = 1 - (dqn_avg**2 / base_avg**2)
        
        # Create a summary bar chart
        plt.figure(figsize=(10, 6))
        metrics = ['Average Frequency (MHz)', 'Max Frequency (MHz)', 'Min Frequency (MHz)']
        dqn_values = [dqn_avg, dqn_max, dqn_min]
        base_values = [base_avg, base_max, base_min]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, dqn_values, width, label='DQN')
        plt.bar(x + width/2, base_values, width, label='Baseline')
        
        plt.title('Frequency Metrics Comparison')
        plt.xticks(x, metrics)
        plt.ylabel('Frequency (MHz)')
        plt.legend()
        plt.grid(True, axis='y')
        
        # Add values on top of bars
        for i, v in enumerate(dqn_values):
            plt.text(i - width/2, v + 10, f"{v:.1f}", ha='center')
        
        for i, v in enumerate(base_values):
            plt.text(i + width/2, v + 10, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_metrics_comparison.png'))
        
        # Create a summary text file
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write("DQN vs. Baseline Comparison Summary\n")
            f.write("===================================\n\n")
            f.write(f"DQN Average Frequency: {dqn_avg:.2f} MHz\n")
            f.write(f"Baseline Average Frequency: {base_avg:.2f} MHz\n")
            f.write(f"Frequency Reduction: {base_avg - dqn_avg:.2f} MHz ({(base_avg - dqn_avg) / base_avg * 100:.2f}%)\n\n")
            f.write(f"Estimated Energy Savings: {energy_savings * 100:.2f}%\n")
    
    print(f"Comparison analysis complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="DQN Training and Inference Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Training analysis command
    train_parser = subparsers.add_parser("train", help="Analyze training data")
    train_parser.add_argument("stats_file", help="Path to training stats CSV file")
    train_parser.add_argument("--output-dir", default="./training_analysis", help="Output directory for analysis")
    
    # Inference analysis command
    infer_parser = subparsers.add_parser("infer", help="Analyze inference data")
    infer_parser.add_argument("inference_file", help="Path to inference CSV file")
    infer_parser.add_argument("--output-dir", default="./inference_analysis", help="Output directory for analysis")
    
    # Comparison command
    compare_parser = subparsers.add_parser("compare", help="Compare DQN against baseline")
    compare_parser.add_argument("dqn_metrics", help="Path to DQN metrics CSV file")
    compare_parser.add_argument("baseline_metrics", help="Path to baseline metrics CSV file")
    compare_parser.add_argument("--output-dir", default="./comparison_analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    if args.command == "train":
        analyze_training_data(args.stats_file, args.output_dir)
    elif args.command == "infer":
        analyze_inference_data(args.inference_file, args.output_dir)
    elif args.command == "compare":
        compare_to_baseline(args.dqn_metrics, args.baseline_metrics, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
