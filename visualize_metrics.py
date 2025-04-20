#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MetricsVisualizer")

class MetricsVisualizer:
    def __init__(self, data_file, output_dir="./visualizations"):
        self.data_file = data_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the data
        self.df = pd.read_csv(data_file)
        logger.info(f"Loaded data from {data_file} with {len(self.df)} rows")
        
        # Convert timestamp to datetime if it exists
        if "timestamp" in self.df.columns:
            try:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], format="%Y%m%d_%H%M%S")
            except:
                logger.warning("Could not convert timestamp column to datetime")
    
    def plot_cpu_usage(self):
        """Plot CPU usage over time"""
        if self.df.empty:
            logger.warning("No data to plot CPU usage")
            return
        
        # Find CPU usage columns
        cpu_cols = [col for col in self.df.columns if col.endswith("_percent")]
        
        if not cpu_cols:
            logger.warning("No CPU usage columns found")
            return
        
        plt.figure(figsize=(12, 6))
        
        x_col = "timestamp" if "timestamp" in self.df.columns else self.df.index
        
        for col in cpu_cols:
            plt.plot(self.df[x_col], self.df[col], label=col)
        
        plt.title("CPU Usage Over Time")
        plt.xlabel("Time")
        plt.ylabel("CPU Usage (%)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_file = os.path.join(self.output_dir, "cpu_usage.png")
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved CPU usage plot to {output_file}")
    
    def plot_memory_usage(self):
        """Plot memory usage over time"""
        if self.df.empty:
            logger.warning("No data to plot memory usage")
            return
        
        # Check if memory columns exist
        if "memory_percent" not in self.df.columns:
            logger.warning("Memory percent column not found")
            return
        
        plt.figure(figsize=(12, 6))
        
        x_col = "timestamp" if "timestamp" in self.df.columns else self.df.index
        
        plt.plot(self.df[x_col], self.df["memory_percent"], label="Memory Usage (%)", color="green")
        
        plt.title("Memory Usage Over Time")
        plt.xlabel("Time")
        plt.ylabel("Memory Usage (%)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_file = os.path.join(self.output_dir, "memory_usage.png")
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved memory usage plot to {output_file}")
    
    def plot_cpu_frequencies(self):
        """Plot CPU frequencies over time"""
        if self.df.empty:
            logger.warning("No data to plot CPU frequencies")
            return
        
        # Find CPU frequency columns
        freq_cols = [col for col in self.df.columns if col.endswith("_freq") and not col.endswith("_delta")]
        
        if not freq_cols:
            logger.warning("No CPU frequency columns found")
            return
        
        plt.figure(figsize=(12, 6))
        
        x_col = "timestamp" if "timestamp" in self.df.columns else self.df.index
        
        for col in freq_cols:
            plt.plot(self.df[x_col], self.df[col]/1000, label=f"{col} (MHz)")
        
        plt.title("CPU Frequencies Over Time")
        plt.xlabel("Time")
        plt.ylabel("Frequency (MHz)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_file = os.path.join(self.output_dir, "cpu_frequencies.png")
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved CPU frequencies plot to {output_file}")
    
    def plot_power_metrics(self):
        """Plot power-related metrics over time"""
        if self.df.empty:
            logger.warning("No data to plot power metrics")
            return
        
        # Find power columns
        power_cols = [col for col in self.df.columns if col.startswith("rapl_") and col.endswith("_per_sec")]
        
        if not power_cols:
            logger.warning("No power metrics columns found")
            return
        
        plt.figure(figsize=(12, 6))
        
        x_col = "timestamp" if "timestamp" in self.df.columns else self.df.index
        
        for col in power_cols:
            plt.plot(self.df[x_col], self.df[col]/1000000, label=f"{col} (W)")
        
        plt.title("Power Consumption Over Time")
        plt.xlabel("Time")
        plt.ylabel("Power (Watts)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_file = os.path.join(self.output_dir, "power_consumption.png")
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved power consumption plot to {output_file}")
    
    def generate_all_plots(self):
        """Generate all available plots"""
        self.plot_cpu_usage()
        self.plot_memory_usage()
        self.plot_cpu_frequencies()
        self.plot_power_metrics()
        
        logger.info(f"Generated all plots in {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed metrics")
    parser.add_argument("--data-file", required=True, help="CSV file with processed metrics")
    parser.add_argument("--output-dir", default="./visualizations", help="Directory to store visualizations")
    
    args = parser.parse_args()
    
    visualizer = MetricsVisualizer(args.data_file, args.output_dir)
    visualizer.generate_all_plots()
