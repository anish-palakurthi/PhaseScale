#!/bin/bash

# Script to clear all contents of the data directory while preserving the structure
# This will help reset the environment between experiments

# Define the data directory
DATA_DIR="./data"

# Check if the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Clearing contents of data directory..."

# Clear metrics directory
if [ -d "$DATA_DIR/metrics" ]; then
    echo "Clearing metrics directory..."
    find "$DATA_DIR/metrics" -mindepth 1 -delete
    # Recreate the directory if it was completely removed
    mkdir -p "$DATA_DIR/metrics"
fi

# Clear workloads directory
if [ -d "$DATA_DIR/workloads" ]; then
    echo "Clearing workloads directory..."
    find "$DATA_DIR/workloads" -mindepth 1 -delete
    # Recreate the directory if it was completely removed
    mkdir -p "$DATA_DIR/workloads"
fi

# Clear run directories (these are named with timestamps)
echo "Clearing run directories..."
find "$DATA_DIR" -maxdepth 1 -type d -name "run_*" -exec rm -rf {} \;

# Clear processed directory
if [ -d "$DATA_DIR/processed" ]; then
    echo "Clearing processed directory..."
    find "$DATA_DIR/processed" -mindepth 1 -delete
    # Recreate the directory if it was completely removed
    mkdir -p "$DATA_DIR/processed"
fi

# Clear visualizations directory
if [ -d "$DATA_DIR/visualizations" ]; then
    echo "Clearing visualizations directory..."
    find "$DATA_DIR/visualizations" -mindepth 1 -delete
    # Recreate the directory if it was completely removed
    mkdir -p "$DATA_DIR/visualizations"
fi

echo "Data directory cleared successfully. Folder structure preserved."