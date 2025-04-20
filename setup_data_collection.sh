#!/bin/bash

# Setup script for data collection dependencies

set -e

echo "=== Setting up data collection dependencies ==="

# Update package repositories
sudo apt-get update

# Install system monitoring tools
echo "Installing system monitoring tools..."
sudo apt-get install -y sysstat linux-tools-common linux-tools-generic linux-tools-$(uname -r) \
                        stress-ng fio sysbench iperf3

# Install Python dependencies
echo "Installing Python dependencies..."
sudo apt-get install -y python3-pip python3-venv

# Install Python packages
echo "Installing Python packages..."
pip3 install numpy pandas matplotlib seaborn scikit-learn \
           psutil pyyaml

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/metrics data/processed data/visualizations data/workloads

# Make scripts executable
chmod +x *.py *.sh

echo "=== Setup completed ==="
