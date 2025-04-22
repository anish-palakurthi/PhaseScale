# PhaseScale: ML-Based CPU Frequency Scaling

This document describes how to use the Deep Q-Network (DQN) controller for CPU frequency scaling in the PhaseScale project.

## Overview

The DQN controller learns to optimize CPU frequencies across multiple cores to balance energy efficiency and performance. It monitors workload phases and makes coordinated decisions across VMs.

There are two DQN implementations provided:

1. **dqn_controller.py**: A controller that takes discrete actions to change CPU frequencies
2. **per_core_dqn_controller.py**: A controller that directly outputs a frequency for each CPU core

## Installation

1. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Make sure you have proper permissions to control CPU frequencies:

```bash
sudo chmod +x *.py
```

## Usage Instructions

### Per-Core DQN Controller

This is the recommended approach for fine-grained control of per-core frequencies.

#### Training the Model

```bash
sudo python3 per_core_dqn_controller.py --train --episodes 20 --steps 500
```

Options:
- `--episodes`: Number of training episodes (default: 20)
- `--steps`: Steps per episode (default: 100)
- `--output-dir`: Directory to store results (default: ./per_core_dqn_results)

The training process:
1. Collects current CPU metrics (usage, frequency)
2. Predicts optimal frequency for each core
3. Applies these frequencies
4. Observes the results and calculates rewards
5. Updates the DQN model based on reward feedback

Training generates model weights files (*.h5) in the output directory.

#### Running the Trained Model

```bash
sudo python3 per_core_dqn_controller.py --run ./per_core_dqn_results/per_core_dqn_ep20.h5 --run-time 1800
```

Options:
- `--run`: Path to the trained model file
- `--run-time`: Duration to run in seconds (default: 3600)
- `--output-dir`: Directory to store results (default: ./per_core_dqn_results)

### Action-Based DQN Controller

This controller uses a simpler approach with discrete actions.

#### Training the Model

```bash
sudo python3 dqn_controller.py --train --episodes 20 --steps 500
```

Options are the same as the per-core version.

#### Running the Trained Model

```bash
sudo python3 dqn_controller.py --run ./dqn_results/dqn_model_ep20.h5 --run-time 1800
```

Options are the same as the per-core version.

## Baseline Controllers

To evaluate the DQN controllers, we provide several baseline controllers:

### Linux Ondemand Governor

```bash
sudo python3 baseline_controllers.py ondemand --run-time 1800
```

### Static Frequency

```bash
sudo python3 baseline_controllers.py static 2600 --run-time 1800
```

The parameter (2600) is the frequency in MHz.

### Reactive Controller

```bash
sudo python3 baseline_controllers.py reactive --run-time 1800
```

This controller reactively sets frequencies based on current usage levels without learning or prediction.

## Analysis Tools

The `dqn_analyzer.py` script provides various tools to analyze the performance of your controllers:

### Analyzing Training Progress

```bash
python3 dqn_analyzer.py train ./per_core_dqn_results/per_core_dqn_stats.csv
```

This generates visualizations of rewards and epsilon decay during training.

### Analyzing Inference Results

```bash
python3 dqn_analyzer.py infer ./per_core_dqn_results/per_core_dqn_inference.csv
```

This generates visualizations showing frequency choices, usage vs. frequency, etc.

### Comparing Against Baselines

```bash
python3 dqn_analyzer.py compare ./per_core_dqn_results/per_core_dqn_inference.csv ./ondemand_results/ondemand_stats.csv
```

This generates comparative visualizations and calculates energy savings.

## Integration with VM Manager

The DQN controllers can be integrated with the VM manager for real-world workload optimization:

1. First, deploy and start your VMs using the VM manager:

```bash
python3 master.py
```

2. Start the host metrics daemon:

```bash
sudo python3 host_metrics_daemon.py
```

3. Start the DQN controller (in inference mode with a pre-trained model):

```bash
sudo python3 per_core_dqn_controller.py --run ./per_core_dqn_results/per_core_dqn_ep20.h5
```

## How the DQN Controller Works

### State Representation

The DQN uses the following features for its state representation:
- Per-core CPU utilization
- Per-core current frequency
- Socket-level average utilization
- Usage variability (standard deviation)
- Usage trends (changes since previous measurement)
- Time-based features (for detecting daily patterns)

### Reward Function

The reward function balances multiple objectives:
1. **Energy Efficiency**: Reward lower frequencies, especially for idle cores
2. **Performance**: Reward higher frequencies for busy cores
3. **Stability**: Penalize frequent or large frequency changes
4. **Application Performance**: Reward improvements in application-level metrics

### Model Architecture

The per-core DQN uses a neural network with:
- Shared layers that process the system state
- Individual output heads for each CPU core
- Each head outputs Q-values for the available frequency options

### Frequency Options

The controller uses the following frequency options (in MHz):
- 1200 (lowest)
- 1550
- 1900
- 2250
- 2600 (highest)

## Troubleshooting

### Permission Issues

If you encounter permission issues:

```
Error: Could not set CPU frequency: [Errno 13] Permission denied
```

Ensure you're running with sudo and that the CPU scaling driver is properly configured.

### Model Not Found

If you see:

```
Error: Model file not found: ./per_core_dqn_results/per_core_dqn_ep20.h5
```

Make sure you've completed the training phase and the model file exists.

### TensorFlow Issues

If TensorFlow fails to initialize properly, try:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Advanced Configuration

### Customizing Reward Weights

Edit the `RewardCalculator` class in the controller file to adjust the weights:

```python
self.energy_weight = 0.5      # Increase for more energy savings
self.performance_weight = 2.0 # Increase for better performance
self.stability_weight = 0.1   # Increase for more stable frequencies
```

### Adding VM Performance Metrics

To incorporate VM-level performance metrics, implement the `update_performance_metrics` method to pass metrics from your VMs to the controller:

```python
controller = PerCoreDQNController()
# When VM performance metrics are available
controller.update_performance_metrics({
    'throughput': throughput_value,
    'latency': latency_value
})
```

### Changing DQN Hyperparameters

Edit the `PerCoreDQNAgent` class to adjust learning parameters:

```python
self.gamma = 0.95        # Discount factor
self.epsilon = 1.0       # Initial exploration rate
self.epsilon_min = 0.01  # Minimum exploration rate
self.epsilon_decay = 0.995  # Exploration decay rate
self.learning_rate = 0.001  # Learning rate
```

## Performance Expectations

With proper training, the DQN controller should achieve:
- 10-20% energy savings compared to the ondemand governor
- Similar or better application performance
- More stable CPU frequencies during phase transitions
- Better cross-VM coordination

The per-core DQN typically outperforms the action-based DQN due to its finer-grained control.
