# CPU-DQN: Deep Q-Network for CPU Frequency Control

This README explains how to use the simplified CPUDQN class for optimizing CPU frequencies.

## Overview

`CPUDQN` is a specialized Deep Q-Network implementation that:
- Takes CPU metrics as input (matching your existing JSON format)
- Outputs frequency decisions for each CPU core
- Learns from reward feedback to optimize its policy
- Supports saving and loading trained models

## Installation

```bash
pip install tensorflow numpy
```

## Basic Usage

```python
from cpu_dqn import CPUDQN
import json

# Initialize with the number of CPU cores
dqn = CPUDQN(n_cpus=32)

# Load JSON metrics
with open('cpu_metrics.json', 'r') as f:
    cpu_data = json.load(f)

# Get frequency recommendations
frequencies = dqn.predict(cpu_data)
print(f"Recommended frequencies: {frequencies}")

# After applying frequencies and observing results,
# provide reward feedback
dqn.reward(5.0)  # Positive reward for good outcome

# Save trained model
dqn.save("./trained_model.h5")
```

## Integration with PhaseScale

### In your master script:

```python
import json
from cpu_dqn import CPUDQN

# Initialize DQN with number of cores
dqn = CPUDQN(n_cpus=32, model_path="./trained_model.h5")

# Inside your main loop where you're collecting metrics
while running:
    # Collect metrics (from your existing code)
    cpu_metrics = collect_metrics()
    
    # Get frequency recommendations
    frequencies = dqn.predict(cpu_metrics)
    
    # Apply frequencies to CPUs
    for cpu_id, frequency in enumerate(frequencies):
        set_cpu_frequency(cpu_id, frequency)
    
    # Wait for effects
    time.sleep(1)
    
    # Calculate reward based on your objectives
    # (energy efficiency, performance, etc.)
    reward_value = calculate_reward()
    
    # Provide feedback to DQN
    dqn.reward(reward_value)
    
    # Periodically save the model
    if step % 1000 == 0:
        dqn.save(f"./model_checkpoint_{step}.h5")
```

## Training Process

The CPUDQN uses experience replay and a target network to stabilize learning:

1. Each time you call `predict()`, the DQN:
   - Processes the input metrics
   - Either explores (random actions) or exploits (best learned actions)
   - Returns frequency recommendations
   - Stores the state for future learning

2. When you call `reward()`, the DQN:
   - Stores the complete experience (state, action, reward, next state)
   - Trains the model using a random batch of past experiences
   - Gradually reduces exploration as it learns

## Example Reward Function

Here's an example of how to calculate rewards:

```python
def calculate_reward():
    # Get new metrics after applying frequencies
    new_metrics = collect_metrics()
    
    # Extract relevant data
    energy_usage = sum(new_metrics['rapl_energy'][pkg]['delta_uj'] 
                       for pkg in ['package-0', 'package-1'])
    
    # Get application performance metrics
    throughput = get_application_throughput()
    
    # Balance energy efficiency and performance
    # Higher is better for both components
    energy_score = -energy_usage / 1000000  # Negative because lower energy is better
    performance_score = throughput / 1000   # Normalize
    
    # Combined reward (weighted sum)
    reward = (0.7 * energy_score) + (0.3 * performance_score)
    
    return reward
```

## Customization

### Adjusting Learning Parameters

You can customize the DQN by adjusting its parameters:

```python
dqn = CPUDQN(n_cpus=32)
dqn.gamma = 0.98         # Increase discount factor (more long-term focused)
dqn.epsilon_decay = 0.99 # Slower exploration decay
dqn.learning_rate = 0.0005 # Lower learning rate for more stability
```

### Changing Available Frequencies

If your system supports different frequency steps:

```python
dqn = CPUDQN(n_cpus=32)
dqn.frequency_options = [1000, 1400, 1800, 2200, 2600, 3000]
# Rebuild model to match the new action space
dqn.model = dqn._build_model()
dqn.target_model = dqn._build_model()
dqn._update_target_model()
```

## Files to Delete

The following files are no longer needed with this simplified implementation:

1. `dqn_controller.py` - Replaced by the simplified `cpu_dqn.py`
2. `per_core_dqn_controller.py` - Functionality integrated into `cpu_dqn.py`
3. `baseline_controllers.py` - Not needed for the core DQN functionality
4. `dqn_with_vms.py` - The integration is now simpler and custom to your needs

Keep these files if you want the additional analysis capabilities:
- `dqn_analyzer.py` - Still useful for analyzing training performance

## Troubleshooting

### Training Issues

If the DQN isn't learning effectively:

1. Check your reward function - ensure it properly reflects your goals
2. Increase the exploration rate - set `dqn.epsilon = 0.5` and `dqn.epsilon_decay = 0.999`
3. Normalize your inputs - extreme values can destabilize training

### Model Saving/Loading

If you encounter issues with model saving:

1. Ensure you have write permissions to the directory
2. Check for TensorFlow compatibility if loading across different versions
