#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import json
import time
import logging
import argparse
import psutil
from metrics_collector import MetricsCollector
from cpu_control import CPUController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dqn_controller.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DQNController")

class DQNAgent:
    """Deep Q-Network agent for CPU frequency scaling"""
    
    def __init__(self, state_size, action_size, freq_steps):
        self.state_size = state_size
        self.action_size = action_size
        self.freq_steps = freq_steps  # Available frequency steps
        
        # Experience replay buffer
        self.memory = deque(maxlen=20000)
        
        # Hyperparameters
        self.gamma = 0.95        # Discount factor
        self.epsilon = 1.0       # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration probability
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.learning_rate = 0.001  # Learning rate
        
        # Build and initialize models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        logger.info(f"DQN Agent initialized with state size {state_size}, action size {action_size}")
        logger.info(f"Available frequency steps: {self.freq_steps}")
    
    def _build_model(self):
        """Build the neural network model for DQN"""
        model = Sequential()
        # First hidden layer with input shape
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        # Second hidden layer
        model.add(Dense(64, activation='relu'))
        # Output layer - one output per action
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated with current model weights")
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train model using experiences from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights from file"""
        self.model.load_weights(name)
        logger.info(f"Model weights loaded from {name}")
    
    def save(self, name):
        """Save model weights to file"""
        self.model.save_weights(name)
        logger.info(f"Model weights saved to {name}")


class FrequencyScaler:
    """Handles the scaling of CPU frequencies based on DQN actions"""
    
    def __init__(self, cpu_controller, available_frequencies):
        self.cpu_controller = cpu_controller
        self.available_frequencies = available_frequencies
        self.cpu_count = self.cpu_controller.cpu_count
        self.action_map = self._create_action_map()
        
        logger.info(f"FrequencyScaler initialized with {len(self.action_map)} actions")
        logger.info(f"Available frequencies: {self.available_frequencies}")
    
    def _create_action_map(self):
        """Create mapping from action index to actual frequency changes"""
        action_map = {
            0: {"type": "no_change", "details": None},  # No change
            1: {"type": "all_up", "details": None},     # Increase all frequencies
            2: {"type": "all_down", "details": None},   # Decrease all frequencies
            3: {"type": "socket0_up", "details": None},   # Increase socket 0 frequencies
            4: {"type": "socket0_down", "details": None}, # Decrease socket 0 frequencies
            5: {"type": "socket1_up", "details": None},   # Increase socket 1 frequencies
            6: {"type": "socket1_down", "details": None}, # Decrease socket 1 frequencies
        }
        
        # Add per-core actions if we want finer control (optional)
        core_id = 0
        for socket in range(2):  # Assuming 2 sockets
            for core_in_socket in range(16):  # Assuming 16 cores per socket
                for direction in ["up", "down"]:
                    if core_id < self.cpu_count:
                        action_id = len(action_map)
                        action_map[action_id] = {
                            "type": "core",
                            "details": {"core": core_id, "direction": direction}
                        }
                core_id += 1
        
        return action_map
    
    def get_action_size(self):
        """Return the total number of possible actions"""
        return len(self.action_map)
    
    def apply_action(self, action, current_frequencies):
        """Apply the chosen action to change CPU frequencies"""
        new_frequencies = current_frequencies.copy()
        action_details = self.action_map[action]
        
        if action_details["type"] == "no_change":
            # Action 0: No change
            pass
        
        elif action_details["type"] == "all_up":
            # Action 1: Increase all frequencies
            for i in range(len(current_frequencies)):
                current_idx = self._get_freq_index(current_frequencies[i])
                if current_idx < len(self.available_frequencies) - 1:
                    new_frequencies[i] = self.available_frequencies[current_idx + 1]
        
        elif action_details["type"] == "all_down":
            # Action 2: Decrease all frequencies
            for i in range(len(current_frequencies)):
                current_idx = self._get_freq_index(current_frequencies[i])
                if current_idx > 0:
                    new_frequencies[i] = self.available_frequencies[current_idx - 1]
        
        elif action_details["type"] == "socket0_up":
            # Action 3: Increase socket 0 frequencies (cores 0-15)
            for i in range(min(16, len(current_frequencies))):
                current_idx = self._get_freq_index(current_frequencies[i])
                if current_idx < len(self.available_frequencies) - 1:
                    new_frequencies[i] = self.available_frequencies[current_idx + 1]
        
        elif action_details["type"] == "socket0_down":
            # Action 4: Decrease socket 0 frequencies (cores 0-15)
            for i in range(min(16, len(current_frequencies))):
                current_idx = self._get_freq_index(current_frequencies[i])
                if current_idx > 0:
                    new_frequencies[i] = self.available_frequencies[current_idx - 1]
        
        elif action_details["type"] == "socket1_up":
            # Action 5: Increase socket 1 frequencies (cores 16-31)
            start = 16
            for i in range(start, min(32, len(current_frequencies))):
                current_idx = self._get_freq_index(current_frequencies[i])
                if current_idx < len(self.available_frequencies) - 1:
                    new_frequencies[i] = self.available_frequencies[current_idx + 1]
        
        elif action_details["type"] == "socket1_down":
            # Action 6: Decrease socket 1 frequencies (cores 16-31)
            start = 16
            for i in range(start, min(32, len(current_frequencies))):
                current_idx = self._get_freq_index(current_frequencies[i])
                if current_idx > 0:
                    new_frequencies[i] = self.available_frequencies[current_idx - 1]
        
        elif action_details["type"] == "core":
            # Individual core actions
            core = action_details["details"]["core"]
            direction = action_details["details"]["direction"]
            
            if core < len(current_frequencies):
                current_idx = self._get_freq_index(current_frequencies[core])
                if direction == "up" and current_idx < len(self.available_frequencies) - 1:
                    new_frequencies[core] = self.available_frequencies[current_idx + 1]
                elif direction == "down" and current_idx > 0:
                    new_frequencies[core] = self.available_frequencies[current_idx - 1]
        
        # Apply the new frequencies
        self._set_new_frequencies(new_frequencies)
        
        return new_frequencies
    
    def _get_freq_index(self, freq):
        """Get the index of the frequency in available_frequencies"""
        # Find the closest frequency in our available list
        closest_idx = 0
        min_diff = float('inf')
        
        for i, available_freq in enumerate(self.available_frequencies):
            diff = abs(available_freq - freq)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
                
        return closest_idx
    
    def _set_new_frequencies(self, new_frequencies):
        """Apply new frequencies to the CPUs"""
        for cpu, freq in enumerate(new_frequencies):
            if cpu < self.cpu_count:
                self.cpu_controller.set_frequency(freq, [cpu])


class FeatureExtractor:
    """Extracts and processes features from CPU metrics for DQN state representation"""
    
    def __init__(self, cpu_count):
        self.cpu_count = cpu_count
        
    def process_cpu_data(self, cpu_data):
        """Process CPU data to extract features for DQN state"""
        features = []
        
        # Extract CPU utilization percentages
        cpu_usage = [cpu.get('usage_percent', 0) for cpu in cpu_data['percpu']]
        
        # Extract current frequencies (in MHz)
        cpu_freq = [cpu.get('frequency_mhz', 0) for cpu in cpu_data['percpu']]
        
        # Count high-usage cores (potential bottlenecks)
        high_usage_cores = sum(1 for usage in cpu_usage if usage > 80)
        
        # Calculate average usage per socket/NUMA node
        # Assuming first half CPUs are socket 0, rest are socket 1
        half_point = len(cpu_usage) // 2
        socket0_avg = np.mean(cpu_usage[:half_point]) if cpu_usage else 0
        socket1_avg = np.mean(cpu_usage[half_point:]) if cpu_usage else 0
        
        # Calculate variability in usage (indicates phase transitions)
        usage_std = np.std(cpu_usage) if cpu_usage else 0
        
        # Add usage metrics to features
        features.extend(cpu_usage)
        
        # Add frequency metrics to features
        features.extend(cpu_freq)
        
        # Add derived metrics
        features.append(high_usage_cores)
        features.append(socket0_avg)
        features.append(socket1_avg)
        features.append(usage_std)
        
        # Add time-based features (can help identify patterns over time)
        current_time = time.time()
        features.append(np.sin(2 * np.pi * current_time / 86400))  # Day cycle
        features.append(np.cos(2 * np.pi * current_time / 86400))  # Day cycle
        
        return np.array(features)


class RewardCalculator:
    """Calculates rewards based on energy efficiency and performance considerations"""
    
    def __init__(self):
        # Weights for different reward components
        self.energy_weight = 0.5
        self.performance_weight = 2.0
        self.stability_weight = 0.1
        self.app_performance_weight = 1.0
        
        # Historical data for detecting performance degradation
        self.performance_history = []
        self.history_max_len = 10
        
    def calculate_reward(self, current_state, next_state, action, performance_metrics=None):
        """
        Calculate reward based on multiple factors:
        1. Energy efficiency (frequency reduction)
        2. Performance maintenance
        3. Frequency stability (avoid oscillations)
        """
        # Extract metrics from states
        # Assuming CPU usage in first half of state, frequencies in second half
        state_len = len(current_state)
        cpu_count = (state_len - 6) // 2  # 6 derived metrics at the end
        
        current_usage = current_state[:cpu_count]
        next_usage = next_state[:cpu_count]
        current_freqs = current_state[cpu_count:2*cpu_count]
        next_freqs = next_state[cpu_count:2*cpu_count]
        
        # 1. Energy efficiency reward
        # Lower frequency = lower power = higher reward, but weighted by CPU usage
        # Normalize frequencies to 0-1 range (assuming 1200-2600 MHz range)
        max_freq = 2600.0
        norm_next_freqs = [min(freq / max_freq, 1.0) for freq in next_freqs]
        
        # Energy reward is higher when we reduce frequency for idle cores
        # and maintain frequency for active cores
        energy_reward = 0
        for i, (usage, freq) in enumerate(zip(next_usage, norm_next_freqs)):
            if usage < 20:  # Idle core
                energy_reward += (1 - freq) * 2  # Reward lower frequency
            else:  # Active core
                energy_reward += (1 - freq) * (1 - min(1, usage/100))  # Smaller reward
        
        # 2. Performance penalty
        performance_penalty = 0
        for i, (prev_usage, curr_usage, old_freq, new_freq) in enumerate(
                zip(current_usage, next_usage, current_freqs, next_freqs)):
            
            # Penalize lowering frequency on high-usage cores
            if curr_usage > 80 and new_freq < old_freq:
                performance_penalty -= 10
            
            # Penalize if usage increased significantly but frequency decreased
            if curr_usage > prev_usage + 20 and new_freq < old_freq:
                performance_penalty -= 5
        
        # 3. Stability reward (avoid frequent changes)
        stability_reward = -sum(abs(next_freqs[i] - current_freqs[i]) 
                               for i in range(len(current_freqs))) / 100
        
        # 4. Application performance reward
        app_performance_reward = 0
        if performance_metrics:
            # Example: If we have throughput or latency metrics
            if 'throughput' in performance_metrics:
                app_performance_reward += performance_metrics['throughput'] / 1000
            if 'latency' in performance_metrics:
                app_performance_reward -= performance_metrics['latency'] / 10
        
        # Track performance metrics history
        if performance_metrics:
            self.performance_history.append(performance_metrics)
            if len(self.performance_history) > self.history_max_len:
                self.performance_history.pop(0)
            
            # Add trend-based rewards
            if len(self.performance_history) > 2:
                # Check if performance is improving over time
                if 'throughput' in performance_metrics and 'throughput' in self.performance_history[-2]:
                    if performance_metrics['throughput'] > self.performance_history[-2]['throughput']:
                        app_performance_reward += 5
        
        # Combine all reward components with appropriate weights
        total_reward = (
            self.energy_weight * energy_reward + 
            self.performance_weight * performance_penalty +
            self.stability_weight * stability_reward + 
            self.app_performance_weight * app_performance_reward
        )
        
        return total_reward


class DQNController:
    """Main controller class that uses DQN to optimize CPU frequencies"""
    
    def __init__(self, output_dir="./dqn_results", model_path=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.cpu_controller = CPUController()
        self.cpu_count = self.cpu_controller.cpu_count
        
        # Get available frequency steps
        min_freq, max_freq = self.cpu_controller.get_min_max_freq()
        self.available_frequencies = [
            min_freq // 1000,  # Convert from KHz to MHz
            min_freq // 1000 + (max_freq - min_freq) // 3000,
            min_freq // 1000 + 2 * (max_freq - min_freq) // 3000,
            max_freq // 1000
        ]
        logger.info(f"Available frequency steps: {self.available_frequencies} MHz")
        
        # Initialize frequency scaler
        self.freq_scaler = FrequencyScaler(
            self.cpu_controller, 
            self.available_frequencies
        )
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.cpu_count)
        
        # Calculate state size
        # CPU usage + CPU freq + derived metrics + time features
        self.state_size = self.cpu_count * 2 + 4 + 2
        
        # Get action size from frequency scaler
        self.action_size = self.freq_scaler.get_action_size()
        
        # Initialize DQN agent
        self.agent = DQNAgent(
            self.state_size, 
            self.action_size,
            self.available_frequencies
        )
        
        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Initialize performance metrics
        self.performance_metrics = {}
        
        # Initialize logging
        self.stats_file = open(os.path.join(output_dir, "dqn_stats.csv"), "w")
        self.stats_file.write("episode,step,action,reward,total_reward,epsilon\n")
        
        logger.info(f"DQN Controller initialized with state size {self.state_size}, "
                   f"action size {self.action_size}")
    
    def collect_metrics(self):
        """Collect current CPU metrics"""
        # Using our metrics collector
        cpu_data = {
            "percpu": []
        }
        
        # Get CPU info from psutil
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        
        for i in range(len(cpu_percent)):
            cpu_entry = {
                "cpu": i,
                "usage_percent": cpu_percent[i],
                "frequency_mhz": cpu_freq[i].current if i < len(cpu_freq) and cpu_freq[i] else 1200.0,
                "times": {}  # We don't need detailed times for the DQN
            }
            cpu_data["percpu"].append(cpu_entry)
        
        return cpu_data
    
    def get_current_frequencies(self, cpu_data):
        """Extract current frequencies from CPU data"""
        return [cpu.get("frequency_mhz", 0) for cpu in cpu_data["percpu"]]
    
    def get_performance_metrics(self):
        """Get application performance metrics if available"""
        # In production, you would implement this to get metrics from VMs
        # For now, we'll return a mock implementation
        return self.performance_metrics
    
    def update_performance_metrics(self, metrics):
        """Update the performance metrics (called by external components)"""
        self.performance_metrics = metrics
    
    def train(self, episodes=100, steps_per_episode=100, batch_size=32):
        """Train the DQN controller"""
        logger.info(f"Starting training for {episodes} episodes, "
                   f"{steps_per_episode} steps per episode")
        
        for episode in range(episodes):
            logger.info(f"Episode {episode+1}/{episodes}")
            
            # Reset environment (set default governor)
            self.cpu_controller.set_governor("userspace")
            
            # Get initial state
            cpu_data = self.collect_metrics()
            current_frequencies = self.get_current_frequencies(cpu_data)
            current_state = self.feature_extractor.process_cpu_data(cpu_data)
            current_state = np.reshape(current_state, [1, self.state_size])
            
            total_reward = 0
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.agent.act(current_state)
                
                # Apply action
                new_frequencies = self.freq_scaler.apply_action(action, current_frequencies)
                
                # Wait for system to stabilize after frequency change
                time.sleep(1)
                
                # Get new state
                cpu_data = self.collect_metrics()
                next_state = self.feature_extractor.process_cpu_data(cpu_data)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Get performance metrics
                performance_metrics = self.get_performance_metrics()
                
                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    current_state[0],
                    next_state[0],
                    action,
                    performance_metrics
                )
                total_reward += reward
                
                # Check if episode is done (always False in our case)
                done = False
                
                # Remember experience
                self.agent.remember(current_state, action, reward, next_state, done)
                
                # Update state and frequencies
                current_state = next_state
                current_frequencies = new_frequencies
                
                # Train the network
                self.agent.replay(batch_size)
                
                # Log statistics
                self.stats_file.write(f"{episode+1},{step+1},{action},{reward},{total_reward},{self.agent.epsilon}\n")
                self.stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step+1}/{steps_per_episode}, "
                               f"Action: {action}, Reward: {reward:.2f}, "
                               f"Total Reward: {total_reward:.2f}, "
                               f"Epsilon: {self.agent.epsilon:.4f}")
            
            # Update target network at the end of each episode
            self.agent.update_target_model()
            
            # Save model periodically
            if (episode + 1) % 5 == 0 or episode == episodes - 1:
                self.agent.save(os.path.join(self.output_dir, f"dqn_model_ep{episode+1}.h5"))
                logger.info(f"Model saved at episode {episode+1}")
            
            logger.info(f"Episode {episode+1}/{episodes} completed, "
                       f"Total Reward: {total_reward:.2f}, "
                       f"Epsilon: {self.agent.epsilon:.4f}")
    
    def run(self, model_path, run_time=3600):
        """Run the trained DQN model"""
        # Load model
        self.agent.load(model_path)
        self.agent.epsilon = 0.01  # Set exploration low for inference
        
        logger.info(f"Running trained model from {model_path} for {run_time} seconds")
        
        # Get initial state
        cpu_data = self.collect_metrics()
        current_frequencies = self.get_current_frequencies(cpu_data)
        current_state = self.feature_extractor.process_cpu_data(cpu_data)
        current_state = np.reshape(current_state, [1, self.state_size])
        
        # Open stats file
        stats_file = open(os.path.join(self.output_dir, "dqn_inference.csv"), "w")
        stats_file.write("timestamp,action,frequencies\n")
        
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < run_time:
                step += 1
                
                # Choose action
                action = self.agent.act(current_state)
                
                # Apply action
                new_frequencies = self.freq_scaler.apply_action(action, current_frequencies)
                
                # Log action and frequencies
                timestamp = time.time()
                freq_str = ",".join(str(f) for f in new_frequencies)
                stats_file.write(f"{timestamp},{action},{freq_str}\n")
                stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step}, Action: {action}, "
                               f"Elapsed: {time.time() - start_time:.1f}s")
                
                # Wait before next adjustment
                time.sleep(5)
                
                # Get new state
                cpu_data = self.collect_metrics()
                next_state = self.feature_extractor.process_cpu_data(cpu_data)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Update state and frequencies
                current_state = next_state
                current_frequencies = new_frequencies
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        
        finally:
            stats_file.close()
            logger.info(f"Run completed, total steps: {step}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stats_file.close()
        # Reset CPU governor to default
        self.cpu_controller.set_governor("ondemand")
        logger.info("Controller cleaned up, CPU governor reset to ondemand")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Controller for CPU Frequency Scaling")
    parser.add_argument("--train", action="store_true", help="Train the DQN model")
    parser.add_argument("--run", help="Run using a trained model", metavar="MODEL_PATH")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--run-time", type=int, default=3600, help="Run time in seconds")
    parser.add_argument("--output-dir", default="./dqn_results", help="Output directory")
    
    args = parser.parse_args()
    
    controller = DQNController(output_dir=args.output_dir)
    
    try:
        if args.train:
            controller.train(episodes=args.episodes, steps_per_episode=args.steps)
        elif args.run:
            controller.run(args.run, run_time=args.run_time)
        else:
            parser.print_help()
    
    finally:
        controller.cleanup()
