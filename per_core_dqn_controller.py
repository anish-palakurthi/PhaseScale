#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Lambda
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
        logging.FileHandler("per_core_dqn.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PerCoreDQN")

class PerCoreDQNAgent:
    """DQN agent that outputs frequencies for each CPU core directly"""
    
    def __init__(self, state_size, cpu_count, available_frequencies):
        self.state_size = state_size
        self.cpu_count = cpu_count
        self.available_frequencies = available_frequencies
        self.num_freq_options = len(available_frequencies)
        
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
        
        logger.info(f"PerCoreDQN initialized with state size {state_size}, CPU count {cpu_count}")
        logger.info(f"Available frequencies: {self.available_frequencies}")
    
    def _build_model(self):
        """Build a model that outputs frequency decisions for each core"""
        # Input layer
        input_layer = Input(shape=(self.state_size,))
        
        # Shared layers
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)
        
        # Create separate output heads for each CPU core
        # Each head will output Q-values for each available frequency option
        output_layers = []
        for i in range(self.cpu_count):
            core_output = Dense(64, activation='relu')(x)
            core_output = Dense(self.num_freq_options, activation='linear')(core_output)
            output_layers.append(core_output)
        
        # Create the model with multiple outputs
        model = Model(inputs=input_layer, outputs=output_layers)
        
        # Compile with MSE loss
        model.compile(
            loss=['mse'] * self.cpu_count,
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated with current model weights")
    
    def remember(self, state, actions, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, actions, reward, next_state, done))
    
    def act(self, state):
        """Choose frequency action for each CPU core"""
        actions = []
        
        # Decide whether to explore or exploit
        if np.random.rand() <= self.epsilon:
            # Exploration: random frequencies for each core
            for i in range(self.cpu_count):
                actions.append(random.randrange(self.num_freq_options))
        else:
            # Exploitation: choose best frequencies based on model prediction
            q_values = self.model.predict(state, verbose=0)
            for core_q_values in q_values:
                actions.append(np.argmax(core_q_values[0]))
        
        return actions
    
    def get_frequencies_from_actions(self, actions):
        """Convert action indices to actual frequency values"""
        return [self.available_frequencies[action] for action in actions]
    
    def replay(self, batch_size):
        """Train model using experiences from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, actions, reward, next_state, done in minibatch:
            # Calculate target Q values
            if not done:
                # Get future Q values from target model
                next_q_values = self.target_model.predict(next_state, verbose=0)
                
                # For each core, get maximum Q value in next state
                next_max_q = [np.amax(next_q_values[i][0]) for i in range(self.cpu_count)]
                
                # Calculate target Q value for each core: r + gamma * max(Q')
                targets = [reward + self.gamma * next_max for next_max in next_max_q]
            else:
                # If done, target is just the reward
                targets = [reward] * self.cpu_count
            
            # Get current Q values
            current_q_values = self.model.predict(state, verbose=0)
            
            # Create target Q value arrays for each core
            target_f = []
            for i in range(self.cpu_count):
                core_q = current_q_values[i][0].copy()
                core_q[actions[i]] = targets[i]
                target_f.append(np.reshape(core_q, (1, self.num_freq_options)))
            
            # Train the model
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


class FeatureExtractor:
    """Extracts and processes features from CPU metrics for DQN state"""
    
    def __init__(self, cpu_count):
        self.cpu_count = cpu_count
        
    def process_cpu_data(self, cpu_data):
        """Process CPU data to extract features for DQN state"""
        features = []
        
        # Extract CPU utilization percentages
        cpu_usage = [cpu.get('usage_percent', 0) for cpu in cpu_data['percpu']]
        
        # Extract current frequencies (in MHz)
        cpu_freq = [cpu.get('frequency_mhz', 0) for cpu in cpu_data['percpu']]
        
        # Calculate average usage per socket (assuming 2 sockets)
        socket0_avg = np.mean(cpu_usage[:self.cpu_count//2]) if cpu_usage else 0
        socket1_avg = np.mean(cpu_usage[self.cpu_count//2:]) if cpu_usage else 0
        
        # Calculate standard deviation of usage (indicates phase transitions)
        usage_std = np.std(cpu_usage) if cpu_usage else 0
        
        # Add CPU usage metrics to features
        features.extend(cpu_usage)
        
        # Add CPU frequency metrics to features
        features.extend(cpu_freq)
        
        # Add derived metrics
        features.append(socket0_avg)
        features.append(socket1_avg)
        features.append(usage_std)
        
        # Add historical usage trend features (percent changes in last interval)
        # This will be zero for the first measurement
        self.prev_usage = getattr(self, 'prev_usage', cpu_usage)
        usage_trends = [(curr - prev) for curr, prev in zip(cpu_usage, self.prev_usage)]
        features.extend(usage_trends)
        self.prev_usage = cpu_usage
        
        # Add time-based features to help capture daily patterns
        current_time = time.time()
        features.append(np.sin(2 * np.pi * current_time / 86400))  # Day cycle
        features.append(np.cos(2 * np.pi * current_time / 86400))  # Day cycle
        
        return np.array(features)


class RewardCalculator:
    """Calculates rewards based on energy efficiency and performance considerations"""
    
    def __init__(self, available_frequencies):
        # Store min and max frequencies for normalization
        self.min_freq = min(available_frequencies)
        self.max_freq = max(available_frequencies)
        
        # Weights for different reward components
        self.energy_weight = 0.5
        self.performance_weight = 2.0
        self.stability_weight = 0.1
        self.app_performance_weight = 1.0
        
        # Historical data for performance tracking
        self.performance_history = []
        self.history_max_len = 10
        
    def calculate_reward(self, current_state, next_state, actions, frequencies, performance_metrics=None):
        """
        Calculate reward based on multiple factors:
        1. Energy efficiency (lower frequency = less energy)
        2. Performance maintenance (higher frequency for busy cores)
        3. Frequency stability (avoid oscillations)
        """
        # Extract metrics from states
        # Find offsets based on state size and cpu count
        state_len = len(current_state)
        cpu_count = (state_len - 5 - cpu_count) // 3  # 5 derived metrics + usage trends
        
        # Extract relevant parts of the state
        current_usage = current_state[:cpu_count]
        next_usage = next_state[:cpu_count]
        current_freqs = current_state[cpu_count:2*cpu_count]
        
        # Calculate energy reward
        energy_reward = 0
        for i, (freq, usage) in enumerate(zip(frequencies, next_usage)):
            # Normalize frequency to 0-1 range
            norm_freq = (freq - self.min_freq) / (self.max_freq - self.min_freq)
            
            # Higher reward for lower frequency on idle cores
            if usage < 20:  # Idle core
                energy_reward += (1 - norm_freq) * 2  # Reward lower frequency
            else:  # Active core
                # Small reward for energy saving, scaled by usage
                energy_reward += (1 - norm_freq) * (1 - min(1, usage/100))
        
        # Calculate performance reward/penalty
        performance_reward = 0
        for i, (curr_usage, prev_usage, freq) in enumerate(zip(next_usage, current_usage, frequencies)):
            # Reward high frequency for high-usage cores
            if curr_usage > 80:
                # Normalize frequency to 0-1 range
                norm_freq = (freq - self.min_freq) / (self.max_freq - self.min_freq)
                performance_reward += norm_freq * 3  # High reward for high frequency on busy cores
            
            # Penalize if usage increased but frequency is low
            if curr_usage > prev_usage + 20:
                # Normalize frequency to 0-1 range
                norm_freq = (freq - self.min_freq) / (self.max_freq - self.min_freq)
                if norm_freq < 0.5:  # Low frequency
                    performance_reward -= (1 - norm_freq) * 5  # Penalty
        
        # Calculate stability reward (penalize frequent large changes)
        stability_reward = 0
        for i, (curr_freq, prev_freq) in enumerate(zip(frequencies, current_freqs)):
            # Penalize large frequency changes
            freq_change = abs(curr_freq - prev_freq) / self.max_freq
            stability_reward -= freq_change * 10
        
        # Application performance reward (if metrics are available)
        app_performance_reward = 0
        if performance_metrics:
            # Example: If we have throughput or latency metrics
            if 'throughput' in performance_metrics:
                app_performance_reward += performance_metrics['throughput'] / 1000
            if 'latency' in performance_metrics:
                app_performance_reward -= performance_metrics['latency'] / 10
            
            # Track performance history
            self.performance_history.append(performance_metrics)
            if len(self.performance_history) > self.history_max_len:
                self.performance_history.pop(0)
                
            # Add trend-based rewards
            if len(self.performance_history) > 2:
                if 'throughput' in performance_metrics and 'throughput' in self.performance_history[-2]:
                    if performance_metrics['throughput'] > self.performance_history[-2]['throughput']:
                        app_performance_reward += 5
        
        # Combine all reward components with weights
        total_reward = (
            self.energy_weight * energy_reward + 
            self.performance_weight * performance_reward +
            self.stability_weight * stability_reward + 
            self.app_performance_weight * app_performance_reward
        )
        
        return total_reward


class PerCoreDQNController:
    """Controller that uses DQN to set frequencies for each core individually"""
    
    def __init__(self, output_dir="./per_core_dqn_results", model_path=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.cpu_controller = CPUController()
        self.cpu_count = self.cpu_controller.cpu_count
        
        # Define available frequency options (in KHz for the controller)
        self.available_frequencies_khz = [1200000, 1550000, 1900000, 2250000, 2600000]
        
        # Also define in MHz for internal use
        self.available_frequencies = [freq // 1000 for freq in self.available_frequencies_khz]
        logger.info(f"Available frequency steps: {self.available_frequencies} MHz")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.cpu_count)
        
        # Calculate state size:
        # CPU usage + CPU freq + derived metrics + usage trends + time features
        self.state_size = self.cpu_count * 3 + 3 + 2
        
        # Initialize DQN agent
        self.agent = PerCoreDQNAgent(
            self.state_size, 
            self.cpu_count,
            self.available_frequencies
        )
        
        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(self.available_frequencies)
        
        # Initialize performance metrics
        self.performance_metrics = {}
        
        # Initialize logging
        self.stats_file = open(os.path.join(output_dir, "per_core_dqn_stats.csv"), "w")
        self.stats_file.write("episode,step,actions,frequencies,reward,total_reward,epsilon\n")
        
        logger.info(f"Per-Core DQN Controller initialized with state size {self.state_size}, "
                   f"CPU count {self.cpu_count}")
    
    def collect_metrics(self):
        """Collect current CPU metrics"""
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
        # In a real system, you would implement this to get metrics from VMs
        # For this example, we'll just return the stored metrics
        return self.performance_metrics
    
    def update_performance_metrics(self, metrics):
        """Update the performance metrics (called by external components)"""
        self.performance_metrics = metrics
    
    def apply_frequencies(self, freq_actions):
        """Apply the chosen frequency actions to the CPU cores"""
        # Convert action indices to actual frequencies (in KHz for the controller)
        frequencies_khz = [self.available_frequencies_khz[action] for action in freq_actions]
        frequencies_mhz = [freq // 1000 for freq in frequencies_khz]
        
        # Apply each frequency to its corresponding core
        for cpu, freq_khz in enumerate(frequencies_khz):
            if cpu < self.cpu_count:
                try:
                    self.cpu_controller.set_frequency(freq_khz, [cpu])
                    logger.debug(f"Set CPU {cpu} to {freq_khz // 1000} MHz")
                except Exception as e:
                    logger.error(f"Failed to set frequency for CPU {cpu}: {str(e)}")
        
        return frequencies_mhz
    
    def train(self, episodes=50, steps_per_episode=100, batch_size=32):
        """Train the per-core DQN controller"""
        logger.info(f"Starting training for {episodes} episodes, "
                   f"{steps_per_episode} steps per episode")
        
        for episode in range(episodes):
            logger.info(f"Episode {episode+1}/{episodes}")
            
            # Reset environment (set userspace governor for all cores)
            self.cpu_controller.set_governor("userspace")
            
            # Get initial state
            cpu_data = self.collect_metrics()
            current_frequencies = self.get_current_frequencies(cpu_data)
            current_state = self.feature_extractor.process_cpu_data(cpu_data)
            current_state = np.reshape(current_state, [1, self.state_size])
            
            total_reward = 0
            
            for step in range(steps_per_episode):
                # Choose frequency actions for each core
                actions = self.agent.act(current_state)
                
                # Apply the chosen frequencies
                frequencies_mhz = self.apply_frequencies(actions)
                
                # Wait for system to stabilize
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
                    actions,
                    frequencies_mhz,
                    performance_metrics
                )
                total_reward += reward
                
                # Check if episode is done (always False in our case)
                done = False
                
                # Remember experience
                self.agent.remember(current_state, actions, reward, next_state, done)
                
                # Update state
                current_state = next_state
                
                # Train the network
                self.agent.replay(batch_size)
                
                # Log statistics
                actions_str = ",".join(str(a) for a in actions)
                freqs_str = ",".join(str(f) for f in frequencies_mhz)
                self.stats_file.write(f"{episode+1},{step+1},{actions_str},{freqs_str},{reward},{total_reward},{self.agent.epsilon}\n")
                self.stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step+1}/{steps_per_episode}, "
                               f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, "
                               f"Epsilon: {self.agent.epsilon:.4f}")
                    logger.info(f"Sample frequencies (MHz): {frequencies_mhz[:5]}...")
            
            # Update target network at the end of each episode
            self.agent.update_target_model()
            
            # Save model periodically
            if (episode + 1) % 5 == 0 or episode == episodes - 1:
                self.agent.save(os.path.join(self.output_dir, f"per_core_dqn_ep{episode+1}.h5"))
                logger.info(f"Model saved at episode {episode+1}")
            
            logger.info(f"Episode {episode+1}/{episodes} completed, "
                       f"Total Reward: {total_reward:.2f}, "
                       f"Epsilon: {self.agent.epsilon:.4f}")
    
    def run(self, model_path, run_time=3600):
        """Run the trained per-core DQN model"""
        # Load model
        self.agent.load(model_path)
        self.agent.epsilon = 0.01  # Set exploration low for inference
        
        logger.info(f"Running trained model from {model_path} for {run_time} seconds")
        
        # Get initial state
        cpu_data = self.collect_metrics()
        current_state = self.feature_extractor.process_cpu_data(cpu_data)
        current_state = np.reshape(current_state, [1, self.state_size])
        
        # Open stats file
        stats_file = open(os.path.join(self.output_dir, "per_core_dqn_inference.csv"), "w")
        stats_file.write("timestamp,cpu_id,frequency_mhz,usage_percent\n")
        
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < run_time:
                step += 1
                
                # Choose frequency actions for each core
                actions = self.agent.act(current_state)
                
                # Apply the chosen frequencies
                frequencies_mhz = self.apply_frequencies(actions)
                
                # Get current CPU data for logging
                cpu_data = self.collect_metrics()
                cpu_usage = [cpu.get('usage_percent', 0) for cpu in cpu_data['percpu']]
                
                # Log data
                timestamp = time.time()
                for cpu_id, (freq, usage) in enumerate(zip(frequencies_mhz, cpu_usage)):
                    if cpu_id < self.cpu_count:
                        stats_file.write(f"{timestamp},{cpu_id},{freq},{usage}\n")
                stats_file.flush()
                
                if step % 10 == 0:
                    logger.info(f"Step {step}, Elapsed: {time.time() - start_time:.1f}s")
                    logger.info(f"Sample frequencies (MHz): {frequencies_mhz[:5]}...")
                
                # Wait before next adjustment
                time.sleep(5)
                
                # Get new state
                next_state = self.feature_extractor.process_cpu_data(cpu_data)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # Update state
                current_state = next_state
        
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
    parser = argparse.ArgumentParser(description="Per-Core DQN Controller for CPU Frequency Scaling")
    parser.add_argument("--train", action="store_true", help="Train the DQN model")
    parser.add_argument("--run", help="Run using a trained model", metavar="MODEL_PATH")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--run-time", type=int, default=3600, help="Run time in seconds")
    parser.add_argument("--output-dir", default="./per_core_dqn_results", help="Output directory")
    
    args = parser.parse_args()
    
    controller = PerCoreDQNController(output_dir=args.output_dir)
    
    try:
        if args.train:
            controller.train(episodes=args.episodes, steps_per_episode=args.steps)
        elif args.run:
            controller.run(args.run, run_time=args.run_time)
        else:
            parser.print_help()
    
    finally:
        controller.cleanup()
