#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cpu_dqn.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CPU_DQN")

class CPUDQN:
    """
    Deep Q-Network for CPU frequency control
    
    This class implements a DQN that takes CPU metrics as input and
    outputs a frequency selection for each CPU core.
    """
    
    def __init__(self, n_cpus, model_path=None, memory_size=10000):
        """
        Initialize the DQN model
        
        Args:
            n_cpus: Number of CPU cores to control
            model_path: Path to saved model weights (optional)
            memory_size: Size of replay memory buffer
        """
        self.n_cpus = n_cpus
        
        # Available frequency options in MHz
        self.frequency_options = [1200, 1550, 1900, 2250, 2600]
        self.n_freq_options = len(self.frequency_options)
        
        # State and action variables
        self.state = None
        self.last_state = None
        self.last_action = None
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Hyperparameters
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Build model architecture
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            # Set low exploration for inference
            self.epsilon = 0.01
            logger.info(f"Loaded model from {model_path}")
        
        logger.info(f"Initialized CPUDQN with {n_cpus} CPUs")
        logger.info(f"Frequency options: {self.frequency_options} MHz")
    
    def _build_model(self):
        """Build neural network model"""
        # Estimate state size based on CPU metrics
        # For each CPU: usage, frequency, + some global features
        state_size = self.n_cpus * 2 + 6
        
        # Input layer
        input_layer = Input(shape=(state_size,))
        
        # Shared layers
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)
        
        # Output heads for each CPU
        outputs = []
        for i in range(self.n_cpus):
            cpu_head = Dense(64, activation='relu')(x)
            cpu_output = Dense(self.n_freq_options, activation='linear')(cpu_head)
            outputs.append(cpu_output)
        
        # Create model
        model = Model(inputs=input_layer, outputs=outputs)
        model.compile(
            loss=['mse'] * self.n_cpus,
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def _update_target_model(self):
        """Copy weights to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def _preprocess_state(self, cpu_data):
        """
        Extract features from CPU data
        
        Args:
            cpu_data: JSON data containing CPU metrics
        
        Returns:
            Numpy array of features
        """
        features = []
        
        # Extract CPU utilization
        cpu_usage = [cpu.get('usage_percent', 0) for cpu in cpu_data['percpu']]
        cpu_usage = cpu_usage[:self.n_cpus]  # Limit to n_cpus
        
        # Extract CPU frequencies (MHz)
        cpu_freq = [cpu.get('frequency_mhz', 0) for cpu in cpu_data['percpu']]
        cpu_freq = cpu_freq[:self.n_cpus]  # Limit to n_cpus
        
        # Calculate socket averages (assuming 2 sockets)
        socket_size = self.n_cpus // 2
        socket0_avg = np.mean(cpu_usage[:socket_size])
        socket1_avg = np.mean(cpu_usage[socket_size:])
        
        # Calculate usage variability
        usage_std = np.std(cpu_usage)
        
        # Add base features
        features.extend(cpu_usage)
        features.extend(cpu_freq)
        
        # Add derived features
        features.append(socket0_avg)
        features.append(socket1_avg)
        features.append(usage_std)
        
        # Add RAPL energy info if available
        if 'rapl_energy' in cpu_data:
            rapl = cpu_data['rapl_energy']
            # Add normalized energy features
            features.append(rapl.get('package-0', {}).get('delta_uj', 0) / 1000000)
            features.append(rapl.get('package-1', {}).get('delta_uj', 0) / 1000000)
            features.append(rapl.get('dram', {}).get('delta_uj', 0) / 1000000)
        else:
            # Placeholder values if RAPL not available
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def predict(self, cpu_data_json):
        """
        Predict optimal CPU frequencies based on current state
        
        Args:
            cpu_data_json: JSON string or object with CPU metrics
        
        Returns:
            List of frequency values in MHz for each CPU
        """
        # Parse JSON if string is provided
        if isinstance(cpu_data_json, str):
            cpu_data = json.loads(cpu_data_json)
        else:
            cpu_data = cpu_data_json
        
        # Preprocess state
        self.last_state = self.state
        self.state = self._preprocess_state(cpu_data)
        state = np.reshape(self.state, [1, len(self.state)])
        
        # Choose action (either explore or exploit)
        if np.random.rand() <= self.epsilon:
            # Exploration: random frequencies
            self.last_action = [random.randrange(self.n_freq_options) for _ in range(self.n_cpus)]
        else:
            # Exploitation: best frequencies from model
            q_values = self.model.predict(state, verbose=0)
            self.last_action = [np.argmax(q_values[i][0]) for i in range(self.n_cpus)]
        
        # Convert action indices to actual frequencies
        frequencies = [self.frequency_options[action] for action in self.last_action]
        
        return frequencies
    
    def reward(self, reward_value):
        """
        Provide reward feedback for the last action
        
        Args:
            reward_value: Scalar reward value
        """
        # Ensure we have a previous state and action
        if self.last_state is None or self.last_action is None:
            logger.warning("Cannot process reward: no previous state or action")
            return
        
        # Store experience in replay memory
        self.memory.append((
            self.last_state, 
            self.last_action, 
            reward_value, 
            self.state, 
            False  # done flag (always False in our case)
        ))
        
        # Train the model if we have enough samples
        if len(self.memory) >= self.batch_size:
            self._replay()
            
            # Occasionally update target model
            if np.random.rand() < 0.1:  # 10% chance each step
                self._update_target_model()
                logger.debug("Updated target model")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _replay(self):
        """Train the model using replay memory"""
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, actions, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, len(state)])
            next_state = np.reshape(next_state, [1, len(next_state)])
            
            # Calculate target Q values
            if not done:
                # Get max Q values for next state using target model
                next_q_values = self.target_model.predict(next_state, verbose=0)
                next_max_q = [np.amax(next_q_values[i][0]) for i in range(self.n_cpus)]
                
                # Calculate targets using Bellman equation: r + gamma * max(Q')
                targets = [reward + self.gamma * next_max for next_max in next_max_q]
            else:
                # If done, target is just the reward
                targets = [reward] * self.n_cpus
            
            # Get current Q values
            current_q_values = self.model.predict(state, verbose=0)
            
            # Create target vectors
            target_f = []
            for i in range(self.n_cpus):
                core_q = current_q_values[i][0].copy()
                core_q[actions[i]] = targets[i]
                target_f.append(np.reshape(core_q, (1, self.n_freq_options)))
            
            # Train model
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def save(self, path):
        """Save model weights to file"""
        self.model.save_weights(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights from file"""
        self.model.load_weights(path)
        # Also update target model
        self._update_target_model()
        logger.info(f"Model loaded from {path}")
