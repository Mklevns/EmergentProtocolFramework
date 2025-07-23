"""
Full Ray RLlib Integration with Algorithm and Learner Classes
Production-ready implementation for scalable bio-inspired MARL training
"""

import os
import ray
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict as DictSpace
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import time

# Ray RLlib imports
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

# Bio-inspired components
import sys
import os

# Add server directory to Python path for absolute imports  
server_dir = os.path.dirname(os.path.dirname(__file__))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

from services.ray_integration import BioInspiredRLModule
from services.marl_framework import PheromoneAttentionNetwork, NeuralPlasticityMemory

logger = logging.getLogger(__name__)

@dataclass
class RayTrainingConfig:
    """Configuration for Ray RLlib training"""
    experiment_name: str = "bio_inspired_marl"
    num_agents: int = 30
    grid_size: Tuple[int, int, int] = (4, 3, 3)
    max_episode_steps: int = 500
    total_timesteps: int = 1000000
    
    # PPO-specific settings
    learning_rate: float = 3e-4
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    gamma: float = 0.99
    lambda_: float = 0.95
    
    # Rollout settings
    num_rollout_workers: int = 4
    num_envs_per_worker: int = 1
    rollout_fragment_length: int = 200
    
    # Bio-inspired settings
    hidden_dim: int = 256
    num_attention_heads: int = 8
    pheromone_decay: float = 0.95
    neural_plasticity_rate: float = 0.1
    communication_range: float = 2.0
    
    # Training settings
    checkpoint_frequency: int = 10
    evaluation_interval: int = 5
    evaluation_duration: int = 10

class BioInspiredMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment for bio-inspired MARL training"""
    
    def __init__(self, config: EnvContext):
        super().__init__()
        
        self.config = config
        self.num_agents = config.get("num_agents", 30)
        self.grid_size = config.get("grid_size", (4, 3, 3))
        self.max_episode_steps = config.get("max_episode_steps", 500)
        self.communication_range = config.get("communication_range", 2.0)
        
        # Initialize agent positions and states
        self.agent_positions = self._initialize_agent_positions()
        self.pheromone_trails = np.zeros(self.grid_size)
        self.shared_memory = {}
        self.step_count = 0
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = Discrete(5)  # 5 possible actions
        
        # Agent IDs
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self._agent_ids = set(self.agent_ids)
        
        # Episode metrics
        self.episode_metrics = {
            "total_reward": 0.0,
            "communication_events": 0,
            "breakthrough_events": 0,
            "coordination_score": 0.0,
        }
        
        logger.info(f"BioInspiredMultiAgentEnv initialized with {self.num_agents} agents")
    
    def _initialize_agent_positions(self) -> Dict[str, np.ndarray]:
        """Initialize agent positions in 3D grid"""
        positions = {}
        x_max, y_max, z_max = self.grid_size
        
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            # Distribute agents across the grid
            x = i % x_max
            y = (i // x_max) % y_max
            z = (i // (x_max * y_max)) % z_max
            positions[agent_id] = np.array([x, y, z], dtype=np.float32)
        
        return positions
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset environment state
        self.step_count = 0
        self.pheromone_trails = np.zeros(self.grid_size)
        self.shared_memory = {}
        self.episode_metrics = {
            "total_reward": 0.0,
            "communication_events": 0,
            "breakthrough_events": 0,
            "coordination_score": 0.0,
        }
        
        # Generate initial observations
        observations = {}
        infos = {}
        
        for agent_id in self.agent_ids:
            observations[agent_id] = self._get_observation(agent_id)
            infos[agent_id] = {"position": self.agent_positions[agent_id].copy()}
        
        return observations, infos
    
    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step"""
        self.step_count += 1
        
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        # Process actions for each agent
        for agent_id, action in action_dict.items():
            # Update agent position based on action
            self._update_agent_position(agent_id, action)
            
            # Calculate reward
            reward = self._calculate_reward(agent_id, action)
            rewards[agent_id] = reward
            
            # Get new observation
            observations[agent_id] = self._get_observation(agent_id)
            
            # Check termination conditions
            terminateds[agent_id] = False
            truncateds[agent_id] = self.step_count >= self.max_episode_steps
            
            # Agent-specific info
            infos[agent_id] = {
                "position": self.agent_positions[agent_id].copy(),
                "reward_breakdown": self._get_reward_breakdown(agent_id, action),
                "neighbors": self._get_neighbors(agent_id),
            }
        
        # Update environment state
        self._update_pheromone_trails()
        self._update_shared_memory()
        
        # Episode-level termination
        episode_done = self.step_count >= self.max_episode_steps
        terminateds["__all__"] = episode_done
        truncateds["__all__"] = episode_done
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent"""
        position = self.agent_positions[agent_id]
        
        # Local pheromone levels
        local_pheromones = self._get_local_pheromones(position)
        
        # Neighbor information
        neighbors = self._get_neighbors(agent_id)
        neighbor_count = len(neighbors)
        
        # Shared memory access
        memory_access = self._get_memory_access(agent_id)
        
        # Combine into observation vector
        obs = np.concatenate([
            position,  # 3D position (3 values)
            local_pheromones,  # Local pheromone levels (3 values)
            [neighbor_count],  # Number of neighbors (1 value)
            [self.step_count / self.max_episode_steps],  # Progress (1 value)
            memory_access,  # Memory access features (4 values)
        ])
        
        return obs.astype(np.float32)
    
    def _update_agent_position(self, agent_id: str, action: int):
        """Update agent position based on action"""
        position = self.agent_positions[agent_id]
        
        # Define movement directions
        movements = {
            0: np.array([0, 0, 0]),   # Stay
            1: np.array([1, 0, 0]),   # Move +X
            2: np.array([-1, 0, 0]),  # Move -X
            3: np.array([0, 1, 0]),   # Move +Y
            4: np.array([0, -1, 0]),  # Move -Y
        }
        
        if action in movements:
            new_position = position + movements[action]
            
            # Clamp to grid bounds
            x_max, y_max, z_max = self.grid_size
            new_position = np.clip(new_position, [0, 0, 0], [x_max-1, y_max-1, z_max-1])
            
            self.agent_positions[agent_id] = new_position
    
    def _calculate_reward(self, agent_id: str, action: int) -> float:
        """Calculate reward for agent action"""
        reward = 0.0
        
        # Base reward for staying active
        reward += 0.1
        
        # Reward for coordination with neighbors
        neighbors = self._get_neighbors(agent_id)
        coordination_reward = len(neighbors) * 0.2
        reward += coordination_reward
        
        # Reward for exploration
        position = self.agent_positions[agent_id]
        exploration_reward = self._calculate_exploration_reward(position)
        reward += exploration_reward
        
        # Penalty for collisions
        collision_penalty = self._calculate_collision_penalty(agent_id)
        reward -= collision_penalty
        
        # Bio-inspired rewards
        pheromone_reward = self._calculate_pheromone_reward(agent_id)
        reward += pheromone_reward
        
        return reward
    
    def _get_neighbors(self, agent_id: str) -> List[str]:
        """Get neighboring agents within communication range"""
        neighbors = []
        agent_pos = self.agent_positions[agent_id]
        
        for other_id, other_pos in self.agent_positions.items():
            if other_id != agent_id:
                distance = np.linalg.norm(agent_pos - other_pos)
                if distance <= self.communication_range:
                    neighbors.append(other_id)
        
        return neighbors
    
    def _get_local_pheromones(self, position: np.ndarray) -> np.ndarray:
        """Get local pheromone levels around position"""
        x, y, z = position.astype(int)
        
        # Sample pheromone levels in 3x3 area
        pheromone_levels = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                px, py = x + dx, y + dy
                if 0 <= px < self.grid_size[0] and 0 <= py < self.grid_size[1]:
                    pheromone_levels.append(self.pheromone_trails[px, py, z])
                else:
                    pheromone_levels.append(0.0)
        
        return np.array(pheromone_levels[:3])  # Return first 3 values
    
    def _get_memory_access(self, agent_id: str) -> np.ndarray:
        """Get memory access features for agent"""
        # Simple memory access representation
        memory_features = [
            len(self.shared_memory),  # Total memory entries
            self.shared_memory.get(agent_id, 0),  # Agent's memory usage
            self.episode_metrics["communication_events"],  # Communication events
            self.episode_metrics["coordination_score"],  # Coordination score
        ]
        
        return np.array(memory_features)
    
    def _calculate_exploration_reward(self, position: np.ndarray) -> float:
        """Calculate exploration reward based on position"""
        # Simple exploration reward - further from center gets more reward
        center = np.array(self.grid_size) / 2
        distance_from_center = np.linalg.norm(position - center)
        return distance_from_center * 0.1
    
    def _calculate_collision_penalty(self, agent_id: str) -> float:
        """Calculate collision penalty"""
        position = self.agent_positions[agent_id]
        
        # Check for collisions with other agents
        collisions = 0
        for other_id, other_pos in self.agent_positions.items():
            if other_id != agent_id and np.array_equal(position, other_pos):
                collisions += 1
        
        return collisions * 0.5
    
    def _calculate_pheromone_reward(self, agent_id: str) -> float:
        """Calculate pheromone-based reward"""
        position = self.agent_positions[agent_id]
        x, y, z = position.astype(int)
        
        # Reward based on pheromone levels
        pheromone_level = self.pheromone_trails[x, y, z]
        return pheromone_level * 0.3
    
    def _update_pheromone_trails(self):
        """Update pheromone trails based on agent positions"""
        # Decay existing pheromones
        self.pheromone_trails *= 0.95
        
        # Add new pheromones at agent positions
        for agent_id, position in self.agent_positions.items():
            x, y, z = position.astype(int)
            self.pheromone_trails[x, y, z] += 0.1
    
    def _update_shared_memory(self):
        """Update shared memory based on agent interactions"""
        # Update episode metrics
        self.episode_metrics["communication_events"] += len(self.agent_ids) * 0.1
        self.episode_metrics["coordination_score"] = self._calculate_coordination_score()
    
    def _calculate_coordination_score(self) -> float:
        """Calculate overall coordination score"""
        total_neighbors = sum(len(self._get_neighbors(agent_id)) for agent_id in self.agent_ids)
        return total_neighbors / len(self.agent_ids)
    
    def _get_reward_breakdown(self, agent_id: str, action: int) -> Dict[str, float]:
        """Get detailed reward breakdown for debugging"""
        return {
            "base_reward": 0.1,
            "coordination_reward": len(self._get_neighbors(agent_id)) * 0.2,
            "exploration_reward": self._calculate_exploration_reward(self.agent_positions[agent_id]),
            "collision_penalty": -self._calculate_collision_penalty(agent_id),
            "pheromone_reward": self._calculate_pheromone_reward(agent_id),
        }

class BioInspiredLearner(Learner):
    """Custom Learner for bio-inspired MARL training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pheromone_decay = kwargs.get("pheromone_decay", 0.95)
        self.neural_plasticity_rate = kwargs.get("neural_plasticity_rate", 0.1)
        
    @override(Learner)
    def additional_update_for_module(
        self, 
        module_id: str, 
        config: Dict[str, Any], 
        sampled_kl_values: Dict[str, float],
        **kwargs
    ) -> Dict[str, Any]:
        """Additional bio-inspired updates for the module"""
        
        # Apply neural plasticity updates
        if hasattr(self.module[module_id], 'neural_plasticity'):
            self.module[module_id].neural_plasticity.update_plasticity(
                self.neural_plasticity_rate
            )
        
        # Update pheromone attention weights
        if hasattr(self.module[module_id], 'pheromone_attention'):
            self.module[module_id].pheromone_attention.decay_pheromones(
                self.pheromone_decay
            )
        
        return {}

class FullRayIntegration:
    """Full Ray RLlib integration manager"""
    
    def __init__(self, config: RayTrainingConfig):
        self.config = config
        self.algorithm: Optional[Algorithm] = None
        self.checkpoint_dir = Path(f"./checkpoints/{config.experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                local_mode=False,
                num_cpus=os.cpu_count(),
                object_store_memory=1000000000,  # 1GB
                _temp_dir="/tmp/ray"
            )
            logger.info("Ray initialized successfully")
    
    def create_algorithm(self) -> Algorithm:
        """Create and configure the Ray Algorithm"""
        
        # Register the environment
        register_env("bio_inspired_marl", lambda config: BioInspiredMultiAgentEnv(config))
        
        # Create algorithm configuration
        config = (
            PPOConfig()
            .environment(
                env="bio_inspired_marl",
                env_config={
                    "num_agents": self.config.num_agents,
                    "grid_size": self.config.grid_size,
                    "max_episode_steps": self.config.max_episode_steps,
                    "communication_range": self.config.communication_range,
                }
            )
            .framework("torch")
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
                policies_to_train=["shared_policy"],
            )
            .rl_module(
                rl_module_spec=MultiAgentRLModuleSpec(
                    module_specs={
                        "shared_policy": SingleAgentRLModuleSpec(
                            module_class=BioInspiredRLModule,
                            observation_space=Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                            action_space=Discrete(5),
                            model_config_dict={
                                "hidden_dim": self.config.hidden_dim,
                                "num_heads": self.config.num_attention_heads,
                                "pheromone_decay": self.config.pheromone_decay,
                                "neural_plasticity_rate": self.config.neural_plasticity_rate,
                            }
                        )
                    }
                )
            )
            .training(
                lr=self.config.learning_rate,
                train_batch_size=self.config.train_batch_size,
                sgd_minibatch_size=self.config.sgd_minibatch_size,
                num_sgd_iter=self.config.num_sgd_iter,
                gamma=self.config.gamma,
                lambda_=self.config.lambda_,
            )
            .rollouts(
                num_rollout_workers=self.config.num_rollout_workers,
                num_envs_per_worker=self.config.num_envs_per_worker,
                rollout_fragment_length=self.config.rollout_fragment_length,
            )
            .evaluation(
                evaluation_interval=self.config.evaluation_interval,
                evaluation_duration=self.config.evaluation_duration,
                evaluation_num_workers=1,
            )
            .experimental(_enable_new_api_stack=True)
        )
        
        # Build the algorithm
        self.algorithm = config.build()
        
        logger.info(f"Algorithm created: {type(self.algorithm).__name__}")
        return self.algorithm
    
    def train(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Train the algorithm for specified iterations"""
        
        if self.algorithm is None:
            self.create_algorithm()
        
        training_metrics = []
        
        logger.info(f"Starting training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            # Train one iteration
            result = self.algorithm.train()
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: "
                           f"Reward: {result['episode_reward_mean']:.2f}, "
                           f"Length: {result['episode_len_mean']:.2f}")
            
            # Save checkpoint
            if iteration % self.config.checkpoint_frequency == 0:
                checkpoint_path = self.algorithm.save(str(self.checkpoint_dir))
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Store metrics
            training_metrics.append({
                "iteration": iteration,
                "episode_reward_mean": result.get("episode_reward_mean", 0.0),
                "episode_len_mean": result.get("episode_len_mean", 0.0),
                "timesteps_total": result.get("timesteps_total", 0),
                "time_this_iter_s": result.get("time_this_iter_s", 0.0),
            })
        
        return {
            "training_completed": True,
            "total_iterations": num_iterations,
            "final_reward": training_metrics[-1]["episode_reward_mean"],
            "metrics": training_metrics,
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained algorithm"""
        
        if self.algorithm is None:
            raise ValueError("Algorithm not initialized. Call create_algorithm() first.")
        
        evaluation_results = []
        
        for episode in range(num_episodes):
            # Run evaluation episode
            result = self.algorithm.evaluate()
            
            evaluation_results.append({
                "episode": episode,
                "episode_reward_mean": result["evaluation"]["episode_reward_mean"],
                "episode_len_mean": result["evaluation"]["episode_len_mean"],
            })
        
        avg_reward = np.mean([r["episode_reward_mean"] for r in evaluation_results])
        avg_length = np.mean([r["episode_len_mean"] for r in evaluation_results])
        
        return {
            "evaluation_completed": True,
            "num_episodes": num_episodes,
            "average_reward": avg_reward,
            "average_length": avg_length,
            "episodes": evaluation_results,
        }
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save algorithm checkpoint"""
        
        if self.algorithm is None:
            raise ValueError("Algorithm not initialized")
        
        if path is None:
            path = str(self.checkpoint_dir / f"checkpoint_{int(time.time())}")
        
        checkpoint_path = self.algorithm.save(path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, path: str):
        """Load algorithm checkpoint"""
        
        if self.algorithm is None:
            self.create_algorithm()
        
        self.algorithm.restore(path)
        logger.info(f"Checkpoint loaded from: {path}")
    
    def cleanup(self):
        """Clean up resources"""
        
        if self.algorithm is not None:
            self.algorithm.cleanup()
            self.algorithm = None
        
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Ray integration cleaned up")

def create_full_ray_integration(config: Optional[RayTrainingConfig] = None) -> FullRayIntegration:
    """Create and return a full Ray integration instance"""
    
    if config is None:
        config = RayTrainingConfig()
    
    return FullRayIntegration(config)

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = RayTrainingConfig(
        experiment_name="bio_inspired_marl_test",
        num_agents=10,  # Smaller for testing
        total_timesteps=50000,
        num_rollout_workers=2,
    )
    
    # Create and test the integration
    integration = create_full_ray_integration(config)
    
    try:
        # Train the algorithm
        training_results = integration.train(num_iterations=20)
        print(f"Training completed: {training_results['training_completed']}")
        
        # Evaluate the algorithm
        evaluation_results = integration.evaluate(num_episodes=5)
        print(f"Average reward: {evaluation_results['average_reward']:.2f}")
        
        # Save checkpoint
        checkpoint_path = integration.save_checkpoint()
        print(f"Checkpoint saved: {checkpoint_path}")
        
    finally:
        # Clean up
        integration.cleanup()