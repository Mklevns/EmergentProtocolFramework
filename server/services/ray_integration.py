
"""
Ray 2.9.0 RLModule Integration for Bio-Inspired MARL
Integrates existing bio-inspired components with Ray's new API stack
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from .marl_framework import PheromoneAttentionNetwork, NeuralPlasticityMemory

class BioInspiredRLModule(TorchRLModule):
    """Bio-inspired RLModule integrating pheromone attention and neural plasticity"""
    
    def __init__(self, config: SingleAgentRLModuleSpec):
        super().__init__(config)
        
        # Get model configuration
        model_config = getattr(config, "model_config_dict", {})
        
        # Agent configuration
        obs_dim = config.observation_space.shape[0]
        action_dim = config.action_space.n
        hidden_dim = model_config.get("hidden_dim", 256)
        
        # Bio-inspired components from your existing framework
        self.pheromone_attention = PheromoneAttentionNetwork(
            hidden_dim=hidden_dim,
            num_heads=model_config.get("num_heads", 8)
        )
        
        self.neural_plasticity = NeuralPlasticityMemory(hidden_dim)
        
        # Standard RL components
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Spatial encoding for 3D grid positions
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Memory state for plasticity
        self.register_buffer("memory_state", torch.zeros(1, hidden_dim))
        
    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        """Return initial state for recurrent components"""
        return {
            "memory_state": torch.zeros(1, self.memory_state.shape[1])
        }
    
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inference/deployment"""
        obs = batch[SampleBatch.OBS]
        
        # Basic encoding
        features = self.encoder(obs)
        
        # Pheromone-based attention (simplified for inference)
        enhanced_features = self._apply_pheromone_attention(features, obs)
        
        # Policy output
        action_logits = self.policy_head(enhanced_features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits
        }
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for exploration during rollouts"""
        # Similar to inference but with exploration noise
        return self._forward_inference(batch)
    
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for training with full bio-inspired processing"""
        obs = batch[SampleBatch.OBS]
        
        # Handle state for neural plasticity
        if "state_in" in batch and batch["state_in"]:
            memory_state = batch["state_in"]["memory_state"]
        else:
            memory_state = self.get_initial_state()["memory_state"]
            memory_state = memory_state.expand(obs.shape[0], -1)
        
        # Encode observations
        features = self.encoder(obs)
        
        # Apply bio-inspired components
        enhanced_features = self._apply_bio_inspired_processing(
            features, obs, memory_state
        )
        
        # Neural plasticity update
        new_memory_state = self.neural_plasticity(
            enhanced_features.unsqueeze(1), memory_state
        )
        
        # Output heads
        action_logits = self.policy_head(enhanced_features)
        values = self.value_head(enhanced_features).squeeze(-1)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: values,
            "state_out": {"memory_state": new_memory_state},
            "bio_metrics": self._compute_bio_metrics(enhanced_features)
        }
    
    def _apply_pheromone_attention(self, features: torch.Tensor, 
                                 obs: torch.Tensor) -> torch.Tensor:
        """Apply pheromone-based attention mechanism"""
        batch_size = features.shape[0]
        
        # Extract spatial information from observations
        # Assuming obs contains position data
        positions = obs[:, :3]  # First 3 dims are x,y,z positions
        
        # Create distance mask for local neighborhoods
        distance_mask = self._compute_distance_mask(positions)
        
        # Apply pheromone attention
        attended_features, attention_weights = self.pheromone_attention(
            query=features.unsqueeze(1),
            key=features.unsqueeze(1),
            value=features.unsqueeze(1),
            positions=positions.unsqueeze(1),
            distance_mask=distance_mask.unsqueeze(1)
        )
        
        return attended_features.squeeze(1)
    
    def _apply_bio_inspired_processing(self, features: torch.Tensor, 
                                     obs: torch.Tensor,
                                     memory_state: torch.Tensor) -> torch.Tensor:
        """Apply full bio-inspired processing for training"""
        
        # Pheromone attention
        attended_features = self._apply_pheromone_attention(features, obs)
        
        # Spatial encoding
        positions = obs[:, :3]
        spatial_features = self.spatial_encoder(positions)
        
        # Combine features
        combined_features = attended_features + spatial_features
        
        return combined_features
    
    def _compute_distance_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute distance-based attention mask"""
        batch_size = positions.shape[0]
        
        # Compute pairwise distances
        pos_expanded = positions.unsqueeze(1)  # [B, 1, 3]
        pos_expanded_t = positions.unsqueeze(0)  # [1, B, 3]
        
        distances = torch.norm(pos_expanded - pos_expanded_t, dim=-1)
        
        # Create mask for communication range (e.g., distance < 2.0)
        mask = (distances < 2.0).float()
        
        return mask
    
    def _compute_bio_metrics(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute bio-inspired metrics for analysis"""
        return {
            "feature_sparsity": (features.abs() < 0.1).float().mean(),
            "activation_entropy": self._compute_entropy(features),
            "spatial_coherence": features.std(dim=0).mean()
        }
    
    def _compute_entropy(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute entropy of tensor activations"""
        # Normalize to probabilities
        probs = torch.softmax(tensor, dim=-1)
        log_probs = torch.log_softmax(tensor, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

class MultiAgentBioInspiredConfig:
    """Configuration helper for multi-agent bio-inspired setup"""
    
    @staticmethod
    def create_config(env_name: str, num_agents: int = 30):
        """Create Ray 2.9.0 compatible configuration"""
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
        
        # Sample environment to get spaces
        # You'd replace this with your actual environment
        import gymnasium as gym
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        action_space = gym.spaces.Discrete(5)
        
        config = (
            PPOConfig()
            .environment(env=env_name)
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
                            observation_space=obs_space,
                            action_space=action_space,
                            model_config_dict={
                                "hidden_dim": 256,
                                "num_heads": 8,
                                "use_bio_inspired": True
                            }
                        )
                    }
                )
            )
            .training(
                lr=3e-4,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                gamma=0.99,
                lambda_=0.95,
            )
            .rollouts(
                num_rollout_workers=4,
                num_envs_per_worker=1,
            )
            .experimental(_enable_new_api_stack=True)
        )
        
        return config

def integrate_with_existing_framework():
    """Integration point with your existing MARL framework"""
    from .marl_framework import get_framework
    
    framework = get_framework()
    
    # Create Ray-compatible environment
    from ray.tune.registry import register_env
    
    def env_creator(config):
        # Create environment using your existing framework
        return framework.create_environment(config)
    
    register_env("bio_inspired_marl", env_creator)
    
    # Create configuration
    config = MultiAgentBioInspiredConfig.create_config("bio_inspired_marl")
    
    return config.build()
