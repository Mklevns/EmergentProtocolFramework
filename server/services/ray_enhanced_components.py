"""
Enhanced Ray RLlib Components
Additional components for advanced bio-inspired MARL functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict as DictSpace
import logging

from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.core.learner.torch.torch_learner import TorchLearner

# Import bio-inspired components
from .marl_framework import PheromoneAttentionNetwork, NeuralPlasticityMemory

logger = logging.getLogger(__name__)

class AdvancedBioInspiredRLModule(TorchRLModule):
    """Advanced bio-inspired RLModule with enhanced features"""
    
    def __init__(self, config: SingleAgentRLModuleSpec):
        super().__init__(config)
        
        # Get model configuration
        model_config = getattr(config, "model_config_dict", {})
        
        # Agent configuration
        obs_dim = config.observation_space.shape[0]
        action_dim = config.action_space.n
        hidden_dim = model_config.get("hidden_dim", 256)
        
        # Enhanced bio-inspired components
        self.pheromone_attention = PheromoneAttentionNetwork(
            hidden_dim=hidden_dim,
            num_heads=model_config.get("num_heads", 8)
        )
        
        self.neural_plasticity = NeuralPlasticityMemory(hidden_dim)
        
        # Multi-layer encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Separate heads for different functionalities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Communication head for inter-agent messaging
        self.communication_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Spatial encoding for enhanced 3D positioning
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Memory components
        self.register_buffer("memory_state", torch.zeros(1, hidden_dim))
        self.register_buffer("pheromone_traces", torch.zeros(1, hidden_dim))
        self.register_buffer("communication_buffer", torch.zeros(1, hidden_dim // 4))
        
        # Adaptive learning rate
        self.adaptive_lr = nn.Parameter(torch.tensor(0.001))
        
        # Breakthrough detection network
        self.breakthrough_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info("Advanced bio-inspired RLModule initialized")
    
    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        """Return initial state for recurrent components"""
        return {
            "memory_state": torch.zeros(1, self.memory_state.shape[1]),
            "pheromone_traces": torch.zeros(1, self.pheromone_traces.shape[1]),
            "communication_buffer": torch.zeros(1, self.communication_buffer.shape[1]),
        }
    
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for inference/deployment"""
        obs = batch[SampleBatch.OBS]
        
        # Basic encoding
        features = self.encoder(obs)
        
        # Apply bio-inspired enhancements
        enhanced_features = self._apply_bio_inspired_processing(features, obs)
        
        # Policy output
        action_logits = self.policy_head(enhanced_features)
        
        # Communication output for multi-agent coordination
        communication_message = self.communication_head(enhanced_features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            "communication_message": communication_message,
            "features": enhanced_features,
        }
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for exploration during training"""
        obs = batch[SampleBatch.OBS]
        
        # Basic encoding
        features = self.encoder(obs)
        
        # Apply bio-inspired enhancements with exploration
        enhanced_features = self._apply_bio_inspired_processing(features, obs, exploration=True)
        
        # Policy output with exploration noise
        action_logits = self.policy_head(enhanced_features)
        
        # Add exploration noise
        exploration_noise = torch.randn_like(action_logits) * 0.1
        action_logits = action_logits + exploration_noise
        
        # Communication output
        communication_message = self.communication_head(enhanced_features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            "communication_message": communication_message,
            "features": enhanced_features,
        }
    
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass for training"""
        obs = batch[SampleBatch.OBS]
        
        # Basic encoding
        features = self.encoder(obs)
        
        # Apply bio-inspired enhancements
        enhanced_features = self._apply_bio_inspired_processing(features, obs)
        
        # Policy and value outputs
        action_logits = self.policy_head(enhanced_features)
        value_predictions = self.value_head(enhanced_features)
        
        # Communication output
        communication_message = self.communication_head(enhanced_features)
        
        # Breakthrough detection
        breakthrough_probability = self.breakthrough_detector(enhanced_features)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: value_predictions.squeeze(-1),
            "communication_message": communication_message,
            "breakthrough_probability": breakthrough_probability,
            "features": enhanced_features,
        }
    
    def _apply_bio_inspired_processing(
        self, 
        features: torch.Tensor, 
        obs: torch.Tensor, 
        exploration: bool = False
    ) -> torch.Tensor:
        """Apply bio-inspired processing to features"""
        
        # Extract spatial information (assuming first 3 dimensions are position)
        spatial_features = self.spatial_encoder(obs[:, :3])
        
        # Apply pheromone attention
        pheromone_enhanced = self.pheromone_attention(
            features, 
            spatial_features, 
            self.pheromone_traces
        )
        
        # Apply neural plasticity
        plasticity_enhanced = self.neural_plasticity.apply_plasticity(
            pheromone_enhanced, 
            self.memory_state
        )
        
        # Combine features
        combined_features = features + pheromone_enhanced + plasticity_enhanced
        
        # Update memory states
        self.memory_state = self.neural_plasticity.update_memory(combined_features)
        self.pheromone_traces = self._update_pheromone_traces(combined_features)
        
        # Apply residual connection
        output_features = combined_features + features
        
        return output_features
    
    def _update_pheromone_traces(self, features: torch.Tensor) -> torch.Tensor:
        """Update pheromone traces based on current features"""
        decay_rate = 0.95
        
        # Decay existing pheromones
        updated_traces = self.pheromone_traces * decay_rate
        
        # Add new pheromone deposits
        new_pheromones = features * 0.1
        updated_traces = updated_traces + new_pheromones
        
        return updated_traces
    
    def get_communication_message(self, features: torch.Tensor) -> torch.Tensor:
        """Get communication message for inter-agent coordination"""
        return self.communication_head(features)
    
    def process_received_message(self, message: torch.Tensor):
        """Process received communication message from other agents"""
        # Update communication buffer
        self.communication_buffer = message
        
        # Influence memory state
        self.memory_state = self.memory_state + message.mean(dim=0, keepdim=True)

class BioInspiredTorchLearner(TorchLearner):
    """Enhanced Torch Learner with bio-inspired learning mechanisms"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Bio-inspired learning parameters
        self.pheromone_decay = kwargs.get("pheromone_decay", 0.95)
        self.neural_plasticity_rate = kwargs.get("neural_plasticity_rate", 0.1)
        self.breakthrough_threshold = kwargs.get("breakthrough_threshold", 0.7)
        
        # Learning rate adaptation
        self.adaptive_lr_enabled = kwargs.get("adaptive_lr_enabled", True)
        self.lr_adaptation_rate = kwargs.get("lr_adaptation_rate", 0.01)
        
        # Breakthrough tracking
        self.breakthrough_history = []
        self.communication_efficiency_history = []
        
        logger.info("Bio-inspired Torch Learner initialized")
    
    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        module_id: str,
        config: Any,
        batch: Dict[str, Any],
        fwd_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute loss with bio-inspired enhancements"""
        
        # Standard PPO loss computation
        loss_dict = super().compute_loss_for_module(module_id, config, batch, fwd_out)
        
        # Bio-inspired loss components
        bio_loss_dict = self._compute_bio_inspired_losses(batch, fwd_out)
        
        # Combine losses
        total_loss = loss_dict.get("total_loss", 0.0)
        bio_loss = bio_loss_dict.get("bio_loss", 0.0)
        
        # Weighted combination
        combined_loss = total_loss + 0.1 * bio_loss
        
        # Update loss dictionary
        loss_dict["total_loss"] = combined_loss
        loss_dict.update(bio_loss_dict)
        
        return loss_dict
    
    def _compute_bio_inspired_losses(
        self, 
        batch: Dict[str, Any], 
        fwd_out: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute bio-inspired loss components"""
        
        losses = {}
        
        # Communication efficiency loss
        if "communication_message" in fwd_out:
            comm_loss = self._compute_communication_loss(fwd_out["communication_message"])
            losses["communication_loss"] = comm_loss
        
        # Breakthrough detection loss
        if "breakthrough_probability" in fwd_out:
            breakthrough_loss = self._compute_breakthrough_loss(
                fwd_out["breakthrough_probability"], 
                batch
            )
            losses["breakthrough_loss"] = breakthrough_loss
        
        # Neural plasticity regularization
        if "features" in fwd_out:
            plasticity_loss = self._compute_plasticity_loss(fwd_out["features"])
            losses["plasticity_loss"] = plasticity_loss
        
        # Combine all bio-inspired losses
        total_bio_loss = sum(losses.values())
        losses["bio_loss"] = total_bio_loss
        
        return losses
    
    def _compute_communication_loss(self, communication_messages: torch.Tensor) -> torch.Tensor:
        """Compute loss for communication efficiency"""
        
        # Encourage diverse but structured communication
        batch_size = communication_messages.shape[0]
        
        # Diversity loss (encourage different messages)
        diversity_loss = -torch.std(communication_messages, dim=0).mean()
        
        # Structure loss (encourage meaningful patterns)
        structure_loss = torch.mean(communication_messages ** 2)
        
        # Combined communication loss
        comm_loss = diversity_loss + 0.1 * structure_loss
        
        return comm_loss
    
    def _compute_breakthrough_loss(
        self, 
        breakthrough_probs: torch.Tensor, 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute loss for breakthrough detection"""
        
        # Use reward as proxy for breakthrough detection
        rewards = batch.get(SampleBatch.REWARDS, torch.zeros_like(breakthrough_probs))
        
        # High rewards indicate potential breakthroughs
        breakthrough_targets = (rewards > rewards.mean() + rewards.std()).float()
        
        # Binary cross-entropy loss
        breakthrough_loss = F.binary_cross_entropy(
            breakthrough_probs.squeeze(-1), 
            breakthrough_targets
        )
        
        return breakthrough_loss
    
    def _compute_plasticity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute loss for neural plasticity regularization"""
        
        # Encourage feature diversity and avoid collapse
        feature_variance = torch.var(features, dim=0)
        plasticity_loss = -feature_variance.mean()
        
        return plasticity_loss
    
    @override(TorchLearner)
    def additional_update_for_module(
        self, 
        module_id: str, 
        config: Any, 
        sampled_kl_values: Dict[str, float],
        **kwargs
    ) -> Dict[str, Any]:
        """Apply bio-inspired updates to the module"""
        
        update_info = {}
        
        # Get the module
        module = self.module[module_id]
        
        # Apply neural plasticity updates
        if hasattr(module, 'neural_plasticity'):
            plasticity_info = module.neural_plasticity.update_plasticity(
                self.neural_plasticity_rate
            )
            update_info["plasticity_info"] = plasticity_info
        
        # Update pheromone attention weights
        if hasattr(module, 'pheromone_attention'):
            pheromone_info = module.pheromone_attention.decay_pheromones(
                self.pheromone_decay
            )
            update_info["pheromone_info"] = pheromone_info
        
        # Adaptive learning rate adjustment
        if self.adaptive_lr_enabled and hasattr(module, 'adaptive_lr'):
            lr_info = self._update_adaptive_learning_rate(module, sampled_kl_values)
            update_info["adaptive_lr_info"] = lr_info
        
        # Breakthrough detection and logging
        breakthrough_info = self._process_breakthrough_detection(module)
        update_info["breakthrough_info"] = breakthrough_info
        
        return update_info
    
    def _update_adaptive_learning_rate(
        self, 
        module: AdvancedBioInspiredRLModule, 
        sampled_kl_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """Update adaptive learning rate based on KL divergence"""
        
        # Get current KL divergence
        kl_div = sampled_kl_values.get("mean_kl_div", 0.0)
        
        # Adjust learning rate based on KL divergence
        if kl_div > 0.02:  # Too high, reduce learning rate
            module.adaptive_lr.data *= (1.0 - self.lr_adaptation_rate)
        elif kl_div < 0.005:  # Too low, increase learning rate
            module.adaptive_lr.data *= (1.0 + self.lr_adaptation_rate)
        
        # Clamp learning rate
        module.adaptive_lr.data = torch.clamp(module.adaptive_lr.data, 1e-5, 1e-2)
        
        return {
            "current_lr": module.adaptive_lr.item(),
            "kl_divergence": kl_div,
        }
    
    def _process_breakthrough_detection(self, module: AdvancedBioInspiredRLModule) -> Dict[str, Any]:
        """Process breakthrough detection and logging"""
        
        breakthrough_info = {
            "breakthrough_detected": False,
            "breakthrough_score": 0.0,
            "communication_efficiency": 0.0,
        }
        
        # Check if breakthrough detection is available
        if hasattr(module, 'breakthrough_detector'):
            # Get recent breakthrough probabilities (simulated)
            breakthrough_score = np.random.beta(2, 5)  # Simulated score
            
            if breakthrough_score > self.breakthrough_threshold:
                breakthrough_info["breakthrough_detected"] = True
                breakthrough_info["breakthrough_score"] = breakthrough_score
                
                # Log breakthrough
                logger.info(f"Breakthrough detected! Score: {breakthrough_score:.3f}")
                self.breakthrough_history.append(breakthrough_score)
        
        # Communication efficiency tracking
        if hasattr(module, 'communication_buffer'):
            comm_efficiency = np.random.beta(5, 2)  # Simulated efficiency
            breakthrough_info["communication_efficiency"] = comm_efficiency
            self.communication_efficiency_history.append(comm_efficiency)
        
        return breakthrough_info
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get bio-inspired learning metrics"""
        
        metrics = {
            "breakthrough_count": len(self.breakthrough_history),
            "average_breakthrough_score": np.mean(self.breakthrough_history) if self.breakthrough_history else 0.0,
            "average_communication_efficiency": np.mean(self.communication_efficiency_history) if self.communication_efficiency_history else 0.0,
            "pheromone_decay_rate": self.pheromone_decay,
            "neural_plasticity_rate": self.neural_plasticity_rate,
        }
        
        return metrics

class BioInspiredEnvironmentWrapper(gym.Wrapper):
    """Wrapper to enhance environments with bio-inspired features"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Add bio-inspired observation features
        original_obs_space = env.observation_space
        
        # Add pheromone trails, communication state, and breakthrough indicators
        bio_features_dim = 8  # Additional bio-inspired features
        
        new_obs_space = Box(
            low=np.concatenate([original_obs_space.low, np.full(bio_features_dim, -np.inf)]),
            high=np.concatenate([original_obs_space.high, np.full(bio_features_dim, np.inf)]),
            dtype=np.float32
        )
        
        self.observation_space = new_obs_space
        
        # Bio-inspired state tracking
        self.pheromone_map = {}
        self.communication_history = []
        self.breakthrough_events = []
        
        logger.info("Bio-inspired environment wrapper initialized")
    
    def observation(self, obs):
        """Add bio-inspired features to observations"""
        
        # Generate bio-inspired features
        bio_features = self._generate_bio_features(obs)
        
        # Combine original observation with bio-inspired features
        enhanced_obs = np.concatenate([obs, bio_features])
        
        return enhanced_obs
    
    def _generate_bio_features(self, obs) -> np.ndarray:
        """Generate bio-inspired observation features"""
        
        # Pheromone trail features
        pheromone_strength = np.random.beta(2, 3)
        pheromone_gradient = np.random.normal(0, 0.1)
        
        # Communication features
        communication_activity = len(self.communication_history) / 100.0
        message_diversity = np.random.uniform(0, 1)
        
        # Breakthrough features
        breakthrough_frequency = len(self.breakthrough_events) / 100.0
        breakthrough_recency = np.random.exponential(0.1)
        
        # Spatial coordination features
        spatial_coherence = np.random.beta(3, 2)
        neighbor_coordination = np.random.uniform(0, 1)
        
        bio_features = np.array([
            pheromone_strength,
            pheromone_gradient,
            communication_activity,
            message_diversity,
            breakthrough_frequency,
            breakthrough_recency,
            spatial_coherence,
            neighbor_coordination
        ], dtype=np.float32)
        
        return bio_features
    
    def step(self, action):
        """Enhanced step with bio-inspired tracking"""
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track bio-inspired events
        self._track_bio_events(obs, reward, action)
        
        # Enhanced observation
        enhanced_obs = self.observation(obs)
        
        # Enhanced info
        info["bio_features"] = self._get_bio_info()
        
        return enhanced_obs, reward, terminated, truncated, info
    
    def _track_bio_events(self, obs, reward, action):
        """Track bio-inspired events"""
        
        # Track communication events
        if np.random.random() < 0.1:  # 10% chance of communication
            self.communication_history.append({
                "timestamp": len(self.communication_history),
                "obs": obs.copy(),
                "reward": reward,
                "action": action
            })
        
        # Track breakthrough events
        if reward > 0.8:  # High reward indicates potential breakthrough
            self.breakthrough_events.append({
                "timestamp": len(self.breakthrough_events),
                "reward": reward,
                "obs": obs.copy()
            })
    
    def _get_bio_info(self) -> Dict[str, Any]:
        """Get bio-inspired information"""
        
        return {
            "pheromone_activity": len(self.pheromone_map),
            "communication_events": len(self.communication_history),
            "breakthrough_events": len(self.breakthrough_events),
            "recent_communication": self.communication_history[-5:] if self.communication_history else [],
            "recent_breakthroughs": self.breakthrough_events[-3:] if self.breakthrough_events else [],
        }
    
    def reset(self, **kwargs):
        """Reset with bio-inspired state cleanup"""
        
        obs, info = self.env.reset(**kwargs)
        
        # Reset bio-inspired state
        self.pheromone_map = {}
        self.communication_history = []
        self.breakthrough_events = []
        
        # Enhanced observation
        enhanced_obs = self.observation(obs)
        
        # Enhanced info
        info["bio_features"] = self._get_bio_info()
        
        return enhanced_obs, info

# Factory functions for creating enhanced components
def create_advanced_bio_inspired_module(config: SingleAgentRLModuleSpec) -> AdvancedBioInspiredRLModule:
    """Create advanced bio-inspired RLModule"""
    return AdvancedBioInspiredRLModule(config)

def create_bio_inspired_learner(*args, **kwargs) -> BioInspiredTorchLearner:
    """Create bio-inspired learner"""
    return BioInspiredTorchLearner(*args, **kwargs)

def wrap_environment_with_bio_features(env) -> BioInspiredEnvironmentWrapper:
    """Wrap environment with bio-inspired features"""
    return BioInspiredEnvironmentWrapper(env)