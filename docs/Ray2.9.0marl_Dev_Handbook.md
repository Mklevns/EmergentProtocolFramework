# Ray 2.9.0 MARL Development Handbook

## Table of Contents

1. [Introduction & Overview](#introduction--overview)
2. [Installation & Environment Setup](#installation--environment-setup)
3. [Critical API Changes in Ray 2.9.0](#critical-api-changes-in-ray-290)
4. [RLModule Architecture Deep Dive](#rlmodule-architecture-deep-dive)
5. [Multi-Agent Reinforcement Learning Setup](#multi-agent-reinforcement-learning-setup)
6. [Common Issues & Solutions](#common-issues--solutions)
7. [Performance Optimization](#performance-optimization)
8. [Debugging Techniques](#debugging-techniques)
9. [Bio-Inspired MARL Implementation](#bio-inspired-marl-implementation)
10. [Advanced Topics](#advanced-topics)
11. [Migration from Legacy Code](#migration-from-legacy-code)
12. [Best Practices & Golden Rules](#best-practices--golden-rules)

---

## Introduction & Overview

Ray RLlib 2.9.0 marks a significant transition point in the framework's evolution, introducing the new API stack that eventually becomes mandatory in Ray 2.40+. This handbook provides comprehensive guidance for developing Multi-Agent Reinforcement Learning (MARL) systems using Ray 2.9.0.

### Key Concepts

- **RLModule**: Replaces ModelV2 as the primary neural network abstraction
- **Learner**: Handles distributed training, replacing parts of Policy and RolloutWorker
- **EnvRunner**: Collects environment samples (replaces RolloutWorker for sampling)
- **ConnectorV2**: Preprocesses data between environments and modules
- **AlgorithmConfig**: Fluent API for configuring algorithms

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Environment   │────▶│    EnvRunner    │────▶│   ConnectorV2   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Algorithm    │◀────│     Learner     │◀────│    RLModule     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Installation & Environment Setup

### System Requirements

- **Python**: 3.8-3.11 (3.9 recommended)
- **OS**: Ubuntu 20.04/22.04, macOS 10.15+, Windows 10+ (WSL2 recommended)
- **GPU**: CUDA 11.6-11.8 compatible (optional but recommended)
- **RAM**: 16GB minimum, 32GB+ recommended for MARL
- **Storage**: 10GB+ free space, NVMe SSD recommended

### Installation Commands

#### Basic Installation (CPU Only)
```bash
# Ray 2.9.0 with RLlib
pip install "ray[rllib]==2.9.0"

# Essential dependencies
pip install \
    torch>=2.0.0 \
    gymnasium==0.28.1 \
    "numpy<2.0.0" \
    supersuit==3.8.0 \
    pettingzoo \
    tensorboard
```

#### GPU Installation
```bash
# CUDA 11.8 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install Ray
pip install "ray[rllib]==2.9.0"
```

#### Complete MARL Environment
```bash
# Create virtual environment
python -m venv ray_marl_env
source ray_marl_env/bin/activate  # Linux/Mac
# or
ray_marl_env\Scripts\activate  # Windows

# Install everything
pip install --upgrade pip
pip install \
    "ray[rllib,tune,serve]==2.9.0" \
    torch>=2.0.0 \
    gymnasium==0.28.1 \
    "numpy<2.0.0" \
    supersuit==3.8.0 \
    pettingzoo \
    matplotlib \
    tensorboard \
    wandb \
    pytest
```

### ⚠️ Critical: Dependencies to AVOID

```bash
# DO NOT INSTALL these - they cause worker crashes:
# ❌ pip install torch-geometric
# ❌ pip install torch-scatter
# ❌ pip install torch-sparse
# ❌ pip install torch-cluster
```

### Verify Installation

```python
# verify_installation.py
import sys
import ray
import torch
import gymnasium as gym
import numpy as np

print(f"Python: {sys.version}")
print(f"Ray: {ray.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Gymnasium: {gym.__version__}")
print(f"NumPy: {np.__version__}")

# Test Ray initialization
ray.init(local_mode=True)
print("✅ Ray initialized successfully!")
ray.shutdown()
```

---

## Critical API Changes in Ray 2.9.0

### 1. SingleAgentRLModuleSpec Configuration

**🚨 Most Common Error Source**

❌ **Wrong** (causes AttributeError):
```python
spec = SingleAgentRLModuleSpec(
    module_class=MyRLModule,
    observation_space=obs_space,
    action_space=act_space,
    model_config=config  # ❌ Wrong parameter name!
)
```

✅ **Correct**:
```python
spec = SingleAgentRLModuleSpec(
    module_class=MyRLModule,
    observation_space=obs_space,
    action_space=act_space,
    model_config_dict=config  # ✅ Correct parameter name
)
```

**Inside Your RLModule**:
```python
def __init__(self, config: SingleAgentRLModuleSpec):
    super().__init__(config)

    # Defensive approach for compatibility
    model_config = getattr(config, "model_config_dict", {})
    if not model_config:
        # Fallback for older patterns
        model_config = getattr(config, "model_config", {})

    # Extract your parameters
    self.hidden_dim = model_config.get("hidden_dim", 128)
    self.num_layers = model_config.get("num_layers", 2)
```

### 2. RLModule vs ModelV2

**Complete Migration Example**

❌ **Old ModelV2 Pattern**:
```python
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

class MyOldModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, num_outputs)
        self.value_head = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        x = self.fc1(input_dict["obs"])
        logits = self.fc2(x)
        value = self.value_head(x)
        self._value = value.squeeze(1)
        return logits, state

    def value_function(self):
        return self._value

# Register with ModelCatalog (old way)
ModelCatalog.register_custom_model("my_model", MyOldModel)
```

✅ **New RLModule Pattern**:
```python
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical

class MyNewRLModule(TorchRLModule):
    def __init__(self, config: SingleAgentRLModuleSpec):
        super().__init__(config)

        # Get configuration
        model_config = getattr(config, "model_config_dict", {})

        # Get spaces
        obs_dim = config.observation_space.shape[0]
        action_dim = config.action_space.n

        # Build network
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def _forward_inference(self, batch: dict) -> dict:
        obs = batch[SampleBatch.OBS]
        features = self.encoder(obs)
        logits = self.policy_head(features)
        return {SampleBatch.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch: dict) -> dict:
        # Same as inference for basic exploration
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch: dict) -> dict:
        obs = batch[SampleBatch.OBS]
        features = self.encoder(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)

        return {
            SampleBatch.ACTION_DIST_INPUTS: logits,
            SampleBatch.VF_PREDS: values,
        }
```

### 3. Algorithm Configuration (Fluent API)

**Complete Configuration Example**

```python
from ray.rllib.algorithms.ppo import PPOConfig

# Create configuration
config = (
    PPOConfig()
    # Environment configuration
    .environment(
        env="CartPole-v1",
        env_config={
            "max_steps": 200,
            "reward_shaping": True,
        },
        disable_env_checking=False,
        render_env=False,
    )
    # Multi-agent configuration
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        policies_to_train=["shared_policy"],
        count_steps_by="agent_steps",
    )
    # RLModule configuration
    .rl_module(
        rl_module_spec=SingleAgentRLModuleSpec(
            module_class=MyNewRLModule,
            model_config_dict={
                "hidden_layers": [256, 256],
                "activation": "relu",
            }
        )
    )
    # Training configuration
    .training(
        lr=3e-4,
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        grad_clip=0.5,
        grad_clip_by="global_norm",
    )
    # Rollout configuration
    .rollouts(
        num_rollout_workers=4,
        num_envs_per_worker=1,
        rollout_fragment_length="auto",
        batch_mode="complete_episodes",
        enable_connectors=True,
        compress_observations=False,
    )
    # Resources configuration
    .resources(
        num_gpus=1,
        num_cpus_per_worker=1,
        num_gpus_per_learner_worker=0.25,  # Split GPU across learners
    )
    # Evaluation configuration
    .evaluation(
        evaluation_interval=10,
        evaluation_num_workers=1,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "explore": False,
            "render_env": False,
        },
    )
    # Framework configuration
    .framework(
        framework="torch",
        eager_tracing=False,
    )
    # Experimental features
    .experimental(
        _enable_new_api_stack=True,
        _disable_preprocessor_api=False,
    )
    # Callbacks
    .callbacks(MyCallbacks)
    # Debugging
    .debugging(
        seed=42,
        log_level="INFO",
    )
)

# Build algorithm
algo = config.build()
```

---

## RLModule Architecture Deep Dive

### Core Components

#### 1. Three Required Forward Methods

```python
class CompleteRLModule(TorchRLModule):
    """Complete example showing all three forward methods"""

    @override(TorchRLModule)
    def _forward_inference(self, batch: dict) -> dict:
        """
        Used for deployment/serving. No training happening.
        - Input: observations
        - Output: action distribution inputs
        - No value function needed
        """
        obs = batch[SampleBatch.OBS]
        # Process observations
        hidden = self.encoder(obs)
        action_logits = self.policy_head(hidden)

        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: dict) -> dict:
        """
        Used during rollout collection. May add exploration noise.
        - Input: observations
        - Output: action distribution inputs (possibly with exploration)
        - No value function needed
        """
        obs = batch[SampleBatch.OBS]
        hidden = self.encoder(obs)

        # Optional: Add exploration noise
        if self.add_exploration_noise:
            hidden = hidden + torch.randn_like(hidden) * 0.1

        action_logits = self.policy_head(hidden)

        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: dict) -> dict:
        """
        Used during training. Must compute everything.
        - Input: full training batch
        - Output: action distributions AND value predictions
        - All losses computed from these outputs
        """
        obs = batch[SampleBatch.OBS]

        # Shared encoder
        hidden = self.encoder(obs)

        # Policy outputs
        action_logits = self.policy_head(hidden)

        # Value outputs (required for training)
        values = self.value_head(hidden).squeeze(-1)

        # Optional: Additional outputs for custom losses
        outputs = {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: values,
        }

        # Add any custom outputs
        if self.use_auxiliary_loss:
            outputs["auxiliary_output"] = self.aux_head(hidden)

        return outputs
```

#### 2. State Management for Recurrent Models

```python
class RecurrentRLModule(TorchRLModule):
    """Example of proper state handling in RLModule"""

    def __init__(self, config: SingleAgentRLModuleSpec):
        super().__init__(config)

        self.lstm = nn.LSTM(
            input_size=config.observation_space.shape[0],
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

    def get_initial_state(self) -> dict:
        """
        Return initial hidden states.
        Shape must be [1, hidden_size] for proper batching.
        """
        return {
            "h": torch.zeros(1, 2, 128),  # [1, num_layers, hidden_size]
            "c": torch.zeros(1, 2, 128),
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: dict) -> dict:
        obs = batch[SampleBatch.OBS]
        seq_lens = batch.get(SampleBatch.SEQ_LENS, None)

        # Handle state
        if "state_in" in batch and batch["state_in"]:
            h = batch["state_in"]["h"]
            c = batch["state_in"]["c"]

            # Adjust dimensions for LSTM
            if h.dim() == 3:  # [B, 1, layers, hidden]
                h = h.squeeze(1).transpose(0, 1)  # [layers, B, hidden]
                c = c.squeeze(1).transpose(0, 1)
        else:
            # Get batch size
            B = obs.shape[0] if seq_lens is None else len(seq_lens)
            h, c = self._get_initial_state_for_batch(B)

        # Process through LSTM
        if seq_lens is not None:
            # Pack padded sequences for efficiency
            packed_obs = nn.utils.rnn.pack_padded_sequence(
                obs, seq_lens, batch_first=True, enforce_sorted=False
            )
            packed_out, (new_h, new_c) = self.lstm(packed_obs, (h, c))
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            lstm_out, (new_h, new_c) = self.lstm(obs.unsqueeze(1), (h, c))
            lstm_out = lstm_out.squeeze(1)

        # Generate outputs
        action_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out).squeeze(-1)

        # Format output state
        state_out = {
            "h": new_h.transpose(0, 1),  # [B, layers, hidden]
            "c": new_c.transpose(0, 1),
        }

        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: values,
            "state_out": state_out,
        }
```

---

## Multi-Agent Reinforcement Learning Setup

### Environment Setup

#### 1. Basic Multi-Agent Environment

```python
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np

class SimpleMultiAgentEnv(MultiAgentEnv):
    """Example multi-agent environment"""

    def __init__(self, config):
        super().__init__()
        self.n_agents = config.get("n_agents", 4)
        self.episode_length = config.get("episode_length", 100)

        # Define spaces for all agents
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        # Agent IDs
        self._agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.step_count = 0

        # Generate initial observations for all agents
        observations = {
            agent_id: self.observation_space.sample()
            for agent_id in self._agent_ids
        }

        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, infos

    def step(self, actions):
        self.step_count += 1

        # Process actions and generate new observations
        observations = {}
        rewards = {}

        for agent_id in self._agent_ids:
            if agent_id in actions:
                # Generate new observation based on action
                observations[agent_id] = self.observation_space.sample()

                # Calculate reward (example: cooperation bonus)
                own_action = actions[agent_id]
                other_actions = [
                    actions.get(other_id, 0)
                    for other_id in self._agent_ids
                    if other_id != agent_id
                ]

                # Reward cooperation
                cooperation_bonus = sum(
                    1.0 for other_action in other_actions
                    if other_action == own_action
                ) / len(other_actions)

                rewards[agent_id] = cooperation_bonus

        # Check termination
        terminated = self.step_count >= self.episode_length
        truncated = False

        # Create done dicts
        dones = {
            agent_id: terminated for agent_id in self._agent_ids
        }
        dones["__all__"] = terminated

        truncateds = {
            agent_id: truncated for agent_id in self._agent_ids
        }
        truncateds["__all__"] = truncated

        infos = {agent_id: {} for agent_id in self._agent_ids}

        return observations, rewards, dones, truncateds, infos
```

#### 2. Multi-Agent Configuration

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env

# Register environment
register_env("multi_agent_env", lambda config: SimpleMultiAgentEnv(config))

# Create sample environment to get spaces
env = SimpleMultiAgentEnv({"n_agents": 4})

# Option 1: Shared Policy (Parameter Sharing)
shared_policy_config = (
    PPOConfig()
    .environment(
        env="multi_agent_env",
        env_config={"n_agents": 4}
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        policies_to_train=["shared_policy"],
    )
    .rl_module(
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={
                "shared_policy": SingleAgentRLModuleSpec(
                    module_class=MyRLModule,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    model_config_dict={"hidden_dim": 256}
                )
            }
        )
    )
)

# Option 2: Independent Policies
independent_policy_config = (
    PPOConfig()
    .environment(
        env="multi_agent_env",
        env_config={"n_agents": 4}
    )
    .multi_agent(
        policies={
            f"policy_{i}": PolicySpec() for i in range(4)
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs:
            f"policy_{agent_id.split('_')[1]}",
        policies_to_train=[f"policy_{i}" for i in range(4)],
    )
    .rl_module(
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={
                f"policy_{i}": SingleAgentRLModuleSpec(
                    module_class=MyRLModule,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    model_config_dict={"hidden_dim": 128}
                )
                for i in range(4)
            }
        )
    )
)

# Option 3: Heterogeneous Policies
heterogeneous_config = (
    PPOConfig()
    .environment(
        env="multi_agent_env",
        env_config={"n_agents": 4}
    )
    .multi_agent(
        policies={
            "attacker": PolicySpec(),
            "defender": PolicySpec(),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs:
            "attacker" if int(agent_id.split('_')[1]) < 2 else "defender",
        policies_to_train=["attacker", "defender"],
    )
    .rl_module(
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={
                "attacker": SingleAgentRLModuleSpec(
                    module_class=AttackerModule,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    model_config_dict={"hidden_dim": 256}
                ),
                "defender": SingleAgentRLModuleSpec(
                    module_class=DefenderModule,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    model_config_dict={"hidden_dim": 128}
                )
            }
        )
    )
)
```

---

## Common Issues & Solutions

### 1. ❌ `TypeError: __init__() got an unexpected keyword argument 'model_config'`

**Problem**: Using wrong parameter name for model configuration

**Solution**:
```python
# Wrong
spec = SingleAgentRLModuleSpec(
    model_config=config  # ❌
)

# Correct
spec = SingleAgentRLModuleSpec(
    model_config_dict=config  # ✅
)
```

### 2. ❌ `AttributeError: 'SingleAgentRLModuleSpec' object has no attribute 'model_config'`

**Problem**: Trying to access config with wrong attribute name

**Solution**:
```python
# In your RLModule __init__:
model_config = getattr(config, "model_config_dict", {})
```

### 3. ❌ `RuntimeError: forward() is not implemented`

**Problem**: Not implementing the required private methods

**Solution**: Implement ALL three methods:
```python
def _forward_inference(self, batch: dict) -> dict:
    # Your inference logic

def _forward_exploration(self, batch: dict) -> dict:
    # Your exploration logic

def _forward_train(self, batch: dict) -> dict:
    # Your training logic (must include value function)
```

### 4. ❌ `KeyError: 'state_in'`

**Problem**: State not properly handled

**Solution**:
```python
# Always check if state exists
if "state_in" in batch and batch["state_in"]:
    state = batch["state_in"]
else:
    state = self.get_initial_state()
```

### 5. ❌ `ValueError: Expected state shape [...] but got [...]`

**Problem**: Incorrect state dimensions

**Solution**:
```python
def get_initial_state(self) -> dict:
    # Shape must be [1, ...] for proper batching
    return {"memory_state": torch.zeros(1, self.hidden_dim)}
```

### 6. ❌ Worker Crashes (SIGABRT)

**Problem**: torch_geometric or incompatible C++ extensions

**Solution**:
```bash
# Remove problematic packages
pip uninstall torch-geometric torch-scatter torch-sparse torch-cluster -y

# Use custom implementations instead
```

### 7. ❌ Gymnasium Float32/64 Warnings

**Problem**: Environment returns wrong dtype

**Solution**:
```python
# In your environment
self.observation_space = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(10,), dtype=np.float32  # Explicit float32
)

# When returning observations
obs = np.array([...], dtype=np.float32)
```

### 8. ❌ `RuntimeError: CUDA out of memory`

**Problem**: Batch size too large for GPU

**Solution**:
```python
config.training(
    train_batch_size=2000,  # Reduce from 4000
    sgd_minibatch_size=64   # Reduce from 128
)

# Or use gradient accumulation
config.training(
    _enable_learner_api=True,
    num_sgd_iter=20,  # More iterations with smaller batches
)
```

### 9. ❌ `AssertionError: New API stack not enabled`

**Problem**: Not enabling experimental features

**Solution**:
```python
config.experimental(_enable_new_api_stack=True)
```

### 10. ❌ `ValueError: observation space mismatch`

**Problem**: Environment and module spaces don't match

**Solution**:
```python
# Get space from actual environment
env = YourEnvironment()
if hasattr(env, "observation_space"):
    # Single agent
    obs_space = env.observation_space
else:
    # Multi-agent - get from one agent
    obs_space = env.observation_space["agent_0"]

spec = SingleAgentRLModuleSpec(
    observation_space=obs_space,  # Use the same space
    action_space=env.action_space,
    ...
)
```

---

## Performance Optimization

### GPU Optimization

#### 1. Efficient GPU Usage
```python
config = (
    PPOConfig()
    .resources(
        num_gpus=1,  # Total GPUs
        num_gpus_per_learner_worker=0.5,  # Split across learners
    )
    .learners(
        num_learners=2,  # 2 learners, each gets 0.5 GPU
    )
    .training(
        train_batch_size=8192,  # Larger batches for GPU
        sgd_minibatch_size=256,
        num_sgd_iter=30,  # More iterations per batch
    )
)
```

#### 2. Mixed Precision Training
```python
class OptimizedRLModule(TorchRLModule):
    def __init__(self, config):
        super().__init__(config)

        # Enable mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

    @override(TorchRLModule)
    def _forward_train(self, batch: dict) -> dict:
        with torch.cuda.amp.autocast():
            # Your forward pass
            obs = batch[SampleBatch.OBS]
            # ... rest of forward pass
```

### CPU Optimization

#### 1. Vectorized Environments
```python
config.rollouts(
    num_rollout_workers=8,  # More workers
    num_envs_per_worker=4,  # Multiple envs per worker
    remote_worker_envs=True,  # Remote env execution
)
```

#### 2. Observation Compression
```python
config.rollouts(
    compress_observations=True,  # Compress large observations
    compression_type="lz4",  # Fast compression
)
```

### Memory Optimization

#### 1. Replay Buffer Management
```python
# For off-policy algorithms
config.training(
    replay_buffer_config={
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 100000,  # Limit size
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
        # Storage optimizations
        "storage_unit": "sequences",  # Store sequences, not timesteps
        "max_seq_len": 20,
    }
)
```

#### 2. Gradient Accumulation
```python
class MemoryEfficientRLModule(TorchRLModule):
    def __init__(self, config):
        super().__init__(config)
        self.accumulation_steps = 4

    def training_step(self, batch):
        # Split batch for gradient accumulation
        mini_batch_size = len(batch) // self.accumulation_steps

        for i in range(self.accumulation_steps):
            start_idx = i * mini_batch_size
            end_idx = (i + 1) * mini_batch_size
            mini_batch = batch[start_idx:end_idx]

            loss = self.compute_loss(mini_batch)
            loss = loss / self.accumulation_steps
            loss.backward()

        # Update weights after accumulation
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### Training Speed Optimization

#### 1. Efficient Data Loading
```python
config.rollouts(
    # Faster data transfer
    num_envs_per_worker=8,
    rollout_fragment_length=200,
    batch_mode="truncate_episodes",

    # Prefetch data
    prefetch_batches=2,

    # Use ray.data for large datasets
    input_="dataset",
    input_config={
        "format": "json",
        "paths": ["s3://bucket/data/"],
        "parallelism": 200,
    }
)
```

#### 2. Compilation and JIT
```python
class CompiledRLModule(TorchRLModule):
    def __init__(self, config):
        super().__init__(config)

        # Compile the model for faster execution
        if torch.__version__ >= "2.0.0":
            self.encoder = torch.compile(self.encoder)
            self.policy_head = torch.compile(self.policy_head)
```

---

## Debugging Techniques

### 1. Enable Debug Mode

```python
# Local mode for easier debugging
ray.init(local_mode=True)

# Or via command line
python train.py --debug
```

### 2. Comprehensive Logging

```python
import logging
from ray.rllib.utils.debug import summarize

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

class DebuggingRLModule(TorchRLModule):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    @override(TorchRLModule)
    def _forward_train(self, batch: dict) -> dict:
        # Log batch information
        self.logger.debug(f"Batch keys: {batch.keys()}")
        self.logger.debug(f"Obs shape: {batch[SampleBatch.OBS].shape}")

        # Log intermediate values
        obs = batch[SampleBatch.OBS]
        features = self.encoder(obs)
        self.logger.debug(f"Features stats: mean={features.mean():.4f}, std={features.std():.4f}")

        # Check for NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            self.logger.error("NaN or Inf detected in features!")
            self.logger.error(f"Features: {features}")

        # Continue with forward pass...
```

### 3. Custom Callbacks for Debugging

```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode

class DebugCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode: Episode, **kwargs):
        # Initialize episode metrics
        episode.user_data["action_counts"] = {}
        episode.user_data["reward_sum"] = 0.0

    def on_episode_step(self, *, worker, base_env, policies, episode: Episode, **kwargs):
        # Track actions
        for agent_id, action in episode.last_action_for().items():
            if action not in episode.user_data["action_counts"]:
                episode.user_data["action_counts"][action] = 0
            episode.user_data["action_counts"][action] += 1

        # Track rewards
        for agent_id, reward in episode.last_reward_for().items():
            episode.user_data["reward_sum"] += reward

    def on_episode_end(self, *, worker, base_env, policies, episode: Episode, **kwargs):
        # Log episode summary
        action_dist = episode.user_data["action_counts"]
        total_reward = episode.user_data["reward_sum"]

        episode.custom_metrics["action_distribution"] = action_dist
        episode.custom_metrics["total_episode_reward"] = total_reward

        # Log if episode performed poorly
        if total_reward < -100:
            print(f"Poor episode detected: reward={total_reward}")
            print(f"Action distribution: {action_dist}")

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Log training metrics
        print(f"Training iteration {result['training_iteration']}")
        print(f"  - Episode reward mean: {result['episode_reward_mean']}")
        print(f"  - Learning rate: {result['info']['learner']['default_policy']['curr_lr']}")

        # Check for training issues
        if result.get('info', {}).get('num_agent_steps_trained', 0) == 0:
            print("WARNING: No agent steps trained in this iteration!")

# Use in configuration
config.callbacks(DebugCallbacks)
```

### 4. Visualization Tools

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class VisualizationCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)

    def on_episode_end(self, *, worker, base_env, policies, episode: Episode, **kwargs):
        self.episode_rewards.append(episode.total_reward)
        self.episode_lengths.append(episode.length)

        # Plot every 100 episodes
        if len(self.episode_rewards) % 100 == 0:
            self._plot_metrics()

    def _plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')

        # Plot lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')

        plt.tight_layout()
        plt.savefig(f'training_metrics_{len(self.episode_rewards)}.png')
        plt.close()
```

### 5. Ray Dashboard Usage

```python
# Start with dashboard
ray.init(include_dashboard=True, dashboard_host="0.0.0.0")

# Access at http://localhost:8265

# Key dashboard features:
# - Jobs: View running algorithms
# - Actors: Monitor worker processes
# - Metrics: Real-time performance metrics
# - Logs: Centralized log viewing
```

---

## Bio-Inspired MARL Implementation

### Complete Example: Ant Colony with Pheromones

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class AntColonyEnvironment(MultiAgentEnv):
    """
    Bio-inspired ant colony environment with pheromone communication.
    """

    def __init__(self, config):
        super().__init__()
        self.grid_size = config.get("grid_size", (20, 20))
        self.n_agents = config.get("n_agents", 10)
        self.n_food_sources = config.get("n_food_sources", 3)
        self.pheromone_decay = config.get("pheromone_decay", 0.95)
        self.pheromone_strength = config.get("pheromone_strength", 0.1)

        # Spaces
        # Observation: [agent_pos_x, agent_pos_y, carrying_food,
        #               pheromone_levels(4 directions), nearest_food_direction(2)]
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.float32
        )
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=pickup/drop
        self.action_space = gym.spaces.Discrete(5)

        self._agent_ids = [f"ant_{i}" for i in range(self.n_agents)]
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Initialize grid
        self.pheromone_grid = np.zeros(self.grid_size, dtype=np.float32)

        # Initialize ant positions
        self.ant_positions = {}
        self.ant_carrying = {}
        for agent_id in self._agent_ids:
            # Start near nest (center)
            center = (self.grid_size[0] // 2, self.grid_size[1] // 2)
            offset = np.random.randint(-2, 3, size=2)
            pos = np.clip(center + offset, 0, np.array(self.grid_size) - 1)
            self.ant_positions[agent_id] = pos
            self.ant_carrying[agent_id] = False

        # Initialize food sources
        self.food_sources = []
        for _ in range(self.n_food_sources):
            pos = np.random.randint(0, self.grid_size[0], size=2)
            self.food_sources.append(pos)

        self.nest_pos = np.array(self.grid_size) // 2
        self.collected_food = 0
        self.steps = 0

        return self._get_observations(), {}

    def _get_observations(self):
        observations = {}

        for agent_id in self._agent_ids:
            pos = self.ant_positions[agent_id]

            # Normalize position
            norm_pos = pos / np.array(self.grid_size)

            # Carrying food flag
            carrying = float(self.ant_carrying[agent_id])

            # Pheromone levels in 4 directions
            pheromone_levels = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = pos + np.array([dx, dy])
                if (0 <= new_pos[0] < self.grid_size[0] and
                    0 <= new_pos[1] < self.grid_size[1]):
                    pheromone_levels.append(
                        self.pheromone_grid[new_pos[0], new_pos[1]]
                    )
                else:
                    pheromone_levels.append(0.0)

            # Direction to nearest food or nest
            if carrying:
                target = self.nest_pos
            else:
                # Find nearest food
                if self.food_sources:
                    distances = [np.linalg.norm(pos - food)
                                for food in self.food_sources]
                    nearest_idx = np.argmin(distances)
                    target = self.food_sources[nearest_idx]
                else:
                    target = pos  # No food left

            direction = target - pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

            # Combine into observation
            obs = np.concatenate([
                norm_pos,
                [carrying],
                pheromone_levels,
                direction
            ]).astype(np.float32)

            observations[agent_id] = obs

        return observations

    def step(self, actions):
        self.steps += 1
        rewards = {}

        # Process ant movements and actions
        for agent_id, action in actions.items():
            pos = self.ant_positions[agent_id]
            old_pos = pos.copy()

            # Movement actions
            if action == 0 and pos[0] > 0:  # Up
                pos[0] -= 1
            elif action == 1 and pos[0] < self.grid_size[0] - 1:  # Down
                pos[0] += 1
            elif action == 2 and pos[1] > 0:  # Left
                pos[1] -= 1
            elif action == 3 and pos[1] < self.grid_size[1] - 1:  # Right
                pos[1] += 1
            elif action == 4:  # Pickup/Drop
                if not self.ant_carrying[agent_id]:
                    # Try to pick up food
                    for i, food_pos in enumerate(self.food_sources):
                        if np.array_equal(pos, food_pos):
                            self.ant_carrying[agent_id] = True
                            self.food_sources.pop(i)
                            rewards[agent_id] = 1.0  # Reward for finding food
                            break
                else:
                    # Try to drop food at nest
                    if np.array_equal(pos, self.nest_pos):
                        self.ant_carrying[agent_id] = False
                        self.collected_food += 1
                        rewards[agent_id] = 10.0  # Big reward for delivering food

            self.ant_positions[agent_id] = pos

            # Deposit pheromones
            if self.ant_carrying[agent_id]:
                # Strong pheromone when carrying food
                self.pheromone_grid[old_pos[0], old_pos[1]] += self.pheromone_strength * 2
            else:
                # Weak pheromone when searching
                self.pheromone_grid[old_pos[0], old_pos[1]] += self.pheromone_strength

            # Default reward
            if agent_id not in rewards:
                rewards[agent_id] = -0.01  # Small penalty for time

        # Decay pheromones
        self.pheromone_grid *= self.pheromone_decay
        self.pheromone_grid = np.clip(self.pheromone_grid, 0, 1)

        # Check termination
        done = self.steps >= 1000 or len(self.food_sources) == 0

        dones = {agent_id: done for agent_id in self._agent_ids}
        dones["__all__"] = done

        truncateds = {agent_id: False for agent_id in self._agent_ids}
        truncateds["__all__"] = False

        return self._get_observations(), rewards, dones, truncateds, {}


class BiologicalNeuralPlasticity(nn.Module):
    """
    Implements biological neural plasticity mechanisms.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Base weights
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)

        # Plasticity coefficients (learnable)
        self.alpha = nn.Parameter(torch.zeros(input_dim, hidden_dim))

        # Hebbian trace
        self.hebbian_trace = torch.zeros(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base activation
        h = torch.matmul(x, self.W)

        # Add plastic component
        if self.training:
            plastic_component = torch.matmul(x, self.alpha * self.hebbian_trace)
            h = h + plastic_component

            # Update Hebbian trace (Oja's rule for stability)
            batch_outer = torch.matmul(x.t(), h) / x.shape[0]
            self.hebbian_trace = 0.9 * self.hebbian_trace + 0.1 * batch_outer

            # Normalize to prevent explosion
            self.hebbian_trace = self.hebbian_trace / (
                self.hebbian_trace.norm() + 1e-8
            )

        return h


class AntColonyRLModule(TorchRLModule):
    """
    Bio-inspired RL module for ant colony optimization.
    """

    def __init__(self, config):
        super().__init__(config)

        model_config = getattr(config, "model_config_dict", {})
        obs_dim = config.observation_space.shape[0]
        act_dim = config.action_space.n
        hidden_dim = model_config.get("hidden_dim", 128)

        # Bio-inspired components
        self.use_plasticity = model_config.get("use_plasticity", True)
        self.use_pheromone_attention = model_config.get("use_pheromone_attention", True)

        # Neural architecture
        if self.use_plasticity:
            self.plastic_layer = BiologicalNeuralPlasticity(obs_dim, hidden_dim)
        else:
            self.plastic_layer = nn.Linear(obs_dim, hidden_dim)

        # Pheromone attention mechanism
        if self.use_pheromone_attention:
            self.pheromone_attention = nn.MultiheadAttention(
                embed_dim=4,  # 4 pheromone directions
                num_heads=2,
                batch_first=True
            )
            self.pheromone_projection = nn.Linear(4, hidden_dim // 4)

        # Main processing layers
        self.hidden_layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Intrinsic curiosity module
        self.curiosity_enabled = model_config.get("use_curiosity", True)
        if self.curiosity_enabled:
            self.forward_model = nn.Sequential(
                nn.Linear(hidden_dim + act_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.inverse_model = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, act_dim)
            )

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        from ray.rllib.models.torch.torch_distributions import TorchCategorical
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        from ray.rllib.models.torch.torch_distributions import TorchCategorical
        return TorchCategorical

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        from ray.rllib.models.torch.torch_distributions import TorchCategorical
        return TorchCategorical

    def _process_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract and process different observation components."""
        # Split observation
        # [agent_pos_x, agent_pos_y, carrying_food, pheromone_levels(4), nearest_food_direction(2)]
        position = obs[:, :2]
        carrying = obs[:, 2:3]
        pheromones = obs[:, 3:7]
        direction = obs[:, 7:9]

        # Process through plastic layer
        features = self.plastic_layer(obs)

        # Apply pheromone attention if enabled
        if self.use_pheromone_attention:
            # Reshape pheromones for attention [batch, seq_len=1, features=4]
            pheromones_reshaped = pheromones.unsqueeze(1)

            # Self-attention on pheromone signals
            attended_pheromones, _ = self.pheromone_attention(
                pheromones_reshaped,
                pheromones_reshaped,
                pheromones_reshaped
            )
            attended_pheromones = attended_pheromones.squeeze(1)

            # Project and add to features
            pheromone_features = self.pheromone_projection(attended_pheromones)
            features[:, :pheromone_features.shape[1]] += pheromone_features

        return features

    @override(TorchRLModule)
    def _forward_inference(self, batch: dict) -> dict:
        obs = batch[SampleBatch.OBS]

        # Process observations with bio-inspired components
        features = self._process_observations(obs)

        # Main processing
        hidden = self.hidden_layers(features)

        # Policy output
        action_logits = self.policy_head(hidden)

        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: dict) -> dict:
        # Same as inference for this module
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch: dict) -> dict:
        obs = batch[SampleBatch.OBS]

        # Process observations
        features = self._process_observations(obs)

        # Main processing
        hidden = self.hidden_layers(features)

        # Outputs
        action_logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)

        outputs = {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: values,
        }

        # Add curiosity bonus if enabled
        if self.curiosity_enabled and SampleBatch.NEXT_OBS in batch:
            next_obs = batch[SampleBatch.NEXT_OBS]
            actions = batch[SampleBatch.ACTIONS]

            # Get features for next obs
            next_features = self._process_observations(next_obs)
            next_hidden = self.hidden_layers(next_features)

            # Forward model: predict next hidden state
            action_one_hot = torch.nn.functional.one_hot(
                actions.long(), num_classes=self.action_space.n
            ).float()
            predicted_next = self.forward_model(
                torch.cat([hidden, action_one_hot], dim=-1)
            )

            # Intrinsic reward: prediction error
            intrinsic_reward = torch.norm(
                next_hidden - predicted_next, dim=-1
            )
            outputs["intrinsic_reward"] = intrinsic_reward

            # Inverse model: predict action from state transition
            predicted_action = self.inverse_model(
                torch.cat([hidden, next_hidden], dim=-1)
            )
            outputs["inverse_model_logits"] = predicted_action

        return outputs
```

### Training Configuration for Bio-Inspired MARL

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Register the environment
register_env("ant_colony", lambda config: AntColonyEnvironment(config))

# Create configuration
bio_inspired_config = (
    PPOConfig()
    .environment(
        env="ant_colony",
        env_config={
            "grid_size": (30, 30),
            "n_agents": 20,
            "n_food_sources": 5,
            "pheromone_decay": 0.98,
            "pheromone_strength": 0.2,
        }
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        policies_to_train=["shared_policy"],
    )
    .rl_module(
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={
                "shared_policy": SingleAgentRLModuleSpec(
                    module_class=AntColonyRLModule,
                    observation_space=gym.spaces.Box(-1, 1, (9,), np.float32),
                    action_space=gym.spaces.Discrete(5),
                    model_config_dict={
                        "hidden_dim": 256,
                        "use_plasticity": True,
                        "use_pheromone_attention": True,
                        "use_curiosity": True,
                    }
                )
            }
        )
    )
    .training(
        lr=1e-4,
        train_batch_size=8192,
        sgd_minibatch_size=256,
        num_sgd_iter=20,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        # Custom loss weights
        model={
            "custom_model_config": {
                "curiosity_weight": 0.1,
                "inverse_model_weight": 0.1,
            }
        }
    )
    .rollouts(
        num_rollout_workers=8,
        num_envs_per_worker=2,
    )
    .resources(
        num_gpus=1,
    )
    .callbacks(BiologicalMetricsCallbacks)
)

# Custom callbacks for bio-inspired metrics
class BiologicalMetricsCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Track biological metrics
        env = base_env.get_sub_environments()[0]

        episode.custom_metrics["food_collected"] = env.collected_food
        episode.custom_metrics["pheromone_coverage"] = (
            np.sum(env.pheromone_grid > 0.1) / env.pheromone_grid.size
        )
        episode.custom_metrics["ant_efficiency"] = (
            env.collected_food / (env.steps + 1) * 1000
        )
```

---

## Advanced Topics

### 1. Custom Loss Functions

```python
class CustomLossRLModule(TorchRLModule):
    """Example of implementing custom losses in RLModule."""

    def compute_loss(
        self,
        fwd_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Override to implement custom loss computation.
        """
        # Get standard losses from parent
        losses = super().compute_loss(fwd_out, batch)

        # Add custom regularization
        if "hidden_features" in fwd_out:
            # Sparsity regularization
            sparsity_loss = torch.mean(torch.abs(fwd_out["hidden_features"]))
            losses["sparsity_loss"] = 0.01 * sparsity_loss

        # Add curiosity loss if available
        if "intrinsic_reward" in fwd_out:
            curiosity_loss = -torch.mean(fwd_out["intrinsic_reward"])
            losses["curiosity_loss"] = 0.1 * curiosity_loss

        # Add inverse model loss
        if "inverse_model_logits" in fwd_out and SampleBatch.ACTIONS in batch:
            inverse_loss = nn.functional.cross_entropy(
                fwd_out["inverse_model_logits"],
                batch[SampleBatch.ACTIONS].long()
            )
            losses["inverse_model_loss"] = 0.1 * inverse_loss

        return losses
```

### 2. Hierarchical Multi-Agent Systems

```python
class HierarchicalMARL:
    """
    Example of hierarchical multi-agent architecture with managers and workers.
    """

    @staticmethod
    def create_hierarchical_config():
        return (
            PPOConfig()
            .multi_agent(
                policies={
                    "manager": PolicySpec(
                        observation_space=gym.spaces.Box(-1, 1, (20,)),
                        action_space=gym.spaces.Discrete(10),  # High-level actions
                    ),
                    "worker": PolicySpec(
                        observation_space=gym.spaces.Box(-1, 1, (15,)),
                        action_space=gym.spaces.Discrete(5),   # Low-level actions
                    ),
                },
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs:
                    "manager" if agent_id.startswith("manager") else "worker",
                policies_to_train=["manager", "worker"],
            )
            .rl_module(
                rl_module_spec=MultiAgentRLModuleSpec(
                    module_specs={
                        "manager": SingleAgentRLModuleSpec(
                            module_class=ManagerRLModule,
                            model_config_dict={
                                "hidden_dim": 256,
                                "num_workers_supervised": 4,
                            }
                        ),
                        "worker": SingleAgentRLModuleSpec(
                            module_class=WorkerRLModule,
                            model_config_dict={
                                "hidden_dim": 128,
                                "manager_instruction_dim": 10,
                            }
                        ),
                    }
                )
            )
        )
```

### 3. Communication Protocols

```python
class CommunicationProtocolRLModule(TorchRLModule):
    """
    Implements learnable communication protocols between agents.
    """

    def __init__(self, config):
        super().__init__(config)

        model_config = getattr(config, "model_config_dict", {})
        self.comm_channels = model_config.get("comm_channels", 4)
        self.message_dim = model_config.get("message_dim", 16)

        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.message_dim * self.comm_channels)
        )

        # Message decoder with attention
        self.message_attention = nn.MultiheadAttention(
            embed_dim=self.message_dim,
            num_heads=4,
            batch_first=True
        )

        self.message_decoder = nn.Linear(
            self.message_dim * self.comm_channels,
            self.hidden_dim
        )

    def communicate(
        self,
        agent_features: torch.Tensor,
        agent_ids: List[str]
    ) -> torch.Tensor:
        """
        Implement communication between agents.
        """
        batch_size = agent_features.shape[0]

        # Generate messages
        messages = self.message_encoder(agent_features)
        messages = messages.view(batch_size, self.comm_channels, self.message_dim)

        # Apply attention to received messages
        attended_messages, attention_weights = self.message_attention(
            messages, messages, messages
        )

        # Decode messages
        attended_flat = attended_messages.view(batch_size, -1)
        comm_features = self.message_decoder(attended_flat)

        # Combine with original features
        enhanced_features = agent_features + comm_features

        return enhanced_features, attention_weights
```

### 4. Curriculum Learning

```python
from ray.rllib.env.env_context import EnvContext

class CurriculumEnvironment(MultiAgentEnv):
    """Environment with curriculum learning support."""

    def __init__(self, config: EnvContext):
        super().__init__()
        self.curriculum_stage = 0
        self.success_threshold = config.get("success_threshold", 0.8)
        self.stages = config.get("curriculum_stages", [
            {"n_agents": 2, "grid_size": 10, "n_goals": 1},
            {"n_agents": 4, "grid_size": 15, "n_goals": 2},
            {"n_agents": 8, "grid_size": 20, "n_goals": 3},
            {"n_agents": 16, "grid_size": 30, "n_goals": 5},
        ])

        self._apply_curriculum_stage()

    def _apply_curriculum_stage(self):
        """Apply current curriculum stage settings."""
        stage = self.stages[min(self.curriculum_stage, len(self.stages) - 1)]
        self.n_agents = stage["n_agents"]
        self.grid_size = stage["grid_size"]
        self.n_goals = stage["n_goals"]

        # Update spaces
        self.observation_space = gym.spaces.Box(
            -1, 1, (self._get_obs_dim(),), np.float32
        )
        self.action_space = gym.spaces.Discrete(5)
        self._agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

    def update_curriculum(self, metrics: dict):
        """Update curriculum based on performance metrics."""
        success_rate = metrics.get("success_rate", 0)

        if success_rate >= self.success_threshold:
            if self.curriculum_stage < len(self.stages) - 1:
                self.curriculum_stage += 1
                self._apply_curriculum_stage()
                print(f"Advanced to curriculum stage {self.curriculum_stage}")

# Curriculum callbacks
class CurriculumCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Calculate success metrics
        success_rate = result.get("custom_metrics", {}).get("success_rate_mean", 0)

        # Update curriculum in all environments
        def update_curriculum(env):
            if hasattr(env, "update_curriculum"):
                env.update_curriculum({"success_rate": success_rate})

        algorithm.workers.foreach_env(update_curriculum)
```

---

## Migration from Legacy Code

### Complete Migration Checklist

1. **Model Migration**
   - [ ] Replace `TorchModelV2` with `TorchRLModule`
   - [ ] Implement three forward methods
   - [ ] Update state handling for recurrent models
   - [ ] Remove `ModelCatalog` registration

2. **Configuration Migration**
   - [ ] Use fluent API instead of dict updates
   - [ ] Replace `model` config with `rl_module` spec
   - [ ] Update multi-agent configuration
   - [ ] Enable new API stack in experimental

3. **Training Loop Migration**
   - [ ] Use `Algorithm.train()` instead of `trainer.train()`
   - [ ] Update metric names in stopping conditions
   - [ ] Migrate custom trainers to Learner API

4. **Callback Migration**
   - [ ] Update callback signatures
   - [ ] Use new episode and batch objects
   - [ ] Migrate metrics to new names

### Migration Script Example

```python
import ast
import re

def migrate_ray_code(old_code: str) -> str:
    """
    Automated migration helper for Ray 2.9.0.
    """
    new_code = old_code

    # Replace ModelV2 imports
    new_code = re.sub(
        r'from ray\.rllib\.models\.torch\.torch_modelv2 import TorchModelV2',
        'from ray.rllib.core.rl_module.torch import TorchRLModule',
        new_code
    )

    # Replace model_config with model_config_dict
    new_code = re.sub(
        r'model_config=([^,\)]+)',
        r'model_config_dict=\1',
        new_code
    )

    # Replace old forward method
    new_code = re.sub(
        r'def forward\(self, input_dict, state, seq_lens\):',
        'def _forward_train(self, batch: dict) -> dict:',
        new_code
    )

    # Update batch access
    new_code = re.sub(
        r'input_dict\["obs"\]',
        'batch[SampleBatch.OBS]',
        new_code
    )

    # Add required imports
    if 'SampleBatch' in new_code and 'from ray.rllib.policy.sample_batch' not in new_code:
        new_code = 'from ray.rllib.policy.sample_batch import SampleBatch\n' + new_code

    return new_code

# Example usage
old_model_code = '''
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        return self.fc(obs), state
'''

new_model_code = migrate_ray_code(old_model_code)
print(new_model_code)
```

---

## Best Practices & Golden Rules

### 1. **Always Use `model_config_dict`**
```python
# This is the #1 source of errors
spec = SingleAgentRLModuleSpec(
    model_config_dict=config  # NOT model_config
)
```

### 2. **Implement All Three Forward Methods**
Even if they're identical, all three must exist:
- `_forward_inference`
- `_forward_exploration`
- `_forward_train`

### 3. **Use SampleBatch Constants**
```python
# Always use constants, never strings
obs = batch[SampleBatch.OBS]  # NOT batch["obs"]
```

### 4. **Handle State Properly**
```python
# Always check for state existence
if "state_in" in batch and batch["state_in"]:
    state = batch["state_in"]
else:
    state = self.get_initial_state()
```

### 5. **Enable New API Stack**
```python
config.experimental(_enable_new_api_stack=True)
```

### 6. **Test in Local Mode First**
```python
ray.init(local_mode=True)  # Easier debugging
```

### 7. **Avoid torch_geometric**
Use custom implementations instead - it causes worker crashes.

### 8. **Use Float32 Consistently**
```python
self.observation_space = gym.spaces.Box(
    low=-1, high=1, shape=(10,), dtype=np.float32
)
```

### 9. **Register Environments Before Use**
```python
register_env("my_env", lambda config: MyEnv(config))
```

### 10. **Monitor Resource Usage**
```python
# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

---

## Conclusion

This handbook provides comprehensive coverage of Ray RLlib 2.9.0 for MARL development. Key takeaways:

1. **API Changes**: The `model_config_dict` parameter is critical
2. **Architecture**: RLModule replaces ModelV2 with cleaner separation
3. **Performance**: Proper GPU utilization and batching are essential
4. **Debugging**: Use local mode and extensive logging
5. **Bio-Inspired**: Ray 2.9.0 supports complex emergent behaviors

For the latest updates and additional resources:
- [Ray Documentation](https://docs.ray.io/en/releases-2.9.0/)
- [RLlib Examples](https://github.com/ray-project/ray/tree/releases/2.9.0/rllib/examples)
- [Community Forum](https://discuss.ray.io/c/rllib/)

Remember: When in doubt, check parameter names and import paths!
