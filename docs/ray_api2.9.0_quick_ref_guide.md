# Ray RLlib 2.9.0 API Quick Reference Card

## 🚨 Critical API Changes for MARL Projects

### 1. SingleAgentRLModuleSpec Configuration Parameter

**The Issue**: Model configuration parameter name changed from `model_config` to `model_config_dict`.

❌ **Wrong** (causes AttributeError):
```python
spec = SingleAgentRLModuleSpec(
    module_class=NextGenBioInspiredRLModule,
    observation_space=obs_space,
    action_space=act_space,
    model_config=model_config  # Wrong parameter name!
)
```

✅ **Correct**:
```python
spec = SingleAgentRLModuleSpec(
    module_class=NextGenBioInspiredRLModule,
    observation_space=obs_space,
    action_space=act_space,
    model_config_dict=model_config  # Correct parameter name
)
```

**Inside the Module**:
```python
def __init__(self, config: SingleAgentRLModuleSpec):
    super().__init__(config)
    
    # Defensive approach for compatibility
    model_config = getattr(config, "model_config_dict", {})
    if not model_config:
        model_config = getattr(config, "model_config", {})
```

---

### 2. RLModule API vs Legacy ModelV2

**The Issue**: Ray 2.9.0 pushes the new RLModule API, deprecating ModelV2.

❌ **Legacy ModelV2** (deprecated):
```python
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MyModel(TorchModelV2, nn.Module):
    def forward(self, input_dict, state, seq_lens):
        # Old API
        pass
```

✅ **New RLModule API**:
```python
from ray.rllib.core.rl_module.torch import TorchRLModule

class MyRLModule(TorchRLModule):
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # New API
        pass
```

---

### 3. Multi-Agent Configuration

**The Issue**: Multi-agent setup requires proper RLModule specs, not policy configs.

❌ **Old Way**:
```python
config.multi_agent(
    policies={
        "policy_1": (None, obs_space, act_space, {"model": {...}}),
    }
)
```

✅ **New Way**:
```python
config.rl_module(
    rl_module_spec=MultiAgentRLModuleSpec(
        module_specs={
            "shared_policy": SingleAgentRLModuleSpec(
                module_class=MyRLModule,
                observation_space=obs_space,
                action_space=act_space,
                model_config_dict={...}
            )
        }
    )
)
```

---

### 4. Batch Keys and Sample Batch

**The Issue**: Batch key names are standardized in SampleBatch.

✅ **Always use SampleBatch constants**:
```python
from ray.rllib.policy.sample_batch import SampleBatch

# Correct
obs = batch[SampleBatch.OBS]
actions = batch[SampleBatch.ACTIONS]
rewards = batch[SampleBatch.REWARDS]
dones = batch[SampleBatch.DONES]
next_obs = batch[SampleBatch.NEXT_OBS]

# For RLModule outputs
outputs = {
    SampleBatch.ACTION_DIST_INPUTS: action_logits,
    SampleBatch.VF_PREDS: value_predictions,
}
```

---

### 5. Forward Methods in RLModule

**The Issue**: RLModule has specific forward methods for different phases.

✅ **Three Required Methods**:
```python
class MyRLModule(TorchRLModule):
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """For inference/deployment - no value function needed"""
        return {"action_dist_inputs": logits}
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """For exploration during rollouts"""
        return {"action_dist_inputs": logits}
    
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """For training - must include value function"""
        return {
            "action_dist_inputs": logits,
            "vf_preds": values
        }
```

---

### 6. Algorithm Configuration (Fluent API)

**The Issue**: Configuration uses method chaining, not dictionary updates.

❌ **Old Dictionary Style**:
```python
config = {"env": "CartPole-v1", "num_workers": 4}
config["lr"] = 0.001
```

✅ **New Fluent API**:
```python
config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .rollouts(num_rollout_workers=4)
    .training(lr=0.001)
    .resources(num_gpus=1)
)
```

---

### 7. Experimental Features

**The Issue**: New API stack requires explicit enabling.

✅ **Enable New Features**:
```python
config.experimental(
    _enable_new_api_stack=True,  # For RLModule
    _disable_preprocessor_api=False,
)

# Or in the module itself
class MyRLModule(TorchRLModule):
    # These are now default in 2.9.0
    framework = "torch"  
    uses_new_env_runners = True
    uses_new_training_stack = True
```

---

### 8. State Handling for Recurrent Models

**The Issue**: State handling is different in RLModule.

✅ **Proper State Management**:
```python
class RecurrentRLModule(TorchRLModule):
    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        """Return initial hidden states"""
        return {
            "h": torch.zeros(1, self.hidden_size),
            "c": torch.zeros(1, self.hidden_size),
        }
    
    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Access input states
        if "state_in" in batch and batch["state_in"]:
            h = batch["state_in"]["h"]
            c = batch["state_in"]["c"]
        else:
            h, c = self.get_initial_state().values()
        
        # ... process ...
        
        # Return output states
        return {
            SampleBatch.ACTION_DIST_INPUTS: logits,
            SampleBatch.VF_PREDS: values,
            "state_out": {"h": new_h, "c": new_c}
        }
```

---

### 9. Environment Registration

**The Issue**: Tune registry is used for environments.

✅ **Register Custom Environments**:
```python
from ray.tune.registry import register_env

# Register once globally
register_env("MyEnv-v0", lambda config: MyEnvironment(config))

# Use in config
config.environment(env="MyEnv-v0", env_config={...})
```

---

### 10. Common Import Changes

**Updated Imports for Ray 2.9.0**:

```python
# RLModule imports
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule

# Algorithm configs
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig

# Utils
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

# Deprecated imports to avoid
# ❌ from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# ❌ from ray.rllib.models import ModelCatalog
```

---

## 🔧 Quick Debugging Tips

### AttributeError: 'SingleAgentRLModuleSpec' object has no attribute 'model_config'
**Fix**: Use `model_config_dict` instead of `model_config`

### KeyError in batch access
**Fix**: Use `SampleBatch.KEY_NAME` constants

### "RLModule not found" errors
**Fix**: Ensure `_enable_new_api_stack=True` in experimental config

### Multi-agent policy errors
**Fix**: Use `MultiAgentRLModuleSpec` with proper module specs

### CUDA/Worker crashes
**Fix**: Check for incompatible C++ extensions (like torch_geometric)

---

## 📚 Resources

- [Ray 2.9.0 Release Notes](https://docs.ray.io/en/releases-2.9.0/rllib/index.html)
- [RLModule Migration Guide](https://docs.ray.io/en/releases-2.9.0/rllib/rlmodule.html)
- [New API Stack Documentation](https://docs.ray.io/en/releases-2.9.0/rllib/rllib-new-api-stack.html)

---

## 💡 Golden Rules

1. **Always use `model_config_dict`** when creating RLModule specs
2. **Use SampleBatch constants** for batch keys
3. **Implement all three forward methods** in RLModule
4. **Enable new API stack** in experimental settings
5. **Use fluent API** for algorithm configuration
6. **Register environments** before using them
7. **Handle states properly** in recurrent models
8. **Test in local mode first** when debugging

Remember: Ray 2.9.0 is transitioning to the new RLModule API. When in doubt, check if you're mixing old ModelV2 patterns with new RLModule code!
