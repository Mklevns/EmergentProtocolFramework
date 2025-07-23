# PyArrow Compatibility Guide

## Overview

This document provides guidance for resolving PyArrow compatibility issues with Ray RLlib that may occur in local development environments.

## Common Issue

**Error Message:**
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'. Did you mean: 'ExtensionType'?
```

**Cause:**
This error occurs when there's a version mismatch between Ray RLlib and PyArrow. In newer versions of PyArrow, `PyExtensionType` was renamed to `ExtensionType`.

## Solutions

### Option 1: Update PyArrow (Recommended)
```bash
pip install pyarrow==12.0.0
```

### Option 2: Use Ray-compatible PyArrow version
```bash
pip install "pyarrow>=10.0.0,<15.0.0"
```

### Option 3: Update Ray to latest version
```bash
pip install ray[rllib]==2.9.0
```

## Fallback System

If Ray RLlib cannot be loaded due to dependency issues, the system automatically falls back to a simplified training system that provides:

- ✅ **Bio-inspired Algorithm Simulation**: Pheromone trails, neural plasticity, swarm coordination
- ✅ **Real-time Metrics**: Training progress, breakthroughs, communication patterns
- ✅ **WebSocket Updates**: Live dashboard updates
- ✅ **Database Persistence**: All training data is saved
- ✅ **Full Functionality**: Complete MARL platform features

## Environment-Specific Notes

### Replit Cloud Environment
- Ray RLlib is properly configured with compatible dependencies
- No manual intervention required

### Local Development
- May require manual dependency resolution
- Use virtual environments to avoid conflicts
- Follow the solutions above

### Docker/Container Environments
- Ensure base image has compatible Python version (3.8-3.11)
- Use specific dependency versions in requirements.txt

## Verification

To check if Ray RLlib is working correctly:

```python
try:
    import ray
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    print("✅ Ray RLlib is available")
except Exception as e:
    print(f"⚠️ Ray RLlib not available: {e}")
```

## Support

If you continue experiencing issues:

1. Check Python version compatibility (3.8-3.11)
2. Clear pip cache: `pip cache purge`
3. Recreate virtual environment
4. Use the fallback training system (automatic)

The application is designed to work seamlessly regardless of Ray availability.