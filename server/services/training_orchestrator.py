"""
Training Orchestrator
Manages the training process and coordinates all components
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor
import signal

from .marl_framework import MARLFramework, initialize_framework
from .shared_memory import get_shared_memory
from .agent_network import get_network_manager
from .communication_protocol import get_communication_protocol
from .breakthrough_detector import get_breakthrough_detector
from .visualization_data import get_visualization_service
from .ray_integration import get_integration_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration for training experiments"""

    def __init__(self, config_dict: Dict[str, Any]):
        self.experiment_name = config_dict.get('experiment_name', 'default_experiment')
        self.total_episodes = config_dict.get('total_episodes', 1000)
        self.max_steps_per_episode = config_dict.get('max_steps_per_episode', 500)
        self.learning_rate = config_dict.get('learning_rate', 0.001)
        self.batch_size = config_dict.get('batch_size', 32)
        self.hidden_dim = config_dict.get('hidden_dim', 256)
        self.num_coordinators = config_dict.get('num_coordinators', 3)
        self.grid_size = tuple(config_dict.get('grid_size', [4, 3, 3]))
        self.breakthrough_threshold = config_dict.get('breakthrough_threshold', 0.7)
        self.memory_capacity = config_dict.get('memory_capacity', 1000)
        self.communication_range = config_dict.get('communication_range', 2.0)
        self.save_interval = config_dict.get('save_interval', 100)
        self.log_interval = config_dict.get('log_interval', 10)
        self.visualization_enabled = config_dict.get('visualization_enabled', True)
        self.checkpoint_dir = config_dict.get('checkpoint_dir', './checkpoints')
        self.log_dir = config_dict.get('log_dir', './logs')

        # Environment-specific settings
        self.env_config = config_dict.get('environment', {})
        self.reward_config = config_dict.get('rewards', {})
        self.agent_config = config_dict.get('agents', {})

class TrainingOrchestrator:
    """Main training orchestrator for the MARL system"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.experiment_id = None
        self.is_training = False
        self.should_stop = False

        # Initialize components
        self.framework = initialize_framework()
        self.ray_integration = get_integration_manager()
        self.shared_memory = get_shared_memory()
        self.network_manager = get_network_manager()
        self.communication_protocol = get_communication_protocol()
        self.breakthrough_detector = get_breakthrough_detector()
        self.visualization_service = get_visualization_service()

        # Training state
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None
        self.episode_rewards = []
        self.episode_metrics = []

        # Optimization
        self.optimizer = None
        self.loss_function = nn.MSELoss()

        # Monitoring
        self.metrics_history = []
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)

        # Callbacks
        self.episode_callbacks = []
        self.step_callbacks = []
        self.breakthrough_callbacks = []

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"TrainingOrchestrator initialized for experiment: {config.experiment_name}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True

    def initialize_training(self, experiment_id: int):
        """Initialize training components"""

        self.experiment_id = experiment_id

        # Initialize agent grid
        self.framework.initialize_agent_grid()

        # Build network topology
        self.network_manager.build_spatial_network(self.framework.agents)
        self.network_manager.build_hierarchical_network(self.framework.agents)

        # Register agents with communication protocol
        for agent_id in self.framework.agents:
            self.communication_protocol.register_agent(agent_id)

        # Start components
        self.communication_protocol.start_protocol()
        self.breakthrough_detector.start_monitoring()
        self.visualization_service.start_updates()

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.framework.pheromone_attention.parameters()) + 
            list(self.framework.neural_plasticity.parameters()),
            lr=self.config.learning_rate
        )

        # Set up callbacks
        self.breakthrough_detector.add_breakthrough_callback(self._on_breakthrough)

        logger.info(f"Training initialized for experiment {experiment_id}")

    def _on_breakthrough(self, breakthrough_event):
        """Handle breakthrough events"""

        logger.info(f"Breakthrough detected: {breakthrough_event.agent_id} - {breakthrough_event.breakthrough_type.value}")

        # Trigger breakthrough callbacks
        for callback in self.breakthrough_callbacks:
            try:
                callback(breakthrough_event)
            except Exception as e:
                logger.error(f"Error in breakthrough callback: {e}")

    async def train(self):
        """Main training loop"""

        if not self.experiment_id:
            raise ValueError("Must initialize training before starting")

        self.is_training = True
        self.start_time = time.time()

        logger.info(f"Starting training for {self.config.total_episodes} episodes")

        try:
            for episode in range(self.config.total_episodes):
                if self.should_stop:
                    logger.info("Training stopped by signal")
                    break

                self.current_episode = episode

                # Run episode
                episode_metrics = await self._run_episode()

                # Update episode metrics
                self.episode_metrics.append(episode_metrics)

                # Log progress
                if episode % self.config.log_interval == 0:
                    self._log_progress(episode, episode_metrics)

                # Save checkpoint
                if episode % self.config.save_interval == 0:
                    self._save_checkpoint(episode)

                # Trigger episode callbacks
                for callback in self.episode_callbacks:
                    try:
                        callback(episode, episode_metrics)
                    except Exception as e:
                        logger.error(f"Error in episode callback: {e}")

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        finally:
            self.is_training = False
            self._cleanup_training()

    async def _run_episode(self) -> Dict[str, Any]:
        """Run a single training episode"""

        episode_start_time = time.time()
        episode_reward = 0.0
        episode_breakthroughs = 0
        episode_communications = 0

        # Reset environment
        self._reset_episode()

        # Episode loop
        for step in range(self.config.max_steps_per_episode):
            if self.should_stop:
                break

            self.current_step = step
            self.total_steps += 1

            # Run framework step
            self.framework.step()

            # Simulate agent actions and communications
            step_metrics = await self._run_step()

            episode_reward += step_metrics.get('reward', 0.0)
            episode_breakthroughs += step_metrics.get('breakthroughs', 0)
            episode_communications += step_metrics.get('communications', 0)

            # Trigger step callbacks
            for callback in self.step_callbacks:
                try:
                    callback(step, step_metrics)
                except Exception as e:
                    logger.error(f"Error in step callback: {e}")

            # Check termination conditions
            if self._should_terminate_episode(step_metrics):
                break

        episode_duration = time.time() - episode_start_time

        return {
            'episode': self.current_episode,
            'reward': episode_reward,
            'steps': step + 1,
            'duration': episode_duration,
            'breakthroughs': episode_breakthroughs,
            'communications': episode_communications,
            'avg_reward_per_step': episode_reward / (step + 1),
            'breakthroughs_per_step': episode_breakthroughs / (step + 1),
            'communications_per_step': episode_communications / (step + 1)
        }

    async def _run_step(self) -> Dict[str, Any]:
        """Run a single training step"""

        step_reward = 0.0
        step_breakthroughs = 0
        step_communications = 0

        # Simulate agent behaviors
        for agent_id, agent in self.framework.agents.items():
            # Generate behavior vector (simplified simulation)
            behavior_vector = torch.randn(self.config.hidden_dim)

            # Check for breakthrough
            breakthrough = self.breakthrough_detector.detect_breakthrough(
                agent_id=agent_id,
                behavior_vector=behavior_vector,
                coordinates=(agent.position.x, agent.position.y, agent.position.z)
            )

            if breakthrough:
                step_breakthroughs += 1
                step_reward += 10.0  # Reward for breakthrough

            # Simulate communication
            if np.random.random() < 0.1:  # 10% chance of communication
                neighbors = self.network_manager.get_neighbors(agent_id)
                if neighbors:
                    target = np.random.choice(neighbors)

                    # Send message
                    message_id = self.communication_protocol.send_message(
                        source_agent=agent_id,
                        target_agent=target,
                        message_type=self.communication_protocol.MessageType.COORDINATION,
                        content={'coordination_signal': True}
                    )

                    if message_id:
                        step_communications += 1
                        step_reward += 1.0  # Reward for communication

        # Update neural networks
        if self.total_steps % 10 == 0:  # Update every 10 steps
            await self._update_networks()

        return {
            'reward': step_reward,
            'breakthroughs': step_breakthroughs,
            'communications': step_communications,
            'memory_usage': self.shared_memory.get_memory_statistics()['usage_percentage'],
            'network_efficiency': self.network_manager.analyze_network_properties().communication_efficiency
        }

    async def _update_networks(self):
        """Update neural networks"""

        # Get training data from shared memory
        breakthrough_vectors = self.shared_memory.search_vectors_by_type(
            self.shared_memory.VectorType.BREAKTHROUGH
        )

        if len(breakthrough_vectors) < self.config.batch_size:
            return

        # Sample batch
        batch_vectors = np.random.choice(breakthrough_vectors, self.config.batch_size, replace=False)

        batch_data = []
        for vector_id in batch_vectors:
            vector = self.shared_memory.retrieve_vector(vector_id)
            if vector is not None:
                batch_data.append(vector)

        if len(batch_data) < self.config.batch_size:
            return

        # Convert to tensor
        batch_tensor = torch.stack(batch_data)

        # Forward pass through networks
        attended_output, attention_weights = self.framework.pheromone_attention(
            batch_tensor, batch_tensor, batch_tensor,
            torch.zeros(self.config.batch_size, batch_tensor.size(1), 3),  # Dummy positions
            torch.ones(self.config.batch_size, batch_tensor.size(1), batch_tensor.size(1))  # Dummy mask
        )

        # Update memory
        memory_output = self.framework.neural_plasticity(
            attended_output,
            torch.randn(self.config.batch_size, self.config.hidden_dim)
        )

        # Calculate loss (simplified)
        loss = self.loss_function(memory_output, batch_tensor.mean(dim=1))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.debug(f"Networks updated, loss: {loss.item():.4f}")

    def _reset_episode(self):
        """Reset environment for new episode"""

        # Reset agent statuses
        for agent in self.framework.agents.values():
            agent.status = "idle"

        # Clear old memory vectors (optional)
        if self.current_episode % 50 == 0:  # Clear every 50 episodes
            self.shared_memory.optimize_memory_layout()

    def _should_terminate_episode(self, step_metrics: Dict[str, Any]) -> bool:
        """Check if episode should terminate early"""

        # Terminate if no activity for too long
        if (step_metrics.get('breakthroughs', 0) == 0 and 
            step_metrics.get('communications', 0) == 0 and
            self.current_step > 100):
            return True

        return False

    def _log_progress(self, episode: int, episode_metrics: Dict[str, Any]):
        """Log training progress"""

        elapsed_time = time.time() - self.start_time

        logger.info(
            f"Episode {episode}/{self.config.total_episodes} - "
            f"Reward: {episode_metrics['reward']:.2f} - "
            f"Breakthroughs: {episode_metrics['breakthroughs']} - "
            f"Communications: {episode_metrics['communications']} - "
            f"Duration: {episode_metrics['duration']:.2f}s - "
            f"Elapsed: {elapsed_time:.1f}s"
        )

        # Update metrics history
        self.metrics_history.append({
            'episode': episode,
            'timestamp': time.time(),
            'metrics': episode_metrics
        })

    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""

        checkpoint_data = {
            'episode': episode,
            'total_steps': self.total_steps,
            'config': self.config.__dict__,
            'pheromone_attention_state': self.framework.pheromone_attention.state_dict(),
            'neural_plasticity_state': self.framework.neural_plasticity.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'episode_metrics': self.episode_metrics
        }

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data, 
            f"checkpoint_episode_{episode}.pt"
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _cleanup_training(self):
        """Clean up training resources"""

        # Stop components
        self.communication_protocol.stop_protocol()
        self.breakthrough_detector.stop_monitoring()
        self.visualization_service.stop_updates()

        # Save final results
        if self.experiment_id:
            self._save_final_results()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Training cleanup completed")

    def _save_final_results(self):
        """Save final training results"""

        results = {
            'experiment_id': self.experiment_id,
            'config': self.config.__dict__,
            'total_episodes': self.current_episode,
            'total_steps': self.total_steps,
            'training_duration': time.time() - self.start_time,
            'episode_metrics': self.episode_metrics,
            'metrics_history': self.metrics_history,
            'final_breakthrough_stats': self.breakthrough_detector.export_breakthrough_data(),
            'final_communication_stats': self.communication_protocol.export_communication_data(),
            'final_memory_stats': self.shared_memory.export_memory_dump()
        }

        results_path = os.path.join(self.config.log_dir, f"results_{self.experiment_id}.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Final results saved: {results_path}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""

        return {
            'is_training': self.is_training,
            'experiment_id': self.experiment_id,
            'current_episode': self.current_episode,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'recent_metrics': self.episode_metrics[-10:] if self.episode_metrics else [],
            'should_stop': self.should_stop
        }

    def stop_training(self):
        """Stop training gracefully"""

        self.should_stop = True
        logger.info("Training stop requested")

    def add_episode_callback(self, callback: callable):
        """Add callback for episode completion"""
        self.episode_callbacks.append(callback)

    def add_step_callback(self, callback: callable):
        """Add callback for step completion"""
        self.step_callbacks.append(callback)

    def add_breakthrough_callback(self, callback: callable):
        """Add callback for breakthrough events"""
        self.breakthrough_callbacks.append(callback)

    async def start_training(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new training experiment"""

        if self.is_training:
            return {"error": "Training already in progress"}

        try:
            self.current_experiment = experiment_config
            self.is_training = True
            self.training_metrics = []

            # Initialize Ray training
            if self.ray_integration.initialize_training():
                logger.info("Ray training initialized successfully")
            else:
                logger.warning("Ray training failed to initialize, using framework simulation")

            # Initialize framework with experiment config
            if 'agents' in experiment_config:
                agent_config = experiment_config['agents']
                # Apply agent configuration

            # Start training loop
            asyncio.create_task(self._training_loop())

            logger.info(f"Started training experiment: {experiment_config.get('name', 'Unnamed')}")

            return {
                "status": "started",
                "experiment": self.current_experiment,
                "message": "Training started successfully"
            }

        except Exception as e:
            self.is_training = False
            logger.error(f"Error starting training: {e}")
            return {"error": str(e)}

    async def _training_loop(self):
        """Main training loop with Ray integration"""

        episode = 0
        max_episodes = self.current_experiment.get('training', {}).get('total_episodes', 100)

        while self.is_training and episode < max_episodes:
            try:
                # Execute Ray training step if available
                if self.ray_integration.algorithm:
                    ray_result = self.ray_integration.train_step()
                    if "error" not in ray_result:
                        logger.info(f"Ray training step {episode}: {ray_result}")
                else:
                    # Fallback to framework simulation
                    await asyncio.sleep(0.1)  # Simulate computation time

                # Execute framework step for bio-inspired components
                self.framework.step()

                # Collect metrics every 10 episodes
                if episode % 10 == 0:
                    framework_metrics = self.framework.get_training_metrics()

                    # Combine Ray and framework metrics
                    if self.ray_integration.algorithm:
                        ray_result = self.ray_integration.train_step()
                        if "error" not in ray_result:
                            framework_metrics.update(ray_result)

                    framework_metrics['episode'] = episode
                    self.training_metrics.append(framework_metrics)

                    logger.info(f"Episode {episode}: {framework_metrics}")

                episode += 1

                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                break

        self.is_training = False
        logger.info("Training loop completed")

    async def stop_training(self) -> Dict[str, Any]:
        """Stop the current training experiment"""

        if not self.is_training:
            return {"error": "No training in progress"}

        self.is_training = False

        # Stop Ray training
        self.ray_integration.stop_training()

        logger.info("Training stopped")

        return {
            "status": "stopped",
            "message": "Training stopped successfully",
            "final_metrics": self.training_metrics[-1] if self.training_metrics else None
        }

class CheckpointManager:
    """Manages training checkpoints"""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, data: Dict[str, Any], filename: str) -> str:
        """Save checkpoint data"""

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(data, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data"""

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path)
        return None

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""

        if not os.path.exists(self.checkpoint_dir):
            return []

        return [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]

async def main():
    """Main entry point for training orchestrator"""

    parser = argparse.ArgumentParser(description='Bio-Inspired MARL Training Orchestrator')
    parser.add_argument('--experiment-id', type=int, required=True, help='Experiment ID')
    parser.add_argument('--config', type=str, required=True, help='Configuration JSON string')
    parser.add_argument('--config-file', type=str, help='Configuration YAML file')

    args = parser.parse_args()

    # Load configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = json.loads(args.config)

    # Create training configuration
    training_config = TrainingConfig(config_dict)

    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(training_config)

    try:
        # Initialize training
        orchestrator.initialize_training(args.experiment_id)

        # Start training
        await orchestrator.train()

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))