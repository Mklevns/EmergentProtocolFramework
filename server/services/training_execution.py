#!/usr/bin/env python3
"""
Training Execution Script
Entry point for training the bio-inspired MARL system
"""

import json
import sys
import time
import logging
import asyncio
import random
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedTrainingOrchestrator:
    """Simplified training orchestrator for bio-inspired MARL"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_id = config.get('experiment_id', 1)
        self.experiment_name = config.get('experiment_name', 'Training Session')
        self.total_episodes = config.get('total_episodes', 100)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 200)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.breakthrough_threshold = config.get('breakthrough_threshold', 0.7)
        self.agents = config.get('agents', [])
        
        # Training state
        self.current_episode = 0
        self.current_step = 0
        self.start_time = None
        self.episode_metrics = []
        
        # Bio-inspired metrics
        self.pheromone_strength = 0.5
        self.neural_plasticity = 0.8
        self.swarm_coordination = 0.6
        self.emergent_patterns = 0
        
        logger.info(f"Training orchestrator initialized for experiment: {self.experiment_name}")
        logger.info(f"Agents: {len(self.agents)}, Episodes: {self.total_episodes}")
    
    def output_metrics(self, metrics: Dict[str, Any]):
        """Output metrics in JSON format for real-time monitoring"""
        metrics_json = {
            'type': 'training_metrics',
            'experiment_id': self.experiment_id,
            'timestamp': time.time(),
            **metrics
        }
        print(json.dumps(metrics_json), flush=True)
    
    async def train(self):
        """Main training loop"""
        self.start_time = time.time()
        
        logger.info(f"Starting training for {self.total_episodes} episodes")
        
        try:
            for episode in range(self.total_episodes):
                self.current_episode = episode
                
                # Run episode
                episode_metrics = await self._run_episode(episode)
                self.episode_metrics.append(episode_metrics)
                
                # Output real-time metrics
                self.output_metrics(episode_metrics)
                
                # Log progress every 10 episodes
                if episode % 10 == 0:
                    self._log_progress(episode, episode_metrics)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
            
            # Final metrics
            final_metrics = self._calculate_final_metrics()
            
            # Output final results
            final_result = {
                'success': True,
                'experiment_id': self.experiment_id,
                'total_episodes': self.total_episodes,
                'final_metrics': final_metrics,
                'training_time': time.time() - self.start_time
            }
            
            print(json.dumps(final_result), flush=True)
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            error_result = {
                'success': False,
                'error': str(e),
                'experiment_id': self.experiment_id
            }
            print(json.dumps(error_result), flush=True)
            raise
    
    async def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode"""
        episode_start_time = time.time()
        episode_reward = 0.0
        episode_breakthroughs = 0
        episode_communications = 0
        
        # Simulate episode steps
        for step in range(self.max_steps_per_episode):
            self.current_step = step
            
            # Simulate agent interactions
            step_metrics = self._simulate_step(episode, step)
            
            episode_reward += step_metrics['reward']
            episode_breakthroughs += step_metrics['breakthroughs']
            episode_communications += step_metrics['communications']
            
            # Update bio-inspired metrics
            self._update_bio_metrics(step_metrics)
            
            # Small delay to simulate computation
            await asyncio.sleep(0.001)
        
        episode_duration = time.time() - episode_start_time
        
        return {
            'episode': episode,
            'reward': episode_reward,
            'avg_reward': episode_reward / self.max_steps_per_episode,
            'breakthroughs': episode_breakthroughs,
            'communications': episode_communications,
            'duration': episode_duration,
            'pheromone_strength': self.pheromone_strength,
            'neural_plasticity': self.neural_plasticity,
            'swarm_coordination': self.swarm_coordination,
            'emergent_patterns': self.emergent_patterns,
            'communication_efficiency': min(episode_communications / 100.0, 1.0),
            'memory_utilization': random.uniform(0.4, 0.9),
            'breakthrough_frequency': episode_breakthroughs / self.max_steps_per_episode,
            'coordination_success': self.swarm_coordination
        }
    
    def _simulate_step(self, episode: int, step: int) -> Dict[str, Any]:
        """Simulate a single training step"""
        
        # Simulate agent behaviors with bio-inspired patterns
        num_agents = len(self.agents)
        
        # Pheromone-based communication simulation
        communication_prob = self.pheromone_strength * 0.2
        communications = sum(1 for _ in range(num_agents) if random.random() < communication_prob)
        
        # Neural plasticity affects learning and breakthroughs
        breakthrough_prob = self.neural_plasticity * 0.01  # 1% base rate
        breakthroughs = sum(1 for _ in range(num_agents) if random.random() < breakthrough_prob)
        
        # Swarm coordination affects reward
        base_reward = random.gauss(0, 1)
        coordination_bonus = self.swarm_coordination * 2.0
        reward = base_reward + coordination_bonus
        
        # Emergent patterns detection
        if communications > 5 and breakthroughs > 0:
            self.emergent_patterns += 1
        
        return {
            'reward': reward,
            'breakthroughs': breakthroughs,
            'communications': communications,
            'coordination_score': self.swarm_coordination
        }
    
    def _update_bio_metrics(self, step_metrics: Dict[str, Any]):
        """Update bio-inspired metrics based on step performance"""
        
        # Pheromone strength adapts based on communication success
        if step_metrics['communications'] > 0:
            self.pheromone_strength = min(1.0, self.pheromone_strength + 0.001)
        else:
            self.pheromone_strength = max(0.1, self.pheromone_strength - 0.0005)
        
        # Neural plasticity increases with breakthroughs
        if step_metrics['breakthroughs'] > 0:
            self.neural_plasticity = min(1.0, self.neural_plasticity + 0.002)
        else:
            self.neural_plasticity = max(0.3, self.neural_plasticity - 0.0001)
        
        # Swarm coordination based on overall performance
        coordination_factor = (step_metrics['reward'] + step_metrics['communications'] * 0.1) / 10.0
        self.swarm_coordination = max(0.2, min(1.0, 
            self.swarm_coordination * 0.99 + coordination_factor * 0.01
        ))
    
    def _log_progress(self, episode: int, episode_metrics: Dict[str, Any]):
        """Log training progress"""
        elapsed_time = time.time() - self.start_time
        
        logger.info(
            f"Episode {episode}/{self.total_episodes} - "
            f"Reward: {episode_metrics['avg_reward']:.3f} - "
            f"Breakthroughs: {episode_metrics['breakthroughs']} - "
            f"Communications: {episode_metrics['communications']} - "
            f"Pheromone: {self.pheromone_strength:.3f} - "
            f"Plasticity: {self.neural_plasticity:.3f} - "
            f"Coordination: {self.swarm_coordination:.3f} - "
            f"Elapsed: {elapsed_time:.1f}s"
        )
    
    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final training metrics"""
        
        if not self.episode_metrics:
            return {}
        
        # Calculate averages
        avg_reward = sum(m['avg_reward'] for m in self.episode_metrics) / len(self.episode_metrics)
        total_breakthroughs = sum(m['breakthroughs'] for m in self.episode_metrics)
        total_communications = sum(m['communications'] for m in self.episode_metrics)
        avg_communication_efficiency = sum(m['communication_efficiency'] for m in self.episode_metrics) / len(self.episode_metrics)
        avg_memory_utilization = sum(m['memory_utilization'] for m in self.episode_metrics) / len(self.episode_metrics)
        
        # Calculate learning curves
        first_half = self.episode_metrics[:len(self.episode_metrics)//2]
        second_half = self.episode_metrics[len(self.episode_metrics)//2:]
        first_half_reward = sum(m['avg_reward'] for m in first_half) / len(first_half) if first_half else 0
        second_half_reward = sum(m['avg_reward'] for m in second_half) / len(second_half) if second_half else 0
        learning_improvement = second_half_reward - first_half_reward
        
        return {
            'avg_reward': avg_reward,
            'total_breakthroughs': total_breakthroughs,
            'total_communications': total_communications,
            'final_pheromone_strength': self.pheromone_strength,
            'final_neural_plasticity': self.neural_plasticity,
            'final_swarm_coordination': self.swarm_coordination,
            'emergent_patterns': self.emergent_patterns,
            'communication_efficiency': avg_communication_efficiency,
            'memory_utilization': avg_memory_utilization,
            'learning_improvement': learning_improvement,
            'training_time': time.time() - self.start_time
        }

async def main():
    """Main entry point"""
    try:
        # Read configuration from stdin
        config_data = sys.stdin.read()
        
        if not config_data.strip():
            raise ValueError("No configuration data provided")
        
        config = json.loads(config_data)
        
        # Create and run training orchestrator
        orchestrator = SimplifiedTrainingOrchestrator(config)
        await orchestrator.train()
        
    except Exception as e:
        logger.error(f"Training execution failed: {e}")
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())