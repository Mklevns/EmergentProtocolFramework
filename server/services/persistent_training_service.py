#!/usr/bin/env python3
"""
Persistent Training Service with Direct Database Access
Demonstrates robust data persistence for long-running MARL experiments
"""

import sys
import json
import time
import logging
import asyncio
import random
import statistics
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    get_db_manager, 
    AgentData, 
    ExperimentData, 
    MetricData, 
    MemoryVectorData,
    BreakthroughData,
    MessageData
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersistentTrainingService:
    """Training service with robust database persistence"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = get_db_manager()
        
        # Extract configuration
        self.experiment_id = config.get('experiment_id')
        self.experiment_name = config.get('experiment_name', 'Persistent MARL Training')
        self.total_episodes = config.get('total_episodes', 200)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 300)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.batch_size = config.get('batch_size', 32)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.breakthrough_threshold = config.get('breakthrough_threshold', 0.75)
        self.checkpoint_interval = config.get('checkpoint_interval', 10)  # episodes
        
        # Load agents from database or config
        self.agents = self._load_agents(config.get('agents', []))
        
        # Training state
        self.current_episode = 0
        self.current_step = 0
        self.start_time = None
        self.experiment = None
        
        # Persistent metrics and state
        self.episode_rewards = []
        self.communication_efficiency_history = []
        self.breakthrough_events = []
        self.memory_utilization_history = []
        
        logger.info(f"Persistent training service initialized")
        logger.info(f"Experiment: {self.experiment_name}, Episodes: {self.total_episodes}")
        logger.info(f"Agents: {len(self.agents)}, Database persistence enabled")
    
    def _load_agents(self, agent_configs: List[Dict]) -> List[Dict]:
        """Load agents from database or configuration"""
        if self.experiment_id:
            # Try to load agents from database
            db_agents = self.db.get_all_agents()
            if db_agents:
                logger.info(f"Loaded {len(db_agents)} agents from database")
                return [self._agent_data_to_dict(agent) for agent in db_agents]
        
        # Use provided configuration
        logger.info(f"Using {len(agent_configs)} agents from configuration")
        return agent_configs
    
    def _agent_data_to_dict(self, agent: AgentData) -> Dict:
        """Convert AgentData to dictionary format"""
        return {
            'agent_id': agent.agent_id,
            'type': agent.type,
            'position': {
                'x': agent.position_x,
                'y': agent.position_y,
                'z': agent.position_z
            },
            'status': agent.status,
            'coordinator_id': agent.coordinator_id,
            'hidden_dim': agent.hidden_dim,
            'is_active': agent.is_active
        }
    
    def _create_or_update_experiment(self) -> ExperimentData:
        """Create or update experiment in database"""
        if self.experiment_id:
            # Try to load existing experiment
            experiment = self.db.get_experiment(self.experiment_id)
            if experiment:
                logger.info(f"Resuming experiment {self.experiment_id}: {experiment.name}")
                self.db.update_experiment_status(self.experiment_id, "running")
                return experiment
        
        # Create new experiment
        experiment_data = ExperimentData(
            name=self.experiment_name,
            description=f"Persistent MARL training with {len(self.agents)} agents",
            config=self.config,
            status="running"
        )
        
        experiment = self.db.create_experiment(experiment_data)
        if experiment:
            self.experiment_id = experiment.id
            logger.info(f"Created new experiment {self.experiment_id}: {experiment.name}")
            return experiment
        else:
            raise RuntimeError("Failed to create experiment in database")
    
    def _save_metric(self, episode: int, step: int, metric_type: str, value: float, agent_id: str = None):
        """Save training metric to database"""
        if not self.experiment_id:
            return
        
        metric_data = MetricData(
            experiment_id=self.experiment_id,
            episode=episode,
            step=step,
            metric_type=metric_type,
            value=value,
            agent_id=agent_id
        )
        
        self.db.create_metric(metric_data)
    
    def _save_breakthrough(self, agent_id: str, breakthrough_type: str, confidence: float, description: str = None):
        """Save breakthrough event to database"""
        breakthrough_data = BreakthroughData(
            agent_id=agent_id,
            breakthrough_type=breakthrough_type,
            confidence=confidence,
            description=description,
            was_shared=False
        )
        
        breakthrough = self.db.create_breakthrough(breakthrough_data)
        if breakthrough:
            self.breakthrough_events.append({
                'agent_id': agent_id,
                'type': breakthrough_type,
                'confidence': confidence,
                'timestamp': time.time()
            })
            logger.info(f"Breakthrough detected - Agent: {agent_id}, Type: {breakthrough_type}, Confidence: {confidence:.3f}")
    
    def _save_memory_vector(self, vector_id: str, content: Dict, vector_type: str, coordinates: str = None, importance: float = 0.5):
        """Save memory vector to database"""
        memory_data = MemoryVectorData(
            vector_id=vector_id,
            content=content,
            vector_type=vector_type,
            coordinates=coordinates,
            importance=importance
        )
        
        return self.db.create_memory_vector(memory_data)
    
    def _simulate_episode_training(self, episode: int) -> Dict[str, Any]:
        """Simulate training for one episode with realistic MARL dynamics"""
        episode_start_time = time.time()
        episode_reward = 0.0
        step_rewards = []
        communication_events = []
        agent_performances = {}
        
        # Initialize agent states
        for agent in self.agents:
            agent_id = agent['agent_id']
            agent_performances[agent_id] = {
                'reward': 0.0,
                'actions': 0,
                'communications': 0,
                'exploration_rate': max(0.1, 1.0 - episode / self.total_episodes)
            }
        
        # Simulate episode steps
        for step in range(self.max_steps_per_episode):
            self.current_step = step
            
            # Simulate agent interactions and learning
            step_reward = self._simulate_step(step, agent_performances, communication_events)
            step_rewards.append(step_reward)
            episode_reward += step_reward
            
            # Save step-level metrics periodically
            if step % 20 == 0:
                self._save_metric(episode, step, "step_reward", step_reward)
                
                # Calculate communication efficiency
                total_communications = sum(perf['communications'] for perf in agent_performances.values())
                communication_efficiency = min(1.0, total_communications / (len(self.agents) * 10)) if total_communications > 0 else 0.0
                self._save_metric(episode, step, "communication_efficiency", communication_efficiency)
        
        # Episode-level calculations
        episode_duration = time.time() - episode_start_time
        avg_reward = episode_reward / len(self.agents)
        
        # Communication analysis
        total_communications = sum(len(events) for events in communication_events)
        network_efficiency = self._calculate_network_efficiency(communication_events)
        
        # Memory utilization simulation
        memory_utilization = min(1.0, len(self.breakthrough_events) / (episode + 1) * 0.1 + random.uniform(0.2, 0.8))
        self.memory_utilization_history.append(memory_utilization)
        
        # Breakthrough detection
        breakthrough_probability = self._calculate_breakthrough_probability(avg_reward, network_efficiency, episode)
        if breakthrough_probability > self.breakthrough_threshold and random.random() < 0.3:
            breakthrough_agent = random.choice(self.agents)['agent_id']
            self._save_breakthrough(
                breakthrough_agent, 
                "coordination_pattern",
                breakthrough_probability,
                f"Novel coordination strategy discovered at episode {episode}"
            )
        
        # Store memory vectors for significant events  
        recent_avg = statistics.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0.0
        if avg_reward > recent_avg or avg_reward > 0.5:
            vector_id = f"episode_{episode}_memory_vector"
            memory_content = {
                'episode': episode,
                'reward': avg_reward,
                'network_efficiency': network_efficiency,
                'agent_performances': agent_performances,
                'breakthrough_count': len(self.breakthrough_events)
            }
            self._save_memory_vector(vector_id, memory_content, "episode_summary", f"{episode}", avg_reward)
        
        # Compile episode metrics
        episode_metrics = {
            'episode': episode,
            'episode_reward': episode_reward,
            'avg_reward': avg_reward,
            'episode_duration': episode_duration,
            'total_communications': total_communications,
            'network_efficiency': network_efficiency,
            'memory_utilization': memory_utilization,
            'breakthrough_count': len(self.breakthrough_events),
            'steps_completed': len(step_rewards)
        }
        
        # Save episode metrics to database
        self._save_metric(episode, 0, "episode_reward", episode_reward)
        self._save_metric(episode, 0, "avg_reward", avg_reward)
        self._save_metric(episode, 0, "network_efficiency", network_efficiency)
        self._save_metric(episode, 0, "memory_utilization", memory_utilization)
        
        return episode_metrics
    
    def _simulate_step(self, step: int, agent_performances: Dict, communication_events: List) -> float:
        """Simulate a single training step with agent interactions"""
        total_step_reward = 0.0
        
        for agent in self.agents:
            agent_id = agent['agent_id']
            perf = agent_performances[agent_id]
            
            # Simulate agent action and reward
            exploration_bonus = perf['exploration_rate'] * random.uniform(-0.1, 0.1)
            base_reward = random.uniform(0.1, 0.9) + exploration_bonus
            
            # Coordination bonus based on nearby agents
            coordination_bonus = self._calculate_coordination_bonus(agent, self.agents) * 0.2
            
            # Communication reward
            if random.random() < 0.3:  # 30% chance of communication
                perf['communications'] += 1
                communication_reward = random.uniform(0.05, 0.15)
                communication_events.append({
                    'from': agent_id,
                    'to': random.choice([a['agent_id'] for a in self.agents if a['agent_id'] != agent_id]),
                    'step': step,
                    'efficiency': random.uniform(0.5, 1.0)
                })
            else:
                communication_reward = 0.0
            
            step_reward = base_reward + coordination_bonus + communication_reward
            perf['reward'] += step_reward
            perf['actions'] += 1
            total_step_reward += step_reward
        
        return total_step_reward / len(self.agents)
    
    def _calculate_coordination_bonus(self, agent: Dict, all_agents: List[Dict]) -> float:
        """Calculate coordination bonus based on spatial relationships"""
        agent_pos = agent['position']
        coordination_score = 0.0
        
        for other_agent in all_agents:
            if other_agent['agent_id'] == agent['agent_id']:
                continue
            
            other_pos = other_agent['position']
            distance = ((agent_pos['x'] - other_pos['x']) ** 2 + 
                       (agent_pos['y'] - other_pos['y']) ** 2 + 
                       (agent_pos['z'] - other_pos['z']) ** 2) ** 0.5
            
            # Closer agents provide better coordination
            if distance < 2.0:
                coordination_score += 1.0 / (1.0 + distance)
        
        return min(coordination_score / len(all_agents), 1.0)
    
    def _calculate_network_efficiency(self, communication_events: List) -> float:
        """Calculate communication network efficiency"""
        if not communication_events:
            return 0.0
        
        total_efficiency = sum(event.get('efficiency', 0.5) for events in communication_events for event in events if isinstance(events, list))
        return min(total_efficiency / len(communication_events) if communication_events else 0.0, 1.0)
    
    def _calculate_breakthrough_probability(self, reward: float, efficiency: float, episode: int) -> float:
        """Calculate probability of breakthrough based on performance metrics"""
        base_probability = reward * 0.4 + efficiency * 0.3
        experience_factor = min(episode / 50.0, 1.0) * 0.3
        return min(base_probability + experience_factor, 1.0)
    
    def _checkpoint_training(self, episode: int):
        """Save training checkpoint to database"""
        if episode % self.checkpoint_interval == 0:
            checkpoint_data = {
                'episode': episode,
                'total_episodes': self.total_episodes,
                'episode_rewards': self.episode_rewards[-50:],  # Last 50 episodes
                'communication_efficiency': statistics.mean(self.communication_efficiency_history[-10:]) if self.communication_efficiency_history else 0.0,
                'breakthrough_count': len(self.breakthrough_events),
                'memory_utilization': statistics.mean(self.memory_utilization_history[-10:]) if self.memory_utilization_history else 0.0,
                'timestamp': time.time()
            }
            
            if self.experiment_id:
                self.db.update_experiment_metrics(self.experiment_id, checkpoint_data)
                logger.info(f"Training checkpoint saved at episode {episode}")
    
    def output_metrics(self, metrics: Dict[str, Any]):
        """Output metrics in JSON format for real-time monitoring"""
        metrics_json = {
            'type': 'persistent_training_metrics',
            'experiment_id': self.experiment_id,
            'timestamp': time.time(),
            **metrics
        }
        print(json.dumps(metrics_json), flush=True)
    
    async def run_training(self):
        """Main training loop with database persistence"""
        try:
            # Initialize experiment
            self.experiment = self._create_or_update_experiment()
            self.start_time = time.time()
            
            logger.info(f"Starting persistent training for {self.total_episodes} episodes")
            logger.info(f"Database persistence enabled - Experiment ID: {self.experiment_id}")
            
            # Training loop
            for episode in range(self.total_episodes):
                self.current_episode = episode
                
                # Run episode
                episode_metrics = self._simulate_episode_training(episode)
                self.episode_rewards.append(episode_metrics['avg_reward'])
                self.communication_efficiency_history.append(episode_metrics['network_efficiency'])
                
                # Output real-time metrics
                self.output_metrics(episode_metrics)
                
                # Checkpoint training periodically
                self._checkpoint_training(episode)
                
                # Log progress
                if episode % 25 == 0 or episode == self.total_episodes - 1:
                    self._log_training_progress(episode, episode_metrics)
                
                # Small delay to prevent system overload
                await asyncio.sleep(0.02)
            
            # Complete training
            await self._complete_training()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.experiment_id:
                self.db.update_experiment_status(self.experiment_id, "failed")
            raise
    
    def _log_training_progress(self, episode: int, metrics: Dict[str, Any]):
        """Log training progress"""
        avg_reward_last_10 = statistics.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else metrics['avg_reward']
        
        logger.info(
            f"Episode {episode:3d}/{self.total_episodes} - "
            f"Reward: {metrics['avg_reward']:.3f} "
            f"(Avg-10: {avg_reward_last_10:.3f}) - "
            f"Network Eff: {metrics['network_efficiency']:.3f} - "
            f"Breakthroughs: {len(self.breakthrough_events)} - "
            f"Memory: {metrics['memory_utilization']:.3f}"
        )
    
    async def _complete_training(self):
        """Complete training and save final results"""
        training_duration = time.time() - self.start_time
        
        # Calculate final metrics
        final_metrics = {
            'training_duration': training_duration,
            'total_episodes_completed': len(self.episode_rewards),
            'final_avg_reward': statistics.mean(self.episode_rewards[-20:]) if len(self.episode_rewards) >= 20 else statistics.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_communication_efficiency': statistics.mean(self.communication_efficiency_history) if self.communication_efficiency_history else 0.0,
            'total_breakthroughs': len(self.breakthrough_events),
            'avg_memory_utilization': statistics.mean(self.memory_utilization_history) if self.memory_utilization_history else 0.0,
            'convergence_rate': self._calculate_convergence_rate(),
            'completion_status': 'success'
        }
        
        # Update experiment in database
        if self.experiment_id:
            self.db.update_experiment_metrics(self.experiment_id, final_metrics)
            self.db.update_experiment_status(self.experiment_id, "completed")
        
        # Output final results
        final_result = {
            'success': True,
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'final_metrics': final_metrics,
            'database_persistence': True
        }
        
        print(json.dumps(final_result), flush=True)
        logger.info("Persistent training completed successfully")
        logger.info(f"Final metrics saved to database - Experiment ID: {self.experiment_id}")
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate training convergence rate"""
        if len(self.episode_rewards) < 20:
            return 0.0
        
        early_rewards = self.episode_rewards[:len(self.episode_rewards)//3]
        late_rewards = self.episode_rewards[-len(self.episode_rewards)//3:]
        
        early_avg = statistics.mean(early_rewards) if early_rewards else 0.0
        late_avg = statistics.mean(late_rewards) if late_rewards else 0.0
        
        return max(0.0, (late_avg - early_avg) / max(early_avg, 0.1))

async def main():
    """Main entry point for persistent training service"""
    try:
        # Read configuration from stdin
        config_input = sys.stdin.read().strip()
        
        if not config_input:
            raise ValueError("No configuration provided via stdin")
        
        config = json.loads(config_input)
        
        # Initialize and run persistent training service
        training_service = PersistentTrainingService(config)
        await training_service.run_training()
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'database_persistence': True
        }
        print(json.dumps(error_result), flush=True)
        logger.error(f"Persistent training service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())