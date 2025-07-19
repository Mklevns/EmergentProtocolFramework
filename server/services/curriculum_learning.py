"""
Curriculum Learning Implementation
Progressive difficulty training for enhanced agent learning
"""

import asyncio
import json
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import torch
import yaml

logger = logging.getLogger(__name__)

class CurriculumStage(Enum):
    """Curriculum learning stages"""
    BASIC_COORDINATION = "basic_coordination"
    ADVANCED_COORDINATION = "advanced_coordination" 
    EXPERT_COORDINATION = "expert_coordination"
    ADAPTIVE_MASTERY = "adaptive_mastery"

@dataclass
class StageConfig:
    """Configuration for a curriculum stage"""
    name: str
    episodes: int
    difficulty: float
    environment_params: Dict[str, Any]
    success_threshold: float = 0.8
    min_episodes: int = 50
    max_episodes: int = 1000
    
class CurriculumMetrics:
    """Tracks curriculum learning progress"""
    
    def __init__(self):
        self.stage_performances = {}
        self.transition_history = []
        self.mastery_scores = {}
        self.adaptation_rates = {}
        
    def record_episode(self, stage: str, episode: int, metrics: Dict[str, float]):
        """Record episode performance for curriculum tracking"""
        if stage not in self.stage_performances:
            self.stage_performances[stage] = []
            
        performance_data = {
            'episode': episode,
            'timestamp': time.time(),
            'metrics': metrics,
            'success_rate': metrics.get('success_rate', 0.0),
            'communication_efficiency': metrics.get('communication_efficiency', 0.0),
            'coordination_score': metrics.get('coordination_score', 0.0)
        }
        
        self.stage_performances[stage].append(performance_data)
        
    def calculate_mastery_score(self, stage: str, window_size: int = 50) -> float:
        """Calculate mastery score for a stage"""
        if stage not in self.stage_performances:
            return 0.0
            
        recent_performances = self.stage_performances[stage][-window_size:]
        if len(recent_performances) < 10:  # Need minimum episodes
            return 0.0
            
        # Weighted average of key metrics
        weights = {
            'success_rate': 0.4,
            'communication_efficiency': 0.3,
            'coordination_score': 0.3
        }
        
        total_score = 0.0
        for performance in recent_performances:
            episode_score = 0.0
            for metric, weight in weights.items():
                episode_score += performance['metrics'].get(metric, 0.0) * weight
            total_score += episode_score
            
        mastery_score = total_score / len(recent_performances)
        self.mastery_scores[stage] = mastery_score
        return mastery_score
        
    def should_advance_stage(self, stage: str, config: StageConfig) -> bool:
        """Determine if ready to advance to next stage"""
        mastery_score = self.calculate_mastery_score(stage)
        episodes_completed = len(self.stage_performances.get(stage, []))
        
        # Check mastery threshold
        if mastery_score >= config.success_threshold:
            if episodes_completed >= config.min_episodes:
                return True
                
        # Force advancement if max episodes reached
        if episodes_completed >= config.max_episodes:
            logger.warning(f"Force advancing from stage {stage} after {episodes_completed} episodes")
            return True
            
        return False

class CurriculumEnvironmentAdapter:
    """Adapts environment difficulty based on curriculum stage"""
    
    def __init__(self):
        self.base_config = {}
        self.current_adaptations = {}
        
    def adapt_environment(self, stage: CurriculumStage, difficulty: float) -> Dict[str, Any]:
        """Adapt environment parameters for curriculum stage"""
        
        adaptations = {}
        
        if stage == CurriculumStage.BASIC_COORDINATION:
            adaptations = {
                'communication_range': 3.0,  # Increased range for easier coordination
                'pheromone_strength': 1.2,   # Stronger pheromones
                'noise_level': 0.1,          # Low noise
                'obstacle_density': 0.2,     # Few obstacles
                'task_complexity': 0.3,      # Simple tasks
                'reward_shaping': True,      # More guidance
                'max_episode_steps': 300     # Shorter episodes
            }
            
        elif stage == CurriculumStage.ADVANCED_COORDINATION:
            adaptations = {
                'communication_range': 2.5,
                'pheromone_strength': 1.0,
                'noise_level': 0.2,
                'obstacle_density': 0.4,
                'task_complexity': 0.6,
                'reward_shaping': True,
                'max_episode_steps': 400
            }
            
        elif stage == CurriculumStage.EXPERT_COORDINATION:
            adaptations = {
                'communication_range': 2.0,
                'pheromone_strength': 0.8,
                'noise_level': 0.3,
                'obstacle_density': 0.6,
                'task_complexity': 0.9,
                'reward_shaping': False,  # Minimal guidance
                'max_episode_steps': 500
            }
            
        elif stage == CurriculumStage.ADAPTIVE_MASTERY:
            # Dynamic difficulty based on performance
            adaptations = {
                'communication_range': 1.5 + difficulty * 1.0,
                'pheromone_strength': 0.6 + difficulty * 0.6,
                'noise_level': 0.1 + difficulty * 0.4,
                'obstacle_density': 0.3 + difficulty * 0.5,
                'task_complexity': 0.5 + difficulty * 0.5,
                'reward_shaping': difficulty < 0.7,
                'max_episode_steps': int(400 + difficulty * 200)
            }
            
        # Apply difficulty scaling
        for key, value in adaptations.items():
            if isinstance(value, (int, float)) and key != 'max_episode_steps':
                adaptations[key] = value * (0.5 + 0.5 * difficulty)
                
        self.current_adaptations = adaptations
        return adaptations

class CurriculumLearningManager:
    """Main curriculum learning manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.curriculum_config = config.get('curriculum_learning', {})
        self.is_enabled = self.curriculum_config.get('enabled', False)
        
        # Initialize curriculum stages
        self.stages = self._initialize_stages()
        self.current_stage_index = 0
        self.current_stage = self.stages[0] if self.stages else None
        
        # Initialize components
        self.metrics = CurriculumMetrics()
        self.environment_adapter = CurriculumEnvironmentAdapter()
        
        # State tracking
        self.stage_episode_count = 0
        self.total_episodes = 0
        self.stage_start_time = time.time()
        
        logger.info(f"Curriculum Learning Manager initialized. Enabled: {self.is_enabled}")
        if self.is_enabled:
            logger.info(f"Loaded {len(self.stages)} curriculum stages")
            
    def _initialize_stages(self) -> List[StageConfig]:
        """Initialize curriculum stages from configuration"""
        stages = []
        
        stage_configs = self.curriculum_config.get('stages', [])
        for stage_config in stage_configs:
            stage = StageConfig(
                name=stage_config['name'],
                episodes=stage_config['episodes'],
                difficulty=stage_config['difficulty'],
                environment_params=stage_config.get('environment_params', {}),
                success_threshold=stage_config.get('success_threshold', 0.8),
                min_episodes=stage_config.get('min_episodes', 50),
                max_episodes=stage_config.get('max_episodes', stage_config['episodes'] * 2)
            )
            stages.append(stage)
            
        return stages
    
    def is_curriculum_enabled(self) -> bool:
        """Check if curriculum learning is enabled"""
        return self.is_enabled
        
    def get_current_stage(self) -> Optional[StageConfig]:
        """Get current curriculum stage"""
        return self.current_stage
        
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage"""
        if not self.is_enabled or not self.current_stage:
            return {}
            
        stage_enum = CurriculumStage(self.current_stage.name)
        return self.environment_adapter.adapt_environment(stage_enum, self.current_stage.difficulty)
        
    def record_episode_performance(self, metrics: Dict[str, float]):
        """Record episode performance and check for stage advancement"""
        if not self.is_enabled or not self.current_stage:
            return
            
        self.stage_episode_count += 1
        self.total_episodes += 1
        
        # Record metrics
        self.metrics.record_episode(
            self.current_stage.name,
            self.stage_episode_count,
            metrics
        )
        
        # Check for stage advancement
        if self.metrics.should_advance_stage(self.current_stage.name, self.current_stage):
            self._advance_to_next_stage()
            
    def _advance_to_next_stage(self):
        """Advance to the next curriculum stage"""
        if self.current_stage_index + 1 < len(self.stages):
            # Log current stage completion
            mastery_score = self.metrics.calculate_mastery_score(self.current_stage.name)
            logger.info(f"Completed stage '{self.current_stage.name}' with mastery score: {mastery_score:.3f}")
            
            # Advance to next stage
            self.current_stage_index += 1
            self.current_stage = self.stages[self.current_stage_index]
            self.stage_episode_count = 0
            self.stage_start_time = time.time()
            
            # Record transition
            transition = {
                'from_stage': self.stages[self.current_stage_index - 1].name,
                'to_stage': self.current_stage.name,
                'timestamp': time.time(),
                'episodes_completed': self.total_episodes,
                'mastery_score': mastery_score
            }
            self.metrics.transition_history.append(transition)
            
            logger.info(f"Advanced to stage '{self.current_stage.name}' (Stage {self.current_stage_index + 1}/{len(self.stages)})")
            
        else:
            logger.info("Curriculum learning completed! All stages mastered.")
            
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current curriculum progress information"""
        if not self.is_enabled:
            return {'enabled': False}
            
        progress = {
            'enabled': True,
            'current_stage': {
                'name': self.current_stage.name if self.current_stage else None,
                'index': self.current_stage_index,
                'total_stages': len(self.stages),
                'difficulty': self.current_stage.difficulty if self.current_stage else 0,
                'episodes_completed': self.stage_episode_count,
                'episodes_target': self.current_stage.episodes if self.current_stage else 0
            },
            'overall_progress': {
                'total_episodes': self.total_episodes,
                'stages_completed': self.current_stage_index,
                'training_time': time.time() - self.stage_start_time
            },
            'performance': {}
        }
        
        # Add performance metrics
        if self.current_stage:
            mastery_score = self.metrics.calculate_mastery_score(self.current_stage.name)
            progress['performance'] = {
                'mastery_score': mastery_score,
                'success_threshold': self.current_stage.success_threshold,
                'ready_to_advance': self.metrics.should_advance_stage(self.current_stage.name, self.current_stage)
            }
            
        return progress
        
    def get_stage_history(self) -> List[Dict[str, Any]]:
        """Get history of all completed stages"""
        history = []
        
        for stage_name, performances in self.metrics.stage_performances.items():
            if performances:
                stage_info = {
                    'stage_name': stage_name,
                    'episodes_completed': len(performances),
                    'mastery_score': self.metrics.mastery_scores.get(stage_name, 0.0),
                    'average_performance': np.mean([p['metrics'].get('success_rate', 0) for p in performances]),
                    'start_time': performances[0]['timestamp'] if performances else 0,
                    'end_time': performances[-1]['timestamp'] if performances else 0
                }
                history.append(stage_info)
                
        return history
        
    def export_curriculum_data(self) -> Dict[str, Any]:
        """Export all curriculum learning data"""
        return {
            'config': asdict(self.current_stage) if self.current_stage else {},
            'progress': self.get_progress_info(),
            'stage_history': self.get_stage_history(),
            'transitions': self.metrics.transition_history,
            'mastery_scores': self.metrics.mastery_scores,
            'environment_adaptations': self.environment_adapter.current_adaptations
        }

# Global curriculum manager instance
_curriculum_manager: Optional[CurriculumLearningManager] = None

def get_curriculum_manager() -> CurriculumLearningManager:
    """Get the global curriculum learning manager"""
    global _curriculum_manager
    if _curriculum_manager is None:
        # Initialize with default config
        _curriculum_manager = CurriculumLearningManager({})
    return _curriculum_manager

def initialize_curriculum_manager(config: Dict[str, Any]) -> CurriculumLearningManager:
    """Initialize curriculum learning manager with configuration"""
    global _curriculum_manager
    _curriculum_manager = CurriculumLearningManager(config)
    return _curriculum_manager

async def test_curriculum_system():
    """Test curriculum learning system"""
    config = {
        'curriculum_learning': {
            'enabled': True,
            'stages': [
                {
                    'name': 'basic_coordination',
                    'episodes': 100,
                    'difficulty': 0.3,
                    'success_threshold': 0.7
                },
                {
                    'name': 'advanced_coordination', 
                    'episodes': 200,
                    'difficulty': 0.7,
                    'success_threshold': 0.8
                }
            ]
        }
    }
    
    manager = initialize_curriculum_manager(config)
    
    # Simulate training episodes
    for episode in range(50):
        # Simulate improving performance
        success_rate = min(0.9, 0.3 + episode * 0.01)
        comm_efficiency = min(0.95, 0.4 + episode * 0.01)
        
        metrics = {
            'success_rate': success_rate,
            'communication_efficiency': comm_efficiency,
            'coordination_score': (success_rate + comm_efficiency) / 2
        }
        
        manager.record_episode_performance(metrics)
        
        if episode % 10 == 0:
            progress = manager.get_progress_info()
            print(f"Episode {episode}: Stage '{progress['current_stage']['name']}', Mastery: {progress['performance']['mastery_score']:.3f}")

if __name__ == "__main__":
    asyncio.run(test_curriculum_system())