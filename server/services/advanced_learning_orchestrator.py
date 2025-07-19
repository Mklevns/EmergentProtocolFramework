"""
Advanced Learning Orchestrator
Integrates curriculum learning, transfer learning, and meta-learning
"""

import asyncio
import json
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from pathlib import Path

from .curriculum_learning import (
    CurriculumLearningManager, 
    initialize_curriculum_manager, 
    get_curriculum_manager
)
from .transfer_learning import (
    TransferLearningManager, 
    initialize_transfer_manager, 
    get_transfer_manager
)
from .meta_learning import (
    MetaLearningManager, 
    initialize_meta_learning_manager, 
    get_meta_learning_manager
)
from .ray_training_orchestrator import RayTrainingOrchestrator

logger = logging.getLogger(__name__)

class AdvancedLearningMode:
    """Learning mode configuration"""
    CURRICULUM_ONLY = "curriculum_only"
    TRANSFER_ONLY = "transfer_only"  
    META_ONLY = "meta_only"
    CURRICULUM_TRANSFER = "curriculum_transfer"
    CURRICULUM_META = "curriculum_meta"
    TRANSFER_META = "transfer_meta"
    FULL_ADVANCED = "full_advanced"

class AdvancedLearningOrchestrator:
    """Main orchestrator for advanced learning features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.advanced_config = config.get('advanced', {})
        
        # Learning mode configuration
        self.learning_mode = self._determine_learning_mode()
        
        # Initialize managers based on configuration
        self.curriculum_manager = None
        self.transfer_manager = None
        self.meta_manager = None
        
        self._initialize_managers()
        
        # Training state
        self.experiment_id = config.get('experiment_id', 1)
        self.is_training = False
        self.current_phase = "initialization"
        self.phase_history = []
        
        # Integration with existing system
        self.ray_orchestrator = None
        
        # Performance tracking
        self.advanced_metrics = {
            'curriculum_progress': {},
            'transfer_performance': {},
            'meta_learning_efficiency': {},
            'overall_improvement': 0.0
        }
        
        logger.info(f"Advanced Learning Orchestrator initialized with mode: {self.learning_mode}")
        
    def _determine_learning_mode(self) -> str:
        """Determine which advanced learning features are enabled"""
        
        curriculum_enabled = self.advanced_config.get('curriculum_learning', {}).get('enabled', False)
        transfer_enabled = self.advanced_config.get('transfer_learning', {}).get('enabled', False)
        meta_enabled = self.advanced_config.get('meta_learning', {}).get('enabled', False)
        
        if curriculum_enabled and transfer_enabled and meta_enabled:
            return AdvancedLearningMode.FULL_ADVANCED
        elif curriculum_enabled and transfer_enabled:
            return AdvancedLearningMode.CURRICULUM_TRANSFER
        elif curriculum_enabled and meta_enabled:
            return AdvancedLearningMode.CURRICULUM_META
        elif transfer_enabled and meta_enabled:
            return AdvancedLearningMode.TRANSFER_META
        elif curriculum_enabled:
            return AdvancedLearningMode.CURRICULUM_ONLY
        elif transfer_enabled:
            return AdvancedLearningMode.TRANSFER_ONLY
        elif meta_enabled:
            return AdvancedLearningMode.META_ONLY
        else:
            return "standard"
            
    def _initialize_managers(self):
        """Initialize enabled learning managers"""
        
        if self._is_curriculum_enabled():
            self.curriculum_manager = initialize_curriculum_manager(self.config)
            
        if self._is_transfer_enabled():
            self.transfer_manager = initialize_transfer_manager(self.config)
            
        if self._is_meta_enabled():
            self.meta_manager = initialize_meta_learning_manager(self.config)
            
    def _is_curriculum_enabled(self) -> bool:
        """Check if curriculum learning is enabled"""
        return self.learning_mode in [
            AdvancedLearningMode.CURRICULUM_ONLY,
            AdvancedLearningMode.CURRICULUM_TRANSFER,
            AdvancedLearningMode.CURRICULUM_META,
            AdvancedLearningMode.FULL_ADVANCED
        ]
        
    def _is_transfer_enabled(self) -> bool:
        """Check if transfer learning is enabled"""
        return self.learning_mode in [
            AdvancedLearningMode.TRANSFER_ONLY,
            AdvancedLearningMode.CURRICULUM_TRANSFER,
            AdvancedLearningMode.TRANSFER_META,
            AdvancedLearningMode.FULL_ADVANCED
        ]
        
    def _is_meta_enabled(self) -> bool:
        """Check if meta-learning is enabled"""
        return self.learning_mode in [
            AdvancedLearningMode.META_ONLY,
            AdvancedLearningMode.CURRICULUM_META,
            AdvancedLearningMode.TRANSFER_META,
            AdvancedLearningMode.FULL_ADVANCED
        ]
        
    def integrate_with_ray_orchestrator(self, ray_orchestrator: RayTrainingOrchestrator):
        """Integrate with existing Ray training orchestrator"""
        self.ray_orchestrator = ray_orchestrator
        logger.info("Integrated with Ray training orchestrator")
        
    async def execute_advanced_training(self) -> Dict[str, Any]:
        """Execute training with advanced learning features"""
        
        self.is_training = True
        training_start = time.time()
        
        logger.info(f"Starting advanced training with mode: {self.learning_mode}")
        
        try:
            # Phase 1: Pre-training setup and transfer learning
            if self._is_transfer_enabled():
                await self._execute_transfer_phase()
                
            # Phase 2: Curriculum learning execution
            if self._is_curriculum_enabled():
                curriculum_result = await self._execute_curriculum_phase()
                
            # Phase 3: Meta-learning integration
            if self._is_meta_enabled():
                meta_result = await self._execute_meta_learning_phase()
                
            # Phase 4: Combined advanced training
            if self.learning_mode == AdvancedLearningMode.FULL_ADVANCED:
                combined_result = await self._execute_combined_phase()
                
            training_time = time.time() - training_start
            
            # Compile final results
            results = await self._compile_training_results(training_time)
            
            logger.info(f"Advanced training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Advanced training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'learning_mode': self.learning_mode,
                'completed_phases': [phase['name'] for phase in self.phase_history]
            }
            
        finally:
            self.is_training = False
            
    async def _execute_transfer_phase(self) -> Dict[str, Any]:
        """Execute transfer learning phase"""
        
        phase_start = time.time()
        self.current_phase = "transfer_learning"
        
        logger.info("Executing transfer learning phase")
        
        # Get transfer recommendations
        target_config = {
            'architecture': self.config.get('networks', {}),
            'required_components': ['attention', 'memory', 'communication']
        }
        
        recommendations = self.transfer_manager.get_transfer_recommendations(target_config)
        
        transfer_results = {
            'recommendations_found': len(recommendations),
            'transfers_executed': 0,
            'performance_improvements': []
        }
        
        # Execute top recommendations
        for recommendation in recommendations[:3]:  # Top 3 recommendations
            try:
                # Prepare transfer
                transfer_plan = self.transfer_manager.prepare_transfer(
                    recommendation['source_experiment'],
                    target_config,
                    recommendation['recommended_components']
                )
                
                if transfer_plan['success']:
                    # Note: In a real implementation, this would transfer to the actual model
                    # For now, we simulate the transfer process
                    logger.info(f"Simulated transfer from {recommendation['source_experiment']}")
                    transfer_results['transfers_executed'] += 1
                    transfer_results['performance_improvements'].append(
                        recommendation['expected_benefit']['final_performance_boost']
                    )
                    
            except Exception as e:
                logger.warning(f"Transfer failed: {e}")
                
        # Record phase completion
        phase_result = {
            'name': 'transfer_learning',
            'duration': time.time() - phase_start,
            'results': transfer_results
        }
        
        self.phase_history.append(phase_result)
        self.advanced_metrics['transfer_performance'] = transfer_results
        
        return phase_result
        
    async def _execute_curriculum_phase(self) -> Dict[str, Any]:
        """Execute curriculum learning phase"""
        
        phase_start = time.time()
        self.current_phase = "curriculum_learning"
        
        logger.info("Executing curriculum learning phase")
        
        curriculum_results = {
            'stages_completed': 0,
            'total_episodes': 0,
            'performance_progression': [],
            'final_mastery_score': 0.0
        }
        
        # Simulate curriculum progression
        if self.curriculum_manager:
            # Get current curriculum stage
            current_stage = self.curriculum_manager.get_current_stage()
            if current_stage:
                # Simulate training episodes for current stage
                for episode in range(min(100, current_stage.episodes)):
                    # Simulate improving performance over episodes
                    base_performance = 0.3 + (episode / 100) * 0.5
                    noise = np.random.normal(0, 0.1)
                    performance = np.clip(base_performance + noise, 0, 1)
                    
                    # Record episode performance
                    metrics = {
                        'success_rate': performance,
                        'communication_efficiency': min(1.0, performance + 0.1),
                        'coordination_score': performance * 0.9
                    }
                    
                    self.curriculum_manager.record_episode_performance(metrics)
                    curriculum_results['total_episodes'] += 1
                    
                    # Check for stage advancement
                    if episode % 20 == 0:
                        progress = self.curriculum_manager.get_progress_info()
                        curriculum_results['performance_progression'].append({
                            'episode': episode,
                            'stage': progress['current_stage']['name'],
                            'mastery_score': progress['performance']['mastery_score']
                        })
                        
                        # Break if stage advanced
                        if progress['performance']['ready_to_advance']:
                            curriculum_results['stages_completed'] += 1
                            break
                            
            # Get final progress
            final_progress = self.curriculum_manager.get_progress_info()
            curriculum_results['final_mastery_score'] = final_progress['performance']['mastery_score']
            
        # Record phase completion
        phase_result = {
            'name': 'curriculum_learning',
            'duration': time.time() - phase_start,
            'results': curriculum_results
        }
        
        self.phase_history.append(phase_result)
        self.advanced_metrics['curriculum_progress'] = curriculum_results
        
        return phase_result
        
    async def _execute_meta_learning_phase(self) -> Dict[str, Any]:
        """Execute meta-learning phase"""
        
        phase_start = time.time()
        self.current_phase = "meta_learning"
        
        logger.info("Executing meta-learning phase")
        
        meta_results = {
            'tasks_adapted': 0,
            'adaptation_successes': 0,
            'average_adaptation_time': 0.0,
            'learning_efficiency': 0.0
        }
        
        if self.meta_manager:
            # Create and adapt to multiple meta-tasks
            task_configs = [
                {
                    'task_id': 'coordination_task_1',
                    'description': 'Basic agent coordination',
                    'difficulty': 0.4,
                    'support_episodes': 20,
                    'query_episodes': 10
                },
                {
                    'task_id': 'communication_task_1', 
                    'description': 'Advanced communication patterns',
                    'difficulty': 0.7,
                    'support_episodes': 30,
                    'query_episodes': 15
                },
                {
                    'task_id': 'breakthrough_task_1',
                    'description': 'Breakthrough detection optimization',
                    'difficulty': 0.6,
                    'support_episodes': 25,
                    'query_episodes': 12
                }
            ]
            
            adaptation_times = []
            
            for task_config in task_configs:
                task = self.meta_manager.create_meta_task(task_config)
                
                # Simulate training data
                training_data = {
                    'support_set': {
                        'episodes': task_config['support_episodes'],
                        'performance': 0.6 + np.random.uniform(-0.1, 0.2)
                    },
                    'query_set': {
                        'episodes': task_config['query_episodes'], 
                        'performance': 0.7 + np.random.uniform(-0.1, 0.2)
                    }
                }
                
                # Perform adaptation
                adaptation_result = await asyncio.to_thread(
                    self.meta_manager.adapt_to_task, task, training_data
                )
                
                meta_results['tasks_adapted'] += 1
                
                if adaptation_result['success']:
                    meta_results['adaptation_successes'] += 1
                    adaptation_times.append(adaptation_result['adaptation_time'])
                    
            # Calculate averages
            if adaptation_times:
                meta_results['average_adaptation_time'] = np.mean(adaptation_times)
                
            # Get meta-learning insights
            insights = self.meta_manager.get_meta_learning_insights()
            meta_results['learning_efficiency'] = insights['insights'].get('adaptation_success_rate', 0.0)
            
        # Record phase completion
        phase_result = {
            'name': 'meta_learning',
            'duration': time.time() - phase_start,
            'results': meta_results
        }
        
        self.phase_history.append(phase_result)
        self.advanced_metrics['meta_learning_efficiency'] = meta_results
        
        return phase_result
        
    async def _execute_combined_phase(self) -> Dict[str, Any]:
        """Execute combined advanced learning phase"""
        
        phase_start = time.time()
        self.current_phase = "combined_advanced"
        
        logger.info("Executing combined advanced learning phase")
        
        # Combine insights from all learning approaches
        combined_results = {
            'integration_success': True,
            'synergy_effects': {},
            'overall_improvement': 0.0
        }
        
        # Calculate synergy effects
        curriculum_score = self.advanced_metrics['curriculum_progress'].get('final_mastery_score', 0)
        transfer_improvements = self.advanced_metrics['transfer_performance'].get('performance_improvements', [])
        meta_efficiency = self.advanced_metrics['meta_learning_efficiency'].get('learning_efficiency', 0)
        
        # Estimate overall improvement from combination
        base_improvement = curriculum_score * 0.4
        transfer_improvement = np.mean(transfer_improvements) if transfer_improvements else 0
        meta_improvement = meta_efficiency * 0.3
        
        # Synergy bonus for using multiple approaches
        synergy_bonus = 0.1 if len(self.phase_history) >= 3 else 0
        
        overall_improvement = base_improvement + transfer_improvement + meta_improvement + synergy_bonus
        combined_results['overall_improvement'] = min(1.0, overall_improvement)
        
        # Record synergy effects
        combined_results['synergy_effects'] = {
            'curriculum_contribution': base_improvement,
            'transfer_contribution': transfer_improvement,
            'meta_contribution': meta_improvement,
            'synergy_bonus': synergy_bonus
        }
        
        self.advanced_metrics['overall_improvement'] = combined_results['overall_improvement']
        
        # Record phase completion
        phase_result = {
            'name': 'combined_advanced',
            'duration': time.time() - phase_start,
            'results': combined_results
        }
        
        self.phase_history.append(phase_result)
        
        return phase_result
        
    async def _compile_training_results(self, training_time: float) -> Dict[str, Any]:
        """Compile final training results"""
        
        results = {
            'success': True,
            'learning_mode': self.learning_mode,
            'training_time': training_time,
            'phases_completed': len(self.phase_history),
            'phase_history': self.phase_history,
            'advanced_metrics': self.advanced_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        # Add specific results based on enabled features
        if self._is_curriculum_enabled():
            results['curriculum_data'] = self.curriculum_manager.export_curriculum_data()
            
        if self._is_transfer_enabled():
            results['transfer_data'] = self.transfer_manager.export_transfer_data()
            
        if self._is_meta_enabled():
            results['meta_learning_data'] = self.meta_manager.export_meta_learning_data()
            
        return results
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on advanced learning results"""
        
        recommendations = []
        
        # Curriculum learning recommendations
        if self._is_curriculum_enabled():
            curriculum_score = self.advanced_metrics['curriculum_progress'].get('final_mastery_score', 0)
            if curriculum_score < 0.7:
                recommendations.append("Consider extending curriculum stages for better mastery")
            elif curriculum_score > 0.9:
                recommendations.append("Excellent curriculum performance - consider adding advanced stages")
                
        # Transfer learning recommendations
        if self._is_transfer_enabled():
            transfers_executed = self.advanced_metrics['transfer_performance'].get('transfers_executed', 0)
            if transfers_executed == 0:
                recommendations.append("No successful transfers - review model compatibility")
            elif transfers_executed > 0:
                recommendations.append("Transfer learning successful - monitor for positive transfer effects")
                
        # Meta-learning recommendations
        if self._is_meta_enabled():
            adaptation_rate = self.advanced_metrics['meta_learning_efficiency'].get('learning_efficiency', 0)
            if adaptation_rate < 0.5:
                recommendations.append("Low meta-learning efficiency - increase adaptation steps or meta-learning rate")
            elif adaptation_rate > 0.8:
                recommendations.append("High meta-learning efficiency - consider more challenging meta-tasks")
                
        # Overall recommendations
        overall_improvement = self.advanced_metrics.get('overall_improvement', 0)
        if overall_improvement > 0.7:
            recommendations.append("Excellent advanced learning performance - system is well-optimized")
        elif overall_improvement < 0.3:
            recommendations.append("Consider reviewing advanced learning configuration and parameters")
            
        return recommendations
        
    def get_advanced_status(self) -> Dict[str, Any]:
        """Get current advanced learning status"""
        
        status = {
            'learning_mode': self.learning_mode,
            'is_training': self.is_training,
            'current_phase': self.current_phase,
            'phases_completed': len(self.phase_history),
            'enabled_features': {
                'curriculum_learning': self._is_curriculum_enabled(),
                'transfer_learning': self._is_transfer_enabled(),
                'meta_learning': self._is_meta_enabled()
            }
        }
        
        # Add manager-specific status
        if self.curriculum_manager:
            status['curriculum_status'] = self.curriculum_manager.get_progress_info()
            
        if self.transfer_manager:
            status['transfer_status'] = self.transfer_manager.get_transfer_status()
            
        if self.meta_manager:
            status['meta_status'] = self.meta_manager.get_meta_learning_insights()
            
        return status
        
    def output_metrics(self, metrics: Dict[str, Any]):
        """Output metrics for real-time monitoring"""
        enhanced_metrics = {
            'type': 'advanced_learning_metrics',
            'experiment_id': self.experiment_id,
            'timestamp': time.time(),
            'learning_mode': self.learning_mode,
            'current_phase': self.current_phase,
            **metrics,
            **self.advanced_metrics
        }
        
        print(json.dumps(enhanced_metrics), flush=True)

# Global advanced orchestrator instance
_advanced_orchestrator: Optional[AdvancedLearningOrchestrator] = None

def get_advanced_orchestrator() -> AdvancedLearningOrchestrator:
    """Get the global advanced learning orchestrator"""
    global _advanced_orchestrator
    if _advanced_orchestrator is None:
        _advanced_orchestrator = AdvancedLearningOrchestrator({})
    return _advanced_orchestrator

def initialize_advanced_orchestrator(config: Dict[str, Any]) -> AdvancedLearningOrchestrator:
    """Initialize advanced learning orchestrator with configuration"""
    global _advanced_orchestrator
    _advanced_orchestrator = AdvancedLearningOrchestrator(config)
    return _advanced_orchestrator

async def test_advanced_orchestrator():
    """Test advanced learning orchestrator"""
    
    config = {
        'experiment_id': 1,
        'advanced': {
            'curriculum_learning': {
                'enabled': True,
                'stages': [
                    {'name': 'basic_coordination', 'episodes': 100, 'difficulty': 0.3},
                    {'name': 'advanced_coordination', 'episodes': 200, 'difficulty': 0.7}
                ]
            },
            'transfer_learning': {
                'enabled': True,
                'source_experiment': 'test_exp_1'
            },
            'meta_learning': {
                'enabled': True,
                'adaptation_steps': 5,
                'meta_lr': 0.01
            }
        }
    }
    
    orchestrator = initialize_advanced_orchestrator(config)
    
    # Execute advanced training
    results = await orchestrator.execute_advanced_training()
    
    print(f"Advanced training completed: {results['success']}")
    print(f"Learning mode: {results['learning_mode']}")
    print(f"Phases completed: {results['phases_completed']}")
    print(f"Overall improvement: {results['advanced_metrics']['overall_improvement']:.3f}")

if __name__ == "__main__":
    asyncio.run(test_advanced_orchestrator())