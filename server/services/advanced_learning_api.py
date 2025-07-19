"""
Advanced Learning API
RESTful API endpoints for curriculum learning, transfer learning, and meta-learning
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
import sys
import os

# Add the server directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_learning_orchestrator import (
    AdvancedLearningOrchestrator,
    initialize_advanced_orchestrator,
    get_advanced_orchestrator
)
from curriculum_learning import get_curriculum_manager, initialize_curriculum_manager
from transfer_learning import get_transfer_manager, initialize_transfer_manager  
from meta_learning import get_meta_learning_manager, initialize_meta_learning_manager

logger = logging.getLogger(__name__)

class AdvancedLearningAPI:
    """API handler for advanced learning features"""
    
    def __init__(self):
        self.orchestrator: Optional[AdvancedLearningOrchestrator] = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize advanced learning system"""
        try:
            self.orchestrator = initialize_advanced_orchestrator(config)
            self.is_initialized = True
            
            return {
                'success': True,
                'learning_mode': self.orchestrator.learning_mode,
                'enabled_features': {
                    'curriculum_learning': self.orchestrator._is_curriculum_enabled(),
                    'transfer_learning': self.orchestrator._is_transfer_enabled(),
                    'meta_learning': self.orchestrator._is_meta_enabled()
                }
            }
        except Exception as e:
            logger.error(f"Failed to initialize advanced learning: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_status(self) -> Dict[str, Any]:
        """Get current advanced learning status"""
        if not self.is_initialized:
            return {
                'initialized': False,
                'message': 'Advanced learning not initialized'
            }
            
        return {
            'initialized': True,
            **self.orchestrator.get_advanced_status()
        }
        
    def start_advanced_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Start advanced training session"""
        if not self.is_initialized:
            return {'success': False, 'error': 'System not initialized'}
            
        try:
            # Update configuration if provided
            if training_config:
                # Reinitialize with new config
                self.orchestrator = initialize_advanced_orchestrator(training_config)
                
            # Start training in background
            asyncio.create_task(self._run_advanced_training())
            
            return {
                'success': True,
                'message': 'Advanced training started',
                'learning_mode': self.orchestrator.learning_mode
            }
            
        except Exception as e:
            logger.error(f"Failed to start advanced training: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _run_advanced_training(self):
        """Run advanced training in background"""
        try:
            results = await self.orchestrator.execute_advanced_training()
            # Output results for monitoring
            self.orchestrator.output_metrics({
                'training_completed': True,
                'results': results
            })
        except Exception as e:
            logger.error(f"Advanced training failed: {e}")
            self.orchestrator.output_metrics({
                'training_failed': True,
                'error': str(e)
            })
            
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get curriculum learning progress"""
        if not self.orchestrator or not self.orchestrator._is_curriculum_enabled():
            return {'enabled': False}
            
        curriculum_manager = get_curriculum_manager()
        return curriculum_manager.get_progress_info()
        
    def get_transfer_recommendations(self, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get transfer learning recommendations"""
        if not self.orchestrator or not self.orchestrator._is_transfer_enabled():
            return {'enabled': False, 'recommendations': []}
            
        try:
            transfer_manager = get_transfer_manager()
            recommendations = transfer_manager.get_transfer_recommendations(target_config)
            
            return {
                'enabled': True,
                'recommendations': recommendations,
                'available_models': len(transfer_manager.available_models)
            }
        except Exception as e:
            logger.error(f"Failed to get transfer recommendations: {e}")
            return {'enabled': True, 'error': str(e)}
            
    def execute_transfer(self, transfer_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transfer learning"""
        if not self.orchestrator or not self.orchestrator._is_transfer_enabled():
            return {'success': False, 'error': 'Transfer learning not enabled'}
            
        try:
            transfer_manager = get_transfer_manager()
            
            source_experiment = transfer_request.get('source_experiment')
            target_config = transfer_request.get('target_config', {})
            components = transfer_request.get('components', ['attention', 'memory'])
            
            # Prepare transfer
            transfer_plan = transfer_manager.prepare_transfer(
                source_experiment, target_config, components
            )
            
            return transfer_plan
            
        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get meta-learning insights"""
        if not self.orchestrator or not self.orchestrator._is_meta_enabled():
            return {'enabled': False}
            
        try:
            meta_manager = get_meta_learning_manager()
            return {
                'enabled': True,
                **meta_manager.get_meta_learning_insights()
            }
        except Exception as e:
            logger.error(f"Failed to get meta-learning insights: {e}")
            return {'enabled': True, 'error': str(e)}
            
    def create_meta_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new meta-learning task"""
        if not self.orchestrator or not self.orchestrator._is_meta_enabled():
            return {'success': False, 'error': 'Meta-learning not enabled'}
            
        try:
            meta_manager = get_meta_learning_manager()
            task = meta_manager.create_meta_task(task_config)
            
            return {
                'success': True,
                'task_id': task.task_id,
                'task_config': {
                    'description': task.description,
                    'difficulty': task.difficulty,
                    'support_episodes': task.support_episodes,
                    'query_episodes': task.query_episodes
                }
            }
        except Exception as e:
            logger.error(f"Failed to create meta-task: {e}")
            return {'success': False, 'error': str(e)}
            
    def save_model_for_transfer(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save model state for future transfer learning"""
        if not self.orchestrator or not self.orchestrator._is_transfer_enabled():
            return {'success': False, 'error': 'Transfer learning not enabled'}
            
        try:
            transfer_manager = get_transfer_manager()
            experiment_id = model_data.get('experiment_id', f"exp_{int(time.time())}")
            
            storage_path = transfer_manager.save_model_state(experiment_id, model_data)
            
            return {
                'success': True,
                'experiment_id': experiment_id,
                'storage_path': storage_path,
                'message': 'Model saved for transfer learning'
            }
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_learning_configuration(self) -> Dict[str, Any]:
        """Get current learning configuration"""
        if not self.is_initialized:
            return {'initialized': False}
            
        config_data = {
            'initialized': True,
            'learning_mode': self.orchestrator.learning_mode,
            'configuration': {}
        }
        
        # Add curriculum configuration
        if self.orchestrator._is_curriculum_enabled():
            curriculum_manager = get_curriculum_manager()
            config_data['configuration']['curriculum'] = curriculum_manager.export_curriculum_data()
            
        # Add transfer configuration  
        if self.orchestrator._is_transfer_enabled():
            transfer_manager = get_transfer_manager()
            config_data['configuration']['transfer'] = transfer_manager.export_transfer_data()
            
        # Add meta-learning configuration
        if self.orchestrator._is_meta_enabled():
            meta_manager = get_meta_learning_manager()
            config_data['configuration']['meta_learning'] = meta_manager.export_meta_learning_data()
            
        return config_data
        
    def update_curriculum_stage(self, episode_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update curriculum learning with episode metrics"""
        if not self.orchestrator or not self.orchestrator._is_curriculum_enabled():
            return {'success': False, 'error': 'Curriculum learning not enabled'}
            
        try:
            curriculum_manager = get_curriculum_manager()
            curriculum_manager.record_episode_performance(episode_metrics)
            
            # Get updated progress
            progress = curriculum_manager.get_progress_info()
            
            return {
                'success': True,
                'progress': progress,
                'stage_advanced': progress['performance'].get('ready_to_advance', False)
            }
        except Exception as e:
            logger.error(f"Failed to update curriculum stage: {e}")
            return {'success': False, 'error': str(e)}

# Global API instance
_advanced_api = AdvancedLearningAPI()

def handle_advanced_learning_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle advanced learning API requests"""
    
    try:
        if endpoint == 'initialize':
            return _advanced_api.initialize(data)
            
        elif endpoint == 'status':
            return _advanced_api.get_status()
            
        elif endpoint == 'start_training':
            return _advanced_api.start_advanced_training(data)
            
        elif endpoint == 'curriculum_progress':
            return _advanced_api.get_curriculum_progress()
            
        elif endpoint == 'transfer_recommendations':
            return _advanced_api.get_transfer_recommendations(data)
            
        elif endpoint == 'execute_transfer':
            return _advanced_api.execute_transfer(data)
            
        elif endpoint == 'meta_insights':
            return _advanced_api.get_meta_learning_insights()
            
        elif endpoint == 'create_meta_task':
            return _advanced_api.create_meta_task(data)
            
        elif endpoint == 'save_model':
            return _advanced_api.save_model_for_transfer(data)
            
        elif endpoint == 'configuration':
            return _advanced_api.get_learning_configuration()
            
        elif endpoint == 'update_curriculum':
            return _advanced_api.update_curriculum_stage(data)
            
        else:
            return {
                'success': False,
                'error': f'Unknown endpoint: {endpoint}',
                'available_endpoints': [
                    'initialize', 'status', 'start_training', 'curriculum_progress',
                    'transfer_recommendations', 'execute_transfer', 'meta_insights',
                    'create_meta_task', 'save_model', 'configuration', 'update_curriculum'
                ]
            }
            
    except Exception as e:
        logger.error(f"API request failed for {endpoint}: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Test the API
    test_config = {
        'advanced': {
            'curriculum_learning': {
                'enabled': True,
                'stages': [
                    {'name': 'basic', 'episodes': 100, 'difficulty': 0.3},
                    {'name': 'advanced', 'episodes': 200, 'difficulty': 0.7}
                ]
            },
            'transfer_learning': {'enabled': True},
            'meta_learning': {'enabled': True}
        }
    }
    
    # Initialize
    result = handle_advanced_learning_request('initialize', test_config)
    print(f"Initialization: {result['success']}")
    
    # Get status
    status = handle_advanced_learning_request('status', {})
    print(f"Status: {status.get('learning_mode', 'N/A')}")