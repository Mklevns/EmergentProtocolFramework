#!/usr/bin/env python3
"""
Ray API Integration
Provides API endpoints for Ray RLlib training integration
"""

import json
import sys
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

import sys
import os

# Add server directory to Python path for absolute imports
server_dir = os.path.dirname(os.path.dirname(__file__))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

from services.ray_training_orchestrator import (
    RayTrainingOrchestrator, 
    create_ray_training_orchestrator,
    run_ray_training
)
from services.ray_full_integration import RayTrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RayTrainingAPI:
    """API interface for Ray RLlib training"""
    
    def __init__(self):
        self.current_orchestrator: Optional[RayTrainingOrchestrator] = None
        self.training_configs = {}
        logger.info("Ray Training API initialized")
    
    async def start_ray_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start Ray RLlib training with given configuration"""
        
        try:
            # Validate configuration
            validated_config = self._validate_training_config(config)
            
            # Create orchestrator
            self.current_orchestrator = create_ray_training_orchestrator(validated_config)
            
            # Store configuration
            experiment_id = validated_config.get('experiment_id', 1)
            self.training_configs[experiment_id] = validated_config
            
            logger.info(f"Starting Ray training for experiment {experiment_id}")
            
            # Start training
            result = await self.current_orchestrator.train()
            
            return {
                'success': True,
                'message': 'Ray training completed successfully',
                'experiment_id': experiment_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Ray training failed: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'message': f'Ray training failed: {str(e)}',
                'error': str(e)
            }
        
        finally:
            # Cleanup
            if self.current_orchestrator:
                self.current_orchestrator.cleanup()
                self.current_orchestrator = None
    
    def stop_ray_training(self) -> Dict[str, Any]:
        """Stop current Ray training"""
        
        try:
            if self.current_orchestrator:
                self.current_orchestrator.stop_training()
                return {
                    'success': True,
                    'message': 'Ray training stop requested'
                }
            else:
                return {
                    'success': False,
                    'message': 'No Ray training session active'
                }
                
        except Exception as e:
            logger.error(f"Failed to stop Ray training: {e}")
            return {
                'success': False,
                'message': f'Failed to stop training: {str(e)}',
                'error': str(e)
            }
    
    def get_ray_training_status(self) -> Dict[str, Any]:
        """Get current Ray training status"""
        
        try:
            if self.current_orchestrator:
                status = self.current_orchestrator.get_training_status()
                return {
                    'success': True,
                    'status': status,
                    'training_active': True
                }
            else:
                return {
                    'success': True,
                    'status': {
                        'is_training': False,
                        'current_iteration': 0,
                        'experiment_id': None,
                        'training_method': 'none',
                        'start_time': None,
                        'elapsed_time': 0,
                        'recent_metrics': []
                    },
                    'training_active': False
                }
                
        except Exception as e:
            logger.error(f"Failed to get Ray training status: {e}")
            return {
                'success': False,
                'message': f'Failed to get status: {str(e)}',
                'error': str(e)
            }
    
    def get_ray_checkpoints(self, experiment_id: Optional[int] = None) -> Dict[str, Any]:
        """Get Ray training checkpoints information"""
        
        try:
            if self.current_orchestrator:
                checkpoint_info = self.current_orchestrator.get_checkpoint_info()
                return {
                    'success': True,
                    'checkpoint_info': checkpoint_info
                }
            else:
                return {
                    'success': True,
                    'checkpoint_info': {
                        'checkpoint_dir': None,
                        'checkpoints': [],
                        'latest_checkpoint': None
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get Ray checkpoints: {e}")
            return {
                'success': False,
                'message': f'Failed to get checkpoints: {str(e)}',
                'error': str(e)
            }
    
    def create_ray_config_template(self, experiment_name: str = "bio_inspired_marl") -> Dict[str, Any]:
        """Create a Ray training configuration template"""
        
        try:
            template_config = {
                'experiment_name': experiment_name,
                'experiment_id': 1,
                'use_ray': True,
                
                # Agent configuration
                'agents': [f'agent_{i}' for i in range(30)],
                'grid_size': [4, 3, 3],
                'num_coordinators': 3,
                
                # Training parameters
                'total_episodes': 1000,
                'max_steps_per_episode': 500,
                'learning_rate': 3e-4,
                'batch_size': 128,
                'train_batch_size': 4000,
                'num_sgd_iter': 10,
                'gamma': 0.99,
                'lambda_': 0.95,
                
                # Ray-specific parameters
                'num_rollout_workers': 4,
                'num_envs_per_worker': 1,
                'rollout_fragment_length': 200,
                
                # Bio-inspired parameters
                'hidden_dim': 256,
                'num_attention_heads': 8,
                'pheromone_decay': 0.95,
                'neural_plasticity_rate': 0.1,
                'communication_range': 2.0,
                'breakthrough_threshold': 0.7,
                
                # Monitoring
                'checkpoint_frequency': 10,
                'evaluation_interval': 5,
                'evaluation_duration': 10,
            }
            
            return {
                'success': True,
                'config_template': template_config,
                'description': 'Ray RLlib training configuration template'
            }
            
        except Exception as e:
            logger.error(f"Failed to create Ray config template: {e}")
            return {
                'success': False,
                'message': f'Failed to create template: {str(e)}',
                'error': str(e)
            }
    
    def _validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance training configuration"""
        
        # Set defaults
        validated_config = {
            'experiment_id': config.get('experiment_id', 1),
            'experiment_name': config.get('experiment_name', 'Bio-Inspired MARL Training'),
            'use_ray': True,  # Force Ray usage
            
            # Agent configuration
            'agents': config.get('agents', [f'agent_{i}' for i in range(30)]),
            'grid_size': config.get('grid_size', [4, 3, 3]),
            'num_coordinators': config.get('num_coordinators', 3),
            
            # Training parameters
            'total_episodes': max(config.get('total_episodes', 1000), 10),
            'max_steps_per_episode': max(config.get('max_steps_per_episode', 500), 50),
            'learning_rate': max(config.get('learning_rate', 3e-4), 1e-6),
            'batch_size': max(config.get('batch_size', 128), 16),
            'train_batch_size': max(config.get('train_batch_size', 4000), 1000),
            'num_sgd_iter': max(config.get('num_sgd_iter', 10), 1),
            'gamma': max(min(config.get('gamma', 0.99), 1.0), 0.0),
            'lambda_': max(min(config.get('lambda_', 0.95), 1.0), 0.0),
            
            # Ray-specific parameters
            'num_rollout_workers': max(config.get('num_rollout_workers', 4), 1),
            'num_envs_per_worker': max(config.get('num_envs_per_worker', 1), 1),
            'rollout_fragment_length': max(config.get('rollout_fragment_length', 200), 50),
            
            # Bio-inspired parameters
            'hidden_dim': max(config.get('hidden_dim', 256), 64),
            'num_attention_heads': max(config.get('num_attention_heads', 8), 1),
            'pheromone_decay': max(min(config.get('pheromone_decay', 0.95), 1.0), 0.0),
            'neural_plasticity_rate': max(min(config.get('neural_plasticity_rate', 0.1), 1.0), 0.0),
            'communication_range': max(config.get('communication_range', 2.0), 0.5),
            'breakthrough_threshold': max(min(config.get('breakthrough_threshold', 0.7), 1.0), 0.0),
            
            # Monitoring
            'checkpoint_frequency': max(config.get('checkpoint_frequency', 10), 1),
            'evaluation_interval': max(config.get('evaluation_interval', 5), 1),
            'evaluation_duration': max(config.get('evaluation_duration', 10), 1),
        }
        
        logger.info(f"Configuration validated for experiment: {validated_config['experiment_name']}")
        return validated_config

# Global API instance
ray_training_api = RayTrainingAPI()

# Command-line interface for training execution
async def main():
    """Main entry point for Ray training"""
    
    if len(sys.argv) < 2:
        print("Usage: python ray_api_integration.py <config_json>")
        print("Example config:")
        template = ray_training_api.create_ray_config_template()
        print(json.dumps(template['config_template'], indent=2))
        return
    
    try:
        # Parse configuration from command line
        config_json = sys.argv[1]
        config = json.loads(config_json)
        
        logger.info(f"Starting Ray training with config: {config.get('experiment_name', 'Unknown')}")
        
        # Start training
        result = await ray_training_api.start_ray_training(config)
        
        # Output result
        print(json.dumps(result, indent=2))
        
        if result['success']:
            logger.info("Ray training completed successfully")
        else:
            logger.error(f"Ray training failed: {result['message']}")
            sys.exit(1)
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())