"""
Transfer Learning Implementation
Advanced knowledge transfer between training experiments and agent models
"""

import asyncio
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import pickle
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)

class TransferType(Enum):
    """Types of transfer learning"""
    FULL_MODEL = "full_model"
    ATTENTION_ONLY = "attention_only"
    MEMORY_ONLY = "memory_only"
    FEATURE_EXTRACTION = "feature_extraction"
    FINE_TUNING = "fine_tuning"
    PROGRESSIVE_UNFREEZING = "progressive_unfreezing"

@dataclass
class TransferConfig:
    """Configuration for transfer learning"""
    source_experiment: str
    target_experiment: str
    transfer_type: TransferType
    components: List[str]  # ["attention", "memory", "communication"]
    freeze_layers: List[str] = None
    learning_rate_multiplier: float = 0.1
    warmup_episodes: int = 50
    validation_threshold: float = 0.7
    
class ModelState:
    """Manages model state for transfer learning"""
    
    def __init__(self, experiment_id: str, model_data: Dict[str, Any]):
        self.experiment_id = experiment_id
        self.model_data = model_data
        self.metadata = model_data.get('metadata', {})
        self.state_dict = model_data.get('state_dict', {})
        self.architecture = model_data.get('architecture', {})
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.timestamp = model_data.get('timestamp', time.time())
        
    def get_component_state(self, component: str) -> Optional[Dict[str, Any]]:
        """Extract state for specific component"""
        component_states = {
            'attention': self._extract_attention_state(),
            'memory': self._extract_memory_state(),
            'communication': self._extract_communication_state(),
            'coordination': self._extract_coordination_state(),
            'plasticity': self._extract_plasticity_state()
        }
        return component_states.get(component)
        
    def _extract_attention_state(self) -> Dict[str, Any]:
        """Extract attention mechanism state"""
        attention_keys = [k for k in self.state_dict.keys() if 'attention' in k.lower()]
        return {k: self.state_dict[k] for k in attention_keys}
        
    def _extract_memory_state(self) -> Dict[str, Any]:
        """Extract memory system state"""
        memory_keys = [k for k in self.state_dict.keys() if 'memory' in k.lower()]
        return {k: self.state_dict[k] for k in memory_keys}
        
    def _extract_communication_state(self) -> Dict[str, Any]:
        """Extract communication protocol state"""
        comm_keys = [k for k in self.state_dict.keys() if any(term in k.lower() for term in ['comm', 'message', 'protocol'])]
        return {k: self.state_dict[k] for k in comm_keys}
        
    def _extract_coordination_state(self) -> Dict[str, Any]:
        """Extract coordination mechanism state"""
        coord_keys = [k for k in self.state_dict.keys() if 'coord' in k.lower()]
        return {k: self.state_dict[k] for k in coord_keys}
        
    def _extract_plasticity_state(self) -> Dict[str, Any]:
        """Extract neural plasticity state"""
        plasticity_keys = [k for k in self.state_dict.keys() if 'plasticity' in k.lower()]
        return {k: self.state_dict[k] for k in plasticity_keys}

class TransferCompatibilityChecker:
    """Checks compatibility between source and target models"""
    
    def __init__(self):
        self.compatibility_rules = {
            'architecture_match': self._check_architecture_compatibility,
            'dimension_match': self._check_dimension_compatibility,
            'component_match': self._check_component_compatibility,
            'version_match': self._check_version_compatibility
        }
        
    def check_compatibility(self, source: ModelState, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check compatibility between source and target"""
        compatibility_results = {}
        
        for rule_name, rule_func in self.compatibility_rules.items():
            try:
                compatibility_results[rule_name] = rule_func(source, target_config)
            except Exception as e:
                logger.warning(f"Compatibility check failed for {rule_name}: {e}")
                compatibility_results[rule_name] = {'compatible': False, 'error': str(e)}
                
        # Overall compatibility score
        compatible_count = sum(1 for result in compatibility_results.values() if result.get('compatible', False))
        compatibility_score = compatible_count / len(compatibility_results)
        
        return {
            'overall_compatibility': compatibility_score,
            'is_compatible': compatibility_score > 0.7,
            'detailed_results': compatibility_results,
            'recommendations': self._generate_recommendations(compatibility_results)
        }
        
    def _check_architecture_compatibility(self, source: ModelState, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if architectures are compatible"""
        source_arch = source.architecture
        target_arch = target_config.get('architecture', {})
        
        # Check key architectural components
        key_components = ['hidden_dim', 'num_heads', 'num_layers']
        matches = {}
        
        for component in key_components:
            source_val = source_arch.get(component)
            target_val = target_arch.get(component)
            matches[component] = source_val == target_val if source_val and target_val else False
            
        compatibility = sum(matches.values()) / len(matches) if matches else 0
        
        return {
            'compatible': compatibility > 0.5,
            'score': compatibility,
            'details': matches
        }
        
    def _check_dimension_compatibility(self, source: ModelState, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if model dimensions are compatible"""
        # Check tensor dimensions in state dict
        dimension_matches = 0
        total_checks = 0
        
        for key, tensor in source.state_dict.items():
            if hasattr(tensor, 'shape'):
                # Check if dimensions are reasonable for transfer
                total_checks += 1
                if len(tensor.shape) <= 3:  # Most layers should be transferable
                    dimension_matches += 1
                    
        compatibility = dimension_matches / total_checks if total_checks > 0 else 0
        
        return {
            'compatible': compatibility > 0.8,
            'score': compatibility,
            'total_parameters': total_checks,
            'compatible_parameters': dimension_matches
        }
        
    def _check_component_compatibility(self, source: ModelState, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if specific components are compatible"""
        source_components = set(source.state_dict.keys())
        required_components = set(target_config.get('required_components', []))
        
        if not required_components:
            # If no specific requirements, assume compatible
            return {'compatible': True, 'score': 1.0}
            
        available_components = source_components.intersection(required_components)
        compatibility = len(available_components) / len(required_components)
        
        return {
            'compatible': compatibility > 0.7,
            'score': compatibility,
            'available_components': list(available_components),
            'missing_components': list(required_components - available_components)
        }
        
    def _check_version_compatibility(self, source: ModelState, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check version compatibility"""
        source_version = source.metadata.get('version', '1.0.0')
        target_version = target_config.get('version', '1.0.0')
        
        # Simple version comparison (can be enhanced)
        compatible = source_version == target_version
        
        return {
            'compatible': compatible,
            'source_version': source_version,
            'target_version': target_version
        }
        
    def _generate_recommendations(self, compatibility_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on compatibility results"""
        recommendations = []
        
        # Architecture recommendations
        arch_result = compatibility_results.get('architecture_match', {})
        if not arch_result.get('compatible', False):
            recommendations.append("Consider using feature extraction or fine-tuning instead of full model transfer")
            
        # Dimension recommendations  
        dim_result = compatibility_results.get('dimension_match', {})
        if not dim_result.get('compatible', False):
            recommendations.append("Model dimensions may require adaptation layers")
            
        # Component recommendations
        comp_result = compatibility_results.get('component_match', {})
        if not comp_result.get('compatible', False):
            missing = comp_result.get('missing_components', [])
            if missing:
                recommendations.append(f"Initialize missing components: {', '.join(missing)}")
                
        return recommendations

class KnowledgeDistillation:
    """Knowledge distillation for transfer learning"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        
    def distillation_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, 
                         true_labels: torch.Tensor) -> torch.Tensor:
        """Calculate knowledge distillation loss"""
        
        # Soft targets from teacher
        soft_targets = torch.nn.functional.softmax(teacher_outputs / self.temperature, dim=-1)
        soft_predictions = torch.nn.functional.log_softmax(student_outputs / self.temperature, dim=-1)
        
        # Distillation loss
        distillation_loss = torch.nn.functional.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)
        
        # Hard targets loss
        hard_loss = torch.nn.functional.cross_entropy(student_outputs, true_labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss

class TransferLearningManager:
    """Main transfer learning manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transfer_config = config.get('transfer_learning', {})
        self.is_enabled = self.transfer_config.get('enabled', False)
        
        # Initialize components
        self.compatibility_checker = TransferCompatibilityChecker()
        self.knowledge_distillation = KnowledgeDistillation()
        
        # State management
        self.available_models = {}
        self.transfer_history = []
        self.active_transfers = {}
        
        # Storage paths
        self.model_storage_path = Path(config.get('model_storage_path', './models'))
        self.model_storage_path.mkdir(exist_ok=True)
        
        logger.info(f"Transfer Learning Manager initialized. Enabled: {self.is_enabled}")
        
    def is_transfer_enabled(self) -> bool:
        """Check if transfer learning is enabled"""
        return self.is_enabled
        
    def save_model_state(self, experiment_id: str, model_data: Dict[str, Any]) -> str:
        """Save model state for future transfer"""
        model_state = ModelState(experiment_id, model_data)
        
        # Save to storage
        storage_path = self.model_storage_path / f"{experiment_id}.pkl"
        
        with open(storage_path, 'wb') as f:
            pickle.dump(model_state, f)
            
        self.available_models[experiment_id] = {
            'path': str(storage_path),
            'metadata': model_state.metadata,
            'timestamp': model_state.timestamp,
            'performance': model_state.performance_metrics
        }
        
        logger.info(f"Saved model state for experiment {experiment_id}")
        return str(storage_path)
        
    def load_model_state(self, experiment_id: str) -> Optional[ModelState]:
        """Load model state from storage"""
        if experiment_id not in self.available_models:
            logger.error(f"Model state not found for experiment {experiment_id}")
            return None
            
        storage_path = self.available_models[experiment_id]['path']
        
        try:
            with open(storage_path, 'rb') as f:
                model_state = pickle.load(f)
            return model_state
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            return None
            
    def prepare_transfer(self, source_experiment: str, target_config: Dict[str, Any], 
                        transfer_components: List[str]) -> Dict[str, Any]:
        """Prepare for transfer learning"""
        
        # Load source model
        source_model = self.load_model_state(source_experiment)
        if not source_model:
            return {
                'success': False,
                'error': f"Source model {source_experiment} not found"
            }
            
        # Check compatibility
        compatibility = self.compatibility_checker.check_compatibility(source_model, target_config)
        
        if not compatibility['is_compatible']:
            logger.warning(f"Low compatibility score: {compatibility['overall_compatibility']:.3f}")
            
        # Prepare transfer plan
        transfer_plan = {
            'source_experiment': source_experiment,
            'target_config': target_config,
            'components': transfer_components,
            'compatibility': compatibility,
            'transfer_strategy': self._determine_transfer_strategy(compatibility, transfer_components),
            'timestamp': time.time()
        }
        
        return {
            'success': True,
            'transfer_plan': transfer_plan,
            'compatibility': compatibility
        }
        
    def _determine_transfer_strategy(self, compatibility: Dict[str, Any], components: List[str]) -> Dict[str, Any]:
        """Determine optimal transfer strategy"""
        
        compatibility_score = compatibility['overall_compatibility']
        
        if compatibility_score > 0.9:
            strategy = TransferType.FULL_MODEL
        elif compatibility_score > 0.7:
            strategy = TransferType.FINE_TUNING
        elif compatibility_score > 0.5:
            strategy = TransferType.FEATURE_EXTRACTION
        else:
            strategy = TransferType.PROGRESSIVE_UNFREEZING
            
        return {
            'type': strategy,
            'freeze_strategy': self._get_freeze_strategy(strategy, components),
            'learning_rate_schedule': self._get_lr_schedule(strategy),
            'warmup_episodes': self._get_warmup_episodes(strategy)
        }
        
    def _get_freeze_strategy(self, transfer_type: TransferType, components: List[str]) -> Dict[str, Any]:
        """Get layer freezing strategy"""
        
        freeze_strategies = {
            TransferType.FULL_MODEL: {'freeze_layers': []},
            TransferType.FINE_TUNING: {'freeze_layers': ['embedding', 'early_layers']},
            TransferType.FEATURE_EXTRACTION: {'freeze_layers': ['all_except_classifier']},
            TransferType.PROGRESSIVE_UNFREEZING: {'freeze_layers': ['progressive']},
            TransferType.ATTENTION_ONLY: {'freeze_layers': ['all_except_attention']},
            TransferType.MEMORY_ONLY: {'freeze_layers': ['all_except_memory']}
        }
        
        return freeze_strategies.get(transfer_type, {'freeze_layers': []})
        
    def _get_lr_schedule(self, transfer_type: TransferType) -> Dict[str, float]:
        """Get learning rate schedule for transfer type"""
        
        schedules = {
            TransferType.FULL_MODEL: {'initial': 1e-4, 'multiplier': 1.0},
            TransferType.FINE_TUNING: {'initial': 1e-5, 'multiplier': 0.1},
            TransferType.FEATURE_EXTRACTION: {'initial': 1e-3, 'multiplier': 1.0},
            TransferType.PROGRESSIVE_UNFREEZING: {'initial': 1e-6, 'multiplier': 0.01}
        }
        
        return schedules.get(transfer_type, {'initial': 1e-4, 'multiplier': 0.1})
        
    def _get_warmup_episodes(self, transfer_type: TransferType) -> int:
        """Get warmup episodes for transfer type"""
        
        warmup_episodes = {
            TransferType.FULL_MODEL: 20,
            TransferType.FINE_TUNING: 50,
            TransferType.FEATURE_EXTRACTION: 30,
            TransferType.PROGRESSIVE_UNFREEZING: 100
        }
        
        return warmup_episodes.get(transfer_type, 50)
        
    def execute_transfer(self, transfer_plan: Dict[str, Any], target_model: Any) -> Dict[str, Any]:
        """Execute the transfer learning process"""
        
        source_experiment = transfer_plan['source_experiment']
        components = transfer_plan['components']
        strategy = transfer_plan['transfer_strategy']
        
        # Load source model
        source_model = self.load_model_state(source_experiment)
        if not source_model:
            return {
                'success': False,
                'error': f"Failed to load source model {source_experiment}"
            }
            
        transfer_results = {
            'transferred_components': [],
            'skipped_components': [],
            'adaptation_info': {},
            'performance_impact': {}
        }
        
        try:
            # Transfer each component
            for component in components:
                component_state = source_model.get_component_state(component)
                if component_state:
                    success = self._transfer_component(target_model, component, component_state, strategy)
                    if success:
                        transfer_results['transferred_components'].append(component)
                    else:
                        transfer_results['skipped_components'].append(component)
                else:
                    transfer_results['skipped_components'].append(component)
                    
            # Record transfer in history
            transfer_record = {
                'source_experiment': source_experiment,
                'target_experiment': transfer_plan.get('target_experiment', 'current'),
                'components': transfer_results['transferred_components'],
                'strategy': strategy,
                'timestamp': time.time(),
                'success_rate': len(transfer_results['transferred_components']) / len(components) if components else 0
            }
            
            self.transfer_history.append(transfer_record)
            
            return {
                'success': True,
                'transfer_results': transfer_results,
                'transfer_record': transfer_record
            }
            
        except Exception as e:
            logger.error(f"Transfer execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _transfer_component(self, target_model: Any, component: str, component_state: Dict[str, Any], 
                          strategy: Dict[str, Any]) -> bool:
        """Transfer a specific component"""
        
        try:
            # Get the component in target model
            if hasattr(target_model, component):
                target_component = getattr(target_model, component)
                
                # Apply transfer based on strategy
                if strategy['type'] == TransferType.FULL_MODEL:
                    # Direct state transfer
                    target_component.load_state_dict(component_state, strict=False)
                    
                elif strategy['type'] == TransferType.FINE_TUNING:
                    # Transfer with frozen layers
                    target_component.load_state_dict(component_state, strict=False)
                    self._freeze_component_layers(target_component, strategy['freeze_strategy'])
                    
                elif strategy['type'] == TransferType.FEATURE_EXTRACTION:
                    # Transfer features only
                    self._transfer_features_only(target_component, component_state)
                    
                logger.info(f"Successfully transferred component: {component}")
                return True
                
            else:
                logger.warning(f"Component {component} not found in target model")
                return False
                
        except Exception as e:
            logger.error(f"Failed to transfer component {component}: {e}")
            return False
            
    def _freeze_component_layers(self, component: nn.Module, freeze_strategy: Dict[str, Any]):
        """Freeze layers in component based on strategy"""
        
        freeze_layers = freeze_strategy.get('freeze_layers', [])
        
        for name, param in component.named_parameters():
            should_freeze = any(layer_pattern in name for layer_pattern in freeze_layers)
            param.requires_grad = not should_freeze
            
    def _transfer_features_only(self, target_component: nn.Module, component_state: Dict[str, Any]):
        """Transfer only feature extraction layers"""
        
        # Only transfer layers that match and are feature extractors
        target_state = target_component.state_dict()
        
        for key, value in component_state.items():
            if key in target_state and 'classifier' not in key.lower():
                target_state[key] = value
                
        target_component.load_state_dict(target_state)
        
    def get_transfer_recommendations(self, target_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations for transfer learning"""
        
        recommendations = []
        
        for experiment_id, model_info in self.available_models.items():
            model_state = self.load_model_state(experiment_id)
            if model_state:
                compatibility = self.compatibility_checker.check_compatibility(model_state, target_config)
                
                if compatibility['overall_compatibility'] > 0.5:
                    recommendation = {
                        'source_experiment': experiment_id,
                        'compatibility_score': compatibility['overall_compatibility'],
                        'recommended_components': self._get_recommended_components(compatibility),
                        'transfer_strategy': self._determine_transfer_strategy(compatibility, ['attention', 'memory']),
                        'expected_benefit': self._estimate_transfer_benefit(model_state, target_config),
                        'model_performance': model_info['performance']
                    }
                    recommendations.append(recommendation)
                    
        # Sort by compatibility score
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return recommendations
        
    def _get_recommended_components(self, compatibility: Dict[str, Any]) -> List[str]:
        """Get recommended components for transfer"""
        
        # Basic components that usually transfer well
        recommended = ['attention', 'memory']
        
        # Add more based on compatibility
        if compatibility['overall_compatibility'] > 0.8:
            recommended.extend(['communication', 'coordination'])
            
        return recommended
        
    def _estimate_transfer_benefit(self, source_model: ModelState, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate potential benefit of transfer"""
        
        source_performance = source_model.performance_metrics
        
        benefit_estimate = {
            'training_speedup': 2.0 if source_performance.get('success_rate', 0) > 0.8 else 1.5,
            'final_performance_boost': 0.1 if source_performance.get('communication_efficiency', 0) > 0.9 else 0.05,
            'convergence_episodes': max(50, 200 - int(source_performance.get('success_rate', 0) * 100))
        }
        
        return benefit_estimate
        
    def get_transfer_status(self) -> Dict[str, Any]:
        """Get current transfer learning status"""
        
        return {
            'enabled': self.is_enabled,
            'available_models': len(self.available_models),
            'completed_transfers': len(self.transfer_history),
            'active_transfers': len(self.active_transfers),
            'model_list': list(self.available_models.keys()),
            'recent_transfers': self.transfer_history[-5:] if self.transfer_history else []
        }
        
    def export_transfer_data(self) -> Dict[str, Any]:
        """Export transfer learning data"""
        
        return {
            'config': self.transfer_config,
            'available_models': self.available_models,
            'transfer_history': self.transfer_history,
            'active_transfers': self.active_transfers,
            'storage_path': str(self.model_storage_path)
        }

# Global transfer manager instance
_transfer_manager: Optional[TransferLearningManager] = None

def get_transfer_manager() -> TransferLearningManager:
    """Get the global transfer learning manager"""
    global _transfer_manager
    if _transfer_manager is None:
        _transfer_manager = TransferLearningManager({})
    return _transfer_manager

def initialize_transfer_manager(config: Dict[str, Any]) -> TransferLearningManager:
    """Initialize transfer learning manager with configuration"""
    global _transfer_manager
    _transfer_manager = TransferLearningManager(config)
    return _transfer_manager

async def test_transfer_system():
    """Test transfer learning system"""
    
    # Mock model data
    mock_model_data = {
        'metadata': {'version': '1.0.0', 'experiment_type': 'bio_inspired_marl'},
        'state_dict': {
            'attention.query_proj.weight': torch.randn(256, 256),
            'attention.key_proj.weight': torch.randn(256, 256),
            'memory.embedding.weight': torch.randn(1000, 256),
            'communication.encoder.weight': torch.randn(256, 128)
        },
        'architecture': {'hidden_dim': 256, 'num_heads': 8, 'num_layers': 4},
        'performance_metrics': {'success_rate': 0.85, 'communication_efficiency': 0.92}
    }
    
    manager = initialize_transfer_manager({
        'transfer_learning': {'enabled': True},
        'model_storage_path': './test_models'
    })
    
    # Save mock model
    manager.save_model_state('experiment_1', mock_model_data)
    
    # Test transfer preparation
    target_config = {
        'architecture': {'hidden_dim': 256, 'num_heads': 8},
        'required_components': ['attention', 'memory']
    }
    
    result = manager.prepare_transfer('experiment_1', target_config, ['attention', 'memory'])
    print(f"Transfer preparation result: {result['success']}")
    
    # Get recommendations
    recommendations = manager.get_transfer_recommendations(target_config)
    print(f"Transfer recommendations: {len(recommendations)}")

if __name__ == "__main__":
    asyncio.run(test_transfer_system())