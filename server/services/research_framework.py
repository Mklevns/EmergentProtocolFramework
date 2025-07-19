"""
Enhanced Research Framework for Emergent Communication
Systematic research framework integrating with the existing MARL platform
"""

import asyncio
import json
import logging
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class HypothesisStatus(Enum):
    """Status of research hypothesis"""
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"

class ExperimentPhase(Enum):
    """Phases of experiment execution"""
    SETUP = "setup"
    BASELINE = "baseline"
    INTERVENTION = "intervention"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    COMPLETE = "complete"

@dataclass
class EmergenceMetrics:
    """Metrics for measuring emergent communication"""
    coordination_efficiency: float
    mutual_information: float
    communication_frequency: float
    protocol_complexity: float
    semantic_stability: float
    compositional_structure: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def combined_score(self) -> float:
        """Calculate weighted combined emergence score"""
        weights = {
            'coordination_efficiency': 0.25,
            'mutual_information': 0.20,
            'communication_frequency': 0.15,
            'protocol_complexity': 0.15,
            'semantic_stability': 0.15,
            'compositional_structure': 0.10
        }
        
        return sum(getattr(self, metric) * weight for metric, weight in weights.items())

@dataclass
class ResearchHypothesis:
    """Structured research hypothesis"""
    hypothesis_id: str
    title: str
    description: str
    independent_variables: List[str]
    dependent_variables: List[str]
    predicted_outcome: str
    status: HypothesisStatus
    confidence_threshold: float = 0.8
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class ExperimentConfig:
    """Configuration for systematic research experiments"""
    experiment_id: str
    hypothesis_id: str
    experiment_name: str
    description: str
    
    # Environment configuration
    environment_type: str
    num_agents: int
    grid_size: Tuple[int, int, int]
    communication_range: float
    
    # Agent architecture
    agent_architecture: str
    hidden_dim: int
    attention_heads: int
    
    # Training parameters
    training_steps: int
    episodes_per_step: int
    learning_rate: float
    batch_size: int
    
    # Research-specific parameters
    baseline_episodes: int
    intervention_episodes: int
    validation_episodes: int
    
    # Reward structure
    reward_structure: Dict[str, float]
    
    # Analysis parameters
    significance_level: float = 0.05
    effect_size_threshold: float = 0.2
    minimum_sample_size: int = 100
    
    # Environmental pressures to test
    pressure_conditions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.pressure_conditions is None:
            self.pressure_conditions = []

class ResearchExperiment(ABC):
    """Abstract base class for research experiments"""
    
    def __init__(self, config: ExperimentConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_phase = ExperimentPhase.SETUP
        self.metrics_history: List[Dict[str, Any]] = []
        self.phase_results: Dict[ExperimentPhase, Dict[str, Any]] = {}
        
        self.start_time = None
        self.end_time = None
        
        logger.info(f"Research experiment initialized: {config.experiment_name}")
    
    @abstractmethod
    def setup_environment(self) -> None:
        """Setup the experimental environment"""
        pass
    
    @abstractmethod
    def setup_agents(self) -> None:
        """Configure and initialize agents"""
        pass
    
    @abstractmethod
    def run_training_step(self) -> EmergenceMetrics:
        """Execute one training step and return metrics"""
        pass
    
    @abstractmethod
    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze emergent communication patterns"""
        pass
    
    def execute_full_experiment(self) -> Dict[str, Any]:
        """Execute the complete research experiment"""
        self.start_time = time.time()
        
        try:
            # Setup phase
            self._transition_phase(ExperimentPhase.SETUP)
            self.setup_environment()
            self.setup_agents()
            
            # Baseline phase
            self._transition_phase(ExperimentPhase.BASELINE)
            baseline_results = self._run_phase_training(self.config.baseline_episodes, "baseline")
            
            # Intervention phase
            self._transition_phase(ExperimentPhase.INTERVENTION)
            intervention_results = self._run_phase_training(self.config.intervention_episodes, "intervention")
            
            # Validation phase
            self._transition_phase(ExperimentPhase.VALIDATION)
            validation_results = self._run_phase_training(self.config.validation_episodes, "validation")
            
            # Analysis phase
            self._transition_phase(ExperimentPhase.ANALYSIS)
            analysis_results = self.analyze_communication_patterns()
            
            # Complete experiment
            self._transition_phase(ExperimentPhase.COMPLETE)
            self.end_time = time.time()
            
            # Compile final results
            final_results = {
                'experiment_id': self.config.experiment_id,
                'hypothesis_id': self.config.hypothesis_id,
                'phases': self.phase_results,
                'analysis': analysis_results,
                'duration': self.end_time - self.start_time,
                'total_episodes': sum([
                    self.config.baseline_episodes,
                    self.config.intervention_episodes,
                    self.config.validation_episodes
                ])
            }
            
            # Save results
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def _transition_phase(self, new_phase: ExperimentPhase):
        """Transition to a new experiment phase"""
        logger.info(f"Transitioning from {self.current_phase} to {new_phase}")
        self.current_phase = new_phase
    
    def _run_phase_training(self, episodes: int, phase_name: str) -> Dict[str, Any]:
        """Run training for a specific phase"""
        phase_metrics = []
        
        logger.info(f"Starting {phase_name} phase with {episodes} episodes")
        
        for episode in range(episodes):
            step_metrics = self.run_training_step()
            
            episode_data = {
                'episode': episode,
                'phase': phase_name,
                'timestamp': time.time(),
                'metrics': step_metrics.to_dict()
            }
            
            phase_metrics.append(episode_data)
            self.metrics_history.append(episode_data)
            
            # Log progress periodically
            if episode % 50 == 0:
                logger.info(f"{phase_name} phase: episode {episode}/{episodes}, "
                          f"emergence score: {step_metrics.combined_score():.3f}")
        
        # Calculate phase summary statistics
        phase_summary = self._calculate_phase_summary(phase_metrics)
        self.phase_results[self.current_phase] = phase_summary
        
        return phase_summary
    
    def _calculate_phase_summary(self, phase_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for a phase"""
        if not phase_metrics:
            return {}
        
        metrics_arrays = {}
        for metric_name in EmergenceMetrics.__dataclass_fields__.keys():
            values = [episode['metrics'][metric_name] for episode in phase_metrics]
            metrics_arrays[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self._calculate_trend(values)
            }
        
        # Calculate combined emergence scores
        emergence_scores = [
            sum(episode['metrics'][metric] * weight 
                for metric, weight in {
                    'coordination_efficiency': 0.25,
                    'mutual_information': 0.20,
                    'communication_frequency': 0.15,
                    'protocol_complexity': 0.15,
                    'semantic_stability': 0.15,
                    'compositional_structure': 0.10
                }.items())
            for episode in phase_metrics
        ]
        
        return {
            'episode_count': len(phase_metrics),
            'individual_metrics': metrics_arrays,
            'emergence_score': {
                'mean': np.mean(emergence_scores),
                'std': np.std(emergence_scores),
                'trend': self._calculate_trend(emergence_scores)
            },
            'start_time': phase_metrics[0]['timestamp'],
            'end_time': phase_metrics[-1]['timestamp']
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to file"""
        results_file = self.output_dir / f"experiment_{self.config.experiment_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")

class StatisticalAnalyzer:
    """Statistical analysis for research results"""
    
    @staticmethod
    def compare_phases(baseline_data: Dict[str, Any], 
                      intervention_data: Dict[str, Any],
                      significance_level: float = 0.05) -> Dict[str, Any]:
        """Compare baseline and intervention phases statistically"""
        
        results = {}
        
        # Extract metrics for comparison
        baseline_metrics = baseline_data.get('individual_metrics', {})
        intervention_metrics = intervention_data.get('individual_metrics', {})
        
        for metric_name in baseline_metrics.keys():
            if metric_name in intervention_metrics:
                baseline_mean = baseline_metrics[metric_name]['mean']
                intervention_mean = intervention_metrics[metric_name]['mean']
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (baseline_metrics[metric_name]['std']**2 + 
                     intervention_metrics[metric_name]['std']**2) / 2
                )
                
                if pooled_std > 0:
                    effect_size = abs(intervention_mean - baseline_mean) / pooled_std
                else:
                    effect_size = 0
                
                # Determine significance (simplified - in practice use proper statistical tests)
                difference = abs(intervention_mean - baseline_mean)
                significance = difference > (baseline_metrics[metric_name]['std'] * 2)
                
                results[metric_name] = {
                    'baseline_mean': baseline_mean,
                    'intervention_mean': intervention_mean,
                    'difference': intervention_mean - baseline_mean,
                    'effect_size': effect_size,
                    'statistically_significant': significance,
                    'practical_significance': effect_size > 0.2
                }
        
        return results
    
    @staticmethod
    def validate_hypothesis(hypothesis: ResearchHypothesis,
                           statistical_results: Dict[str, Any],
                           confidence_threshold: float) -> Tuple[bool, float, str]:
        """Validate research hypothesis based on statistical results"""
        
        # Calculate overall confidence based on multiple metrics
        significant_metrics = sum(1 for result in statistical_results.values() 
                                if result.get('statistically_significant', False))
        
        total_metrics = len(statistical_results)
        confidence = significant_metrics / total_metrics if total_metrics > 0 else 0
        
        # Determine validation status
        if confidence >= confidence_threshold:
            validation_status = "validated"
            explanation = f"Hypothesis validated with {confidence:.2%} of metrics showing significance"
        elif confidence >= 0.5:
            validation_status = "partially_validated" 
            explanation = f"Hypothesis partially validated with {confidence:.2%} of metrics showing significance"
        else:
            validation_status = "not_validated"
            explanation = f"Hypothesis not validated - only {confidence:.2%} of metrics showed significance"
        
        return confidence >= confidence_threshold, confidence, explanation

class ExperimentOrchestrator:
    """Orchestrates multiple research experiments"""
    
    def __init__(self, base_config: Dict[str, Any], output_dir: Path):
        self.base_config = base_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: List[ResearchExperiment] = []
        self.hypotheses: Dict[str, ResearchHypothesis] = {}
        self.campaign_results: Dict[str, Any] = {}
        
        logger.info("Experiment orchestrator initialized")
    
    def add_hypothesis(self, hypothesis: ResearchHypothesis):
        """Add a research hypothesis to test"""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        logger.info(f"Added hypothesis: {hypothesis.title}")
    
    def create_experiment_variations(self, 
                                   base_config: ExperimentConfig,
                                   variations: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Create experiment configurations for parameter variations"""
        
        experiment_configs = []
        
        # Generate all combinations of variations
        variation_keys = list(variations.keys())
        variation_values = list(variations.values())
        
        import itertools
        for combination in itertools.product(*variation_values):
            # Create new config with variations
            config_dict = asdict(base_config)
            
            for key, value in zip(variation_keys, combination):
                if '.' in key:  # Handle nested parameters
                    keys = key.split('.')
                    nested_dict = config_dict
                    for nested_key in keys[:-1]:
                        nested_dict = nested_dict[nested_key]
                    nested_dict[keys[-1]] = value
                else:
                    config_dict[key] = value
            
            # Generate unique experiment ID
            variation_str = "_".join(f"{k}_{v}" for k, v in zip(variation_keys, combination))
            config_dict['experiment_id'] = f"{base_config.experiment_id}_{variation_str}"
            config_dict['experiment_name'] = f"{base_config.experiment_name} - {variation_str}"
            
            experiment_configs.append(ExperimentConfig(**config_dict))
        
        return experiment_configs
    
    def run_research_campaign(self, 
                            campaign_name: str,
                            experiment_factory,
                            parallel_execution: bool = False) -> Dict[str, Any]:
        """Run a complete research campaign"""
        
        logger.info(f"Starting research campaign: {campaign_name}")
        campaign_start = time.time()
        
        campaign_dir = self.output_dir / campaign_name
        campaign_dir.mkdir(exist_ok=True)
        
        experiment_results = []
        
        if parallel_execution:
            # Run experiments in parallel (simplified implementation)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for experiment in self.experiments:
                    future = executor.submit(experiment.execute_full_experiment)
                    futures.append((experiment, future))
                
                for experiment, future in futures:
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout
                        experiment_results.append(result)
                    except Exception as e:
                        logger.error(f"Experiment {experiment.config.experiment_id} failed: {e}")
        else:
            # Run experiments sequentially
            for experiment in self.experiments:
                try:
                    result = experiment.execute_full_experiment()
                    experiment_results.append(result)
                except Exception as e:
                    logger.error(f"Experiment {experiment.config.experiment_id} failed: {e}")
        
        # Analyze campaign results
        campaign_analysis = self._analyze_campaign_results(experiment_results)
        
        campaign_results = {
            'campaign_name': campaign_name,
            'start_time': campaign_start,
            'end_time': time.time(),
            'experiment_count': len(experiment_results),
            'successful_experiments': len(experiment_results),
            'experiments': experiment_results,
            'campaign_analysis': campaign_analysis
        }
        
        # Save campaign results
        campaign_file = campaign_dir / f"{campaign_name}_results.json"
        with open(campaign_file, 'w') as f:
            json.dump(campaign_results, f, indent=2, default=str)
        
        logger.info(f"Campaign {campaign_name} completed with {len(experiment_results)} experiments")
        
        return campaign_results
    
    def _analyze_campaign_results(self, experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results across all experiments in a campaign"""
        
        if not experiment_results:
            return {}
        
        # Aggregate emergence scores across all experiments
        all_emergence_scores = []
        hypothesis_validations = {}
        
        for result in experiment_results:
            # Extract emergence scores from each phase
            for phase_name, phase_data in result.get('phases', {}).items():
                if 'emergence_score' in phase_data:
                    score_data = phase_data['emergence_score']
                    all_emergence_scores.append(score_data['mean'])
            
            # Track hypothesis validation
            hypothesis_id = result.get('hypothesis_id')
            if hypothesis_id and hypothesis_id in self.hypotheses:
                # Simplified validation based on emergence score improvement
                baseline_score = result.get('phases', {}).get('baseline', {}).get('emergence_score', {}).get('mean', 0)
                intervention_score = result.get('phases', {}).get('intervention', {}).get('emergence_score', {}).get('mean', 0)
                
                improvement = intervention_score - baseline_score
                hypothesis_validations[hypothesis_id] = {
                    'improvement': improvement,
                    'validated': improvement > 0.05,  # Threshold for practical significance
                    'confidence': min(1.0, abs(improvement) * 10)  # Simplified confidence calculation
                }
        
        return {
            'emergence_score_statistics': {
                'mean': np.mean(all_emergence_scores) if all_emergence_scores else 0,
                'std': np.std(all_emergence_scores) if all_emergence_scores else 0,
                'min': np.min(all_emergence_scores) if all_emergence_scores else 0,
                'max': np.max(all_emergence_scores) if all_emergence_scores else 0
            },
            'hypothesis_validations': hypothesis_validations,
            'successful_experiment_rate': len(experiment_results) / len(self.experiments) if self.experiments else 0
        }

# Global instance for research framework
_research_orchestrator: Optional[ExperimentOrchestrator] = None

def initialize_research_framework(config: Dict[str, Any], output_dir: Path) -> ExperimentOrchestrator:
    """Initialize the global research framework"""
    global _research_orchestrator
    _research_orchestrator = ExperimentOrchestrator(config, output_dir)
    return _research_orchestrator

def get_research_orchestrator() -> ExperimentOrchestrator:
    """Get the global research orchestrator instance"""
    if _research_orchestrator is None:
        raise RuntimeError("Research framework not initialized. Call initialize_research_framework first.")
    return _research_orchestrator