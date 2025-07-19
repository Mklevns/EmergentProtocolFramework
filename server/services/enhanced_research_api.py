"""
Enhanced Research API
REST API endpoints for the systematic research framework
"""

import asyncio
import json
import logging
import time
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from .research_framework import (
    ExperimentOrchestrator, 
    ResearchHypothesis, 
    ExperimentConfig, 
    StatisticalAnalyzer,
    HypothesisStatus,
    initialize_research_framework,
    get_research_orchestrator
)
from .rllib_experiment import create_rllib_experiment, EXAMPLE_CONFIGS
from .ray_training_orchestrator import RayTrainingOrchestrator

logger = logging.getLogger(__name__)

class EnhancedResearchAPI:
    """API service for systematic research framework"""
    
    def __init__(self):
        self.orchestrator: Optional[ExperimentOrchestrator] = None
        self.active_experiments: Dict[str, Any] = {}
        self.research_results: Dict[str, Any] = {}
        
        # Initialize with default configuration
        self._initialize_default_framework()
    
    def _initialize_default_framework(self):
        """Initialize framework with default settings"""
        try:
            default_config = {
                "framework_name": "Emergent Communication Research",
                "version": "1.0.0",
                "initialized_at": time.time()
            }
            
            output_dir = Path("./research_results")
            self.orchestrator = initialize_research_framework(default_config, output_dir)
            
            # Load default hypotheses
            self._load_default_hypotheses()
            
            logger.info("Enhanced research framework initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize research framework: {e}")
    
    def _load_default_hypotheses(self):
        """Load default research hypotheses"""
        if not self.orchestrator:
            return
        
        default_hypotheses = [
            ResearchHypothesis(
                hypothesis_id="H1_pheromone_emergence",
                title="Pheromone-like Communication Emergence",
                description="Agents will develop pheromone-like trail communication under resource scarcity",
                independent_variables=["resource_scarcity", "agent_density"],
                dependent_variables=["communication_frequency", "coordination_efficiency"],
                predicted_outcome="Higher resource scarcity will lead to increased pheromone communication",
                status=HypothesisStatus.PROPOSED,
                confidence_threshold=0.8
            ),
            ResearchHypothesis(
                hypothesis_id="H2_swarm_coordination", 
                title="Swarm Coordination Protocol Development",
                description="Larger agent groups will develop hierarchical communication patterns",
                independent_variables=["num_agents", "grid_complexity"],
                dependent_variables=["protocol_complexity", "semantic_stability"],
                predicted_outcome="Larger swarms will exhibit more complex but stable communication protocols",
                status=HypothesisStatus.PROPOSED,
                confidence_threshold=0.75
            ),
            ResearchHypothesis(
                hypothesis_id="H3_environmental_pressure",
                title="Environmental Pressure Communication Adaptation", 
                description="Communication protocols adapt to environmental pressures and noise",
                independent_variables=["environmental_noise", "task_complexity"],
                dependent_variables=["protocol_complexity", "mutual_information"],
                predicted_outcome="Higher environmental pressure will lead to more robust communication",
                status=HypothesisStatus.PROPOSED,
                confidence_threshold=0.7
            )
        ]
        
        for hypothesis in default_hypotheses:
            self.orchestrator.add_hypothesis(hypothesis)
    
    async def get_research_status(self) -> Dict[str, Any]:
        """Get current research framework status"""
        
        if not self.orchestrator:
            return {"status": "not_initialized", "error": "Research framework not initialized"}
        
        status = {
            "status": "active",
            "framework_initialized": True,
            "active_experiments": len(self.active_experiments),
            "total_hypotheses": len(self.orchestrator.hypotheses),
            "hypotheses": {
                hypothesis_id: {
                    "title": hypothesis.title,
                    "status": hypothesis.status.value,
                    "confidence_threshold": hypothesis.confidence_threshold
                }
                for hypothesis_id, hypothesis in self.orchestrator.hypotheses.items()
            },
            "available_experiment_types": list(EXAMPLE_CONFIGS.keys()),
            "last_updated": time.time()
        }
        
        return status
    
    async def create_experiment_from_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new research experiment from configuration"""
        
        try:
            # Validate required fields
            required_fields = ["experiment_name", "hypothesis_id", "environment_type", "num_agents"]
            for field in required_fields:
                if field not in config_data:
                    return {"error": f"Missing required field: {field}"}
            
            # Generate experiment ID if not provided
            if "experiment_id" not in config_data:
                timestamp = int(time.time())
                config_data["experiment_id"] = f"exp_{timestamp}"
            
            # Set defaults for optional fields
            defaults = {
                "description": f"Research experiment for {config_data['experiment_name']}",
                "grid_size": (4, 3, 3),
                "communication_range": 2.0,
                "agent_architecture": "bio_inspired_ppo",
                "hidden_dim": 256,
                "attention_heads": 8,
                "training_steps": 1000,
                "episodes_per_step": 1,
                "learning_rate": 3e-4,
                "batch_size": 128,
                "baseline_episodes": 200,
                "intervention_episodes": 500,
                "validation_episodes": 300,
                "reward_structure": {
                    "individual_collection": 1.0,
                    "collective_efficiency": 2.0,
                    "communication_cost": -0.1
                },
                "significance_level": 0.05,
                "effect_size_threshold": 0.2,
                "minimum_sample_size": 100,
                "pressure_conditions": []
            }
            
            # Apply defaults for missing fields
            for key, default_value in defaults.items():
                if key not in config_data:
                    config_data[key] = default_value
            
            # Create experiment configuration
            experiment_config = ExperimentConfig(**config_data)
            
            # Create output directory
            output_dir = Path(f"./research_results/{experiment_config.experiment_id}")
            
            # Create RLlib experiment
            experiment = create_rllib_experiment(experiment_config, output_dir)
            
            # Store experiment
            experiment_data = {
                "experiment": experiment,
                "config": experiment_config,
                "created_at": time.time(),
                "status": "created",
                "progress": 0.0
            }
            
            self.active_experiments[experiment_config.experiment_id] = experiment_data
            
            return {
                "success": True,
                "experiment_id": experiment_config.experiment_id,
                "experiment_name": experiment_config.experiment_name,
                "hypothesis_id": experiment_config.hypothesis_id,
                "estimated_duration": self._estimate_experiment_duration(experiment_config),
                "created_at": experiment_data["created_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return {"error": f"Failed to create experiment: {str(e)}"}
    
    async def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Execute a research experiment"""
        
        if experiment_id not in self.active_experiments:
            return {"error": f"Experiment {experiment_id} not found"}
        
        experiment_data = self.active_experiments[experiment_id]
        experiment = experiment_data["experiment"]
        
        try:
            # Update status
            experiment_data["status"] = "running"
            experiment_data["started_at"] = time.time()
            
            logger.info(f"Starting experiment {experiment_id}")
            
            # Execute experiment
            results = experiment.execute_full_experiment()
            
            # Update status and store results
            experiment_data["status"] = "completed"
            experiment_data["completed_at"] = time.time()
            experiment_data["results"] = results
            
            # Store in research results
            self.research_results[experiment_id] = results
            
            # Analyze hypothesis validation
            hypothesis_analysis = await self._analyze_hypothesis_validation(experiment_id, results)
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "results": results,
                "hypothesis_analysis": hypothesis_analysis,
                "duration": experiment_data["completed_at"] - experiment_data["started_at"]
            }
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment_data["status"] = "failed"
            experiment_data["error"] = str(e)
            
            return {"error": f"Experiment failed: {str(e)}"}
        
        finally:
            # Cleanup experiment resources
            if hasattr(experiment, 'cleanup'):
                experiment.cleanup()
    
    async def get_experiment_progress(self, experiment_id: str) -> Dict[str, Any]:
        """Get progress of a running experiment"""
        
        if experiment_id not in self.active_experiments:
            return {"error": f"Experiment {experiment_id} not found"}
        
        experiment_data = self.active_experiments[experiment_id]
        
        progress_data = {
            "experiment_id": experiment_id,
            "status": experiment_data["status"],
            "created_at": experiment_data["created_at"],
            "progress": experiment_data.get("progress", 0.0)
        }
        
        if "started_at" in experiment_data:
            progress_data["started_at"] = experiment_data["started_at"]
            progress_data["running_time"] = time.time() - experiment_data["started_at"]
        
        if "completed_at" in experiment_data:
            progress_data["completed_at"] = experiment_data["completed_at"]
            progress_data["total_duration"] = experiment_data["completed_at"] - experiment_data["started_at"]
        
        if "error" in experiment_data:
            progress_data["error"] = experiment_data["error"]
        
        return progress_data
    
    async def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        
        experiments_list = []
        
        for experiment_id, experiment_data in self.active_experiments.items():
            exp_info = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_data["config"].experiment_name,
                "hypothesis_id": experiment_data["config"].hypothesis_id,
                "status": experiment_data["status"],
                "created_at": experiment_data["created_at"]
            }
            
            if "started_at" in experiment_data:
                exp_info["started_at"] = experiment_data["started_at"]
            
            if "completed_at" in experiment_data:
                exp_info["completed_at"] = experiment_data["completed_at"]
                exp_info["duration"] = experiment_data["completed_at"] - experiment_data["started_at"]
            
            experiments_list.append(exp_info)
        
        return {
            "experiments": experiments_list,
            "total_count": len(experiments_list),
            "status_summary": self._get_status_summary()
        }
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed results for an experiment"""
        
        if experiment_id not in self.research_results:
            if experiment_id in self.active_experiments:
                exp_data = self.active_experiments[experiment_id]
                if exp_data["status"] != "completed":
                    return {"error": f"Experiment {experiment_id} not yet completed"}
                else:
                    return {"error": f"Results not found for experiment {experiment_id}"}
            else:
                return {"error": f"Experiment {experiment_id} not found"}
        
        results = self.research_results[experiment_id]
        
        # Add statistical analysis
        statistical_analysis = await self._perform_statistical_analysis(results)
        
        return {
            "experiment_id": experiment_id,
            "results": results,
            "statistical_analysis": statistical_analysis,
            "retrieved_at": time.time()
        }
    
    async def run_campaign_from_yaml(self, yaml_config: str) -> Dict[str, Any]:
        """Run a research campaign from YAML configuration"""
        
        try:
            # Parse YAML configuration
            config = yaml.safe_load(yaml_config)
            
            campaign_name = config.get("campaign_name", "research_campaign")
            base_config = config.get("base_config", {})
            variations = config.get("variations", {})
            experiments = config.get("experiments", {})
            
            # Create base experiment configuration
            base_exp_config = ExperimentConfig(
                experiment_id="base",
                hypothesis_id=base_config.get("hypothesis_id", "H1_pheromone_emergence"),
                **base_config
            )
            
            # Generate experiment variations
            experiment_configs = []
            
            if variations:
                # Create variations of base config
                variation_configs = self.orchestrator.create_experiment_variations(
                    base_exp_config, variations
                )
                experiment_configs.extend(variation_configs)
            
            # Add specific experiment configurations
            for exp_name, exp_config in experiments.items():
                # Merge with base config
                merged_config = {**base_config, **exp_config}
                merged_config["experiment_id"] = exp_name
                merged_config["experiment_name"] = exp_config.get("experiment_name", exp_name)
                
                exp_config_obj = ExperimentConfig(**merged_config)
                experiment_configs.append(exp_config_obj)
            
            # Create experiments
            for config in experiment_configs:
                output_dir = Path(f"./research_results/{campaign_name}/{config.experiment_id}")
                experiment = create_rllib_experiment(config, output_dir)
                self.orchestrator.experiments.append(experiment)
            
            # Run campaign
            campaign_results = self.orchestrator.run_research_campaign(
                campaign_name,
                create_rllib_experiment,
                parallel_execution=False  # Sequential for stability
            )
            
            return {
                "success": True,
                "campaign_name": campaign_name,
                "experiment_count": len(experiment_configs),
                "results": campaign_results
            }
            
        except Exception as e:
            logger.error(f"Campaign execution failed: {e}")
            return {"error": f"Campaign failed: {str(e)}"}
    
    async def get_hypothesis_validation_summary(self) -> Dict[str, Any]:
        """Get summary of hypothesis validation across all experiments"""
        
        if not self.orchestrator:
            return {"error": "Research framework not initialized"}
        
        validation_summary = {}
        
        for hypothesis_id, hypothesis in self.orchestrator.hypotheses.items():
            # Find experiments testing this hypothesis
            relevant_experiments = [
                (exp_id, results) for exp_id, results in self.research_results.items()
                if results.get("hypothesis_id") == hypothesis_id
            ]
            
            if relevant_experiments:
                # Analyze validation across experiments
                validation_results = []
                for exp_id, results in relevant_experiments:
                    analysis = await self._analyze_hypothesis_validation(exp_id, results)
                    validation_results.append(analysis)
                
                # Calculate overall validation
                overall_confidence = sum(r.get("confidence", 0) for r in validation_results) / len(validation_results)
                overall_validated = overall_confidence >= hypothesis.confidence_threshold
                
                validation_summary[hypothesis_id] = {
                    "hypothesis": asdict(hypothesis),
                    "experiment_count": len(relevant_experiments),
                    "overall_confidence": overall_confidence,
                    "overall_validated": overall_validated,
                    "individual_results": validation_results
                }
            else:
                validation_summary[hypothesis_id] = {
                    "hypothesis": asdict(hypothesis),
                    "experiment_count": 0,
                    "status": "not_tested"
                }
        
        return {
            "hypothesis_summary": validation_summary,
            "total_hypotheses": len(self.orchestrator.hypotheses),
            "tested_hypotheses": sum(1 for h in validation_summary.values() if h.get("experiment_count", 0) > 0)
        }
    
    def _estimate_experiment_duration(self, config: ExperimentConfig) -> float:
        """Estimate experiment duration in seconds"""
        
        total_episodes = config.baseline_episodes + config.intervention_episodes + config.validation_episodes
        
        # Rough estimation: 0.1 seconds per episode for simple environments
        base_time = total_episodes * 0.1
        
        # Adjust for complexity
        complexity_factor = 1.0
        complexity_factor *= (config.num_agents / 8.0)  # Scale with agent count
        complexity_factor *= (config.hidden_dim / 256.0)  # Scale with model complexity
        
        estimated_duration = base_time * complexity_factor
        
        return estimated_duration
    
    async def _analyze_hypothesis_validation(self, experiment_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hypothesis validation for an experiment"""
        
        try:
            hypothesis_id = results.get("hypothesis_id")
            if not hypothesis_id or hypothesis_id not in self.orchestrator.hypotheses:
                return {"error": "No valid hypothesis found for experiment"}
            
            hypothesis = self.orchestrator.hypotheses[hypothesis_id]
            
            # Extract phase results for statistical comparison
            phases = results.get("phases", {})
            baseline_phase = phases.get("ExperimentPhase.BASELINE")
            intervention_phase = phases.get("ExperimentPhase.INTERVENTION")
            
            if not baseline_phase or not intervention_phase:
                return {"error": "Missing baseline or intervention phase data"}
            
            # Perform statistical comparison
            statistical_results = StatisticalAnalyzer.compare_phases(
                baseline_phase, intervention_phase, hypothesis.confidence_threshold
            )
            
            # Validate hypothesis
            validated, confidence, explanation = StatisticalAnalyzer.validate_hypothesis(
                hypothesis, statistical_results, hypothesis.confidence_threshold
            )
            
            return {
                "experiment_id": experiment_id,
                "hypothesis_id": hypothesis_id,
                "validated": validated,
                "confidence": confidence,
                "explanation": explanation,
                "statistical_results": statistical_results,
                "analyzed_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Hypothesis validation analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _perform_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on experiment results"""
        
        try:
            phases = results.get("phases", {})
            
            if len(phases) < 2:
                return {"error": "Insufficient phase data for statistical analysis"}
            
            analysis = {
                "phase_comparisons": {},
                "trend_analysis": {},
                "effect_sizes": {},
                "significance_tests": {}
            }
            
            # Compare each phase with baseline
            baseline_key = None
            for phase_key in phases.keys():
                if "baseline" in phase_key.lower():
                    baseline_key = phase_key
                    break
            
            if baseline_key:
                baseline_data = phases[baseline_key]
                
                for phase_key, phase_data in phases.items():
                    if phase_key != baseline_key:
                        comparison = StatisticalAnalyzer.compare_phases(
                            baseline_data, phase_data
                        )
                        analysis["phase_comparisons"][phase_key] = comparison
            
            # Calculate trends
            for metric_name in ["coordination_efficiency", "mutual_information", "communication_frequency"]:
                metric_trends = []
                for phase_data in phases.values():
                    individual_metrics = phase_data.get("individual_metrics", {})
                    if metric_name in individual_metrics:
                        metric_trends.append(individual_metrics[metric_name]["mean"])
                
                if len(metric_trends) >= 2:
                    # Simple trend calculation
                    trend_direction = "increasing" if metric_trends[-1] > metric_trends[0] else "decreasing"
                    trend_magnitude = abs(metric_trends[-1] - metric_trends[0])
                    
                    analysis["trend_analysis"][metric_name] = {
                        "direction": trend_direction,
                        "magnitude": trend_magnitude,
                        "values": metric_trends
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _get_status_summary(self) -> Dict[str, int]:
        """Get summary of experiment statuses"""
        
        status_counts = {}
        for experiment_data in self.active_experiments.values():
            status = experiment_data["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return status_counts

# Global instance
_research_api: Optional[EnhancedResearchAPI] = None

def get_research_api() -> EnhancedResearchAPI:
    """Get global research API instance"""
    global _research_api
    if _research_api is None:
        _research_api = EnhancedResearchAPI()
    return _research_api

def initialize_research_api() -> EnhancedResearchAPI:
    """Initialize and return research API"""
    global _research_api
    _research_api = EnhancedResearchAPI()
    return _research_api