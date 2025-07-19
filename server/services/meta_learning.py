"""
Meta-Learning Implementation
Learn to learn - adaptive learning algorithms that improve from experience
"""

import asyncio
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import copy

logger = logging.getLogger(__name__)

class MetaAlgorithm(Enum):
    """Meta-learning algorithm types"""
    MAML = "model_agnostic_meta_learning"
    REPTILE = "reptile"
    ADAPTIVE_LR = "adaptive_learning_rate"
    GRADIENT_EPISODIC = "gradient_episodic_memory"
    PROGRESSIVE_NEURAL = "progressive_neural_networks"

@dataclass
class MetaTask:
    """Represents a meta-learning task"""
    task_id: str
    description: str
    support_episodes: int
    query_episodes: int
    difficulty: float
    environment_config: Dict[str, Any]
    success_threshold: float
    
class MetaLearningBuffer:
    """Buffer for storing meta-learning experiences"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.task_histories = defaultdict(list)
        
    def add_experience(self, task_id: str, episode_data: Dict[str, Any]):
        """Add a learning experience"""
        experience = {
            'task_id': task_id,
            'timestamp': time.time(),
            'episode_data': episode_data,
            'gradients': episode_data.get('gradients'),
            'loss': episode_data.get('loss'),
            'performance_metrics': episode_data.get('metrics', {})
        }
        
        self.experiences.append(experience)
        self.task_histories[task_id].append(experience)
        
    def get_task_experiences(self, task_id: str, num_episodes: int = None) -> List[Dict[str, Any]]:
        """Get experiences for a specific task"""
        task_exps = self.task_histories[task_id]
        if num_episodes:
            return task_exps[-num_episodes:]
        return task_exps
        
    def get_similar_tasks(self, target_task: MetaTask, num_tasks: int = 5) -> List[str]:
        """Find similar tasks based on characteristics"""
        task_similarities = []
        
        for task_id in self.task_histories.keys():
            # Simple similarity based on difficulty and environment
            # In practice, this would use more sophisticated similarity metrics
            if task_id != target_task.task_id:
                similarity = 1.0 - abs(target_task.difficulty - 0.5)  # Placeholder
                task_similarities.append((task_id, similarity))
                
        # Sort by similarity and return top tasks
        task_similarities.sort(key=lambda x: x[1], reverse=True)
        return [task_id for task_id, _ in task_similarities[:num_tasks]]

class MAMLOptimizer:
    """Model-Agnostic Meta-Learning optimizer"""
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.001, inner_lr: float = 0.01, 
                 inner_steps: int = 5):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        
    def meta_update(self, support_tasks: List[Dict[str, Any]], query_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform MAML meta-update"""
        
        meta_gradients = []
        task_losses = []
        
        for support_task, query_task in zip(support_tasks, query_tasks):
            # Inner loop: adapt to support task
            adapted_model = self._inner_loop_adaptation(support_task)
            
            # Outer loop: evaluate on query task
            query_loss = self._evaluate_adapted_model(adapted_model, query_task)
            task_losses.append(query_loss.item())
            
            # Compute meta-gradients
            meta_grad = torch.autograd.grad(query_loss, adapted_model.parameters(), 
                                          create_graph=True, retain_graph=True)
            meta_gradients.append(meta_grad)
            
        # Average meta-gradients and update
        avg_meta_grad = self._average_gradients(meta_gradients)
        self._apply_meta_gradients(avg_meta_grad)
        
        return {
            'meta_loss': np.mean(task_losses),
            'task_losses': task_losses,
            'gradient_norm': self._compute_gradient_norm(avg_meta_grad)
        }
        
    def _inner_loop_adaptation(self, support_task: Dict[str, Any]) -> nn.Module:
        """Adapt model to support task"""
        adapted_model = copy.deepcopy(self.model)
        
        # Create inner optimizer
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Inner loop updates
        for step in range(self.inner_steps):
            loss = self._compute_task_loss(adapted_model, support_task)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
        return adapted_model
        
    def _evaluate_adapted_model(self, adapted_model: nn.Module, query_task: Dict[str, Any]) -> torch.Tensor:
        """Evaluate adapted model on query task"""
        return self._compute_task_loss(adapted_model, query_task)
        
    def _compute_task_loss(self, model: nn.Module, task_data: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a specific task"""
        # This would be implemented based on the specific task structure
        # For now, return a placeholder
        return torch.tensor(0.5, requires_grad=True)
        
    def _average_gradients(self, gradient_list: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """Average gradients across tasks"""
        if not gradient_list:
            return tuple()
            
        avg_gradients = []
        for param_gradients in zip(*gradient_list):
            avg_grad = torch.stack(param_gradients).mean(dim=0)
            avg_gradients.append(avg_grad)
            
        return tuple(avg_gradients)
        
    def _apply_meta_gradients(self, meta_gradients: Tuple[torch.Tensor]):
        """Apply meta-gradients to model parameters"""
        for param, meta_grad in zip(self.model.parameters(), meta_gradients):
            param.grad = meta_grad
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        
    def _compute_gradient_norm(self, gradients: Tuple[torch.Tensor]) -> float:
        """Compute norm of gradients"""
        total_norm = 0.0
        for grad in gradients:
            total_norm += grad.norm().item() ** 2
        return total_norm ** 0.5

class AdaptiveLearningRate:
    """Adaptive learning rate based on meta-learning"""
    
    def __init__(self, initial_lr: float = 0.001, adaptation_rate: float = 0.1, 
                 window_size: int = 100):
        self.initial_lr = initial_lr
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        self.current_lr = initial_lr
        self.performance_history = deque(maxlen=window_size)
        self.lr_history = deque(maxlen=window_size)
        
    def update_learning_rate(self, performance_metrics: Dict[str, float]) -> float:
        """Update learning rate based on performance"""
        
        # Store current performance
        current_performance = performance_metrics.get('success_rate', 0.0)
        self.performance_history.append(current_performance)
        self.lr_history.append(self.current_lr)
        
        # Adapt learning rate if we have enough history
        if len(self.performance_history) >= 10:
            # Compute performance trend
            recent_performance = np.mean(list(self.performance_history)[-5:])
            older_performance = np.mean(list(self.performance_history)[-10:-5])
            
            performance_trend = recent_performance - older_performance
            
            # Adjust learning rate based on trend
            if performance_trend > 0.01:  # Improving
                self.current_lr *= (1 + self.adaptation_rate)
            elif performance_trend < -0.01:  # Declining
                self.current_lr *= (1 - self.adaptation_rate)
                
            # Clamp learning rate
            self.current_lr = np.clip(self.current_lr, self.initial_lr * 0.1, self.initial_lr * 10)
            
        return self.current_lr

class GradientEpisodicMemory:
    """Gradient Episodic Memory for continual learning"""
    
    def __init__(self, memory_size: int = 1000, similarity_threshold: float = 0.9):
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.memory = []
        
    def add_gradient(self, gradient: torch.Tensor, task_id: str, performance: float):
        """Add gradient to episodic memory"""
        gradient_entry = {
            'gradient': gradient.clone(),
            'task_id': task_id,
            'performance': performance,
            'timestamp': time.time()
        }
        
        self.memory.append(gradient_entry)
        
        # Maintain memory size
        if len(self.memory) > self.memory_size:
            # Remove oldest entry
            self.memory.pop(0)
            
    def check_interference(self, current_gradient: torch.Tensor) -> bool:
        """Check if current gradient interferes with past learning"""
        
        for memory_entry in self.memory:
            stored_gradient = memory_entry['gradient']
            similarity = self._compute_cosine_similarity(current_gradient, stored_gradient)
            
            if similarity < -self.similarity_threshold:  # Negative similarity = interference
                return True
                
        return False
        
    def project_gradient(self, current_gradient: torch.Tensor) -> torch.Tensor:
        """Project gradient to avoid catastrophic forgetting"""
        
        projected_gradient = current_gradient.clone()
        
        for memory_entry in self.memory:
            stored_gradient = memory_entry['gradient']
            similarity = self._compute_cosine_similarity(current_gradient, stored_gradient)
            
            if similarity < 0:  # Interference detected
                # Project away from interfering direction
                projection = self._project_vector(current_gradient, stored_gradient)
                projected_gradient = projected_gradient - projection
                
        return projected_gradient
        
    def _compute_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        """Compute cosine similarity between gradients"""
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
        
        similarity = torch.cosine_similarity(grad1_flat, grad2_flat, dim=0)
        return similarity.item()
        
    def _project_vector(self, vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Project vector onto direction"""
        vector_flat = vector.flatten()
        direction_flat = direction.flatten()
        
        # Compute projection
        projection_length = torch.dot(vector_flat, direction_flat) / torch.norm(direction_flat)**2
        projection = projection_length * direction_flat
        
        return projection.reshape(vector.shape)

class MetaLearningManager:
    """Main meta-learning manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_config = config.get('meta_learning', {})
        self.is_enabled = self.meta_config.get('enabled', False)
        
        # Meta-learning parameters
        self.adaptation_steps = self.meta_config.get('adaptation_steps', 5)
        self.meta_lr = self.meta_config.get('meta_lr', 0.01)
        self.inner_lr = self.meta_config.get('inner_lr', 0.1)
        
        # Initialize components
        self.learning_buffer = MetaLearningBuffer()
        self.adaptive_lr = AdaptiveLearningRate(initial_lr=self.meta_lr)
        self.episodic_memory = GradientEpisodicMemory()
        
        # Task management
        self.current_tasks = []
        self.completed_tasks = []
        self.task_performance = defaultdict(list)
        
        # Meta-learning state
        self.meta_optimizer = None
        self.meta_model = None
        self.adaptation_history = []
        
        logger.info(f"Meta-Learning Manager initialized. Enabled: {self.is_enabled}")
        
    def is_meta_learning_enabled(self) -> bool:
        """Check if meta-learning is enabled"""
        return self.is_enabled
        
    def initialize_meta_model(self, base_model: nn.Module):
        """Initialize meta-learning with base model"""
        self.meta_model = base_model
        self.meta_optimizer = MAMLOptimizer(
            base_model, 
            meta_lr=self.meta_lr,
            inner_lr=self.inner_lr,
            inner_steps=self.adaptation_steps
        )
        
    def create_meta_task(self, task_config: Dict[str, Any]) -> MetaTask:
        """Create a new meta-learning task"""
        task = MetaTask(
            task_id=task_config.get('task_id', f"task_{int(time.time())}"),
            description=task_config.get('description', ''),
            support_episodes=task_config.get('support_episodes', 50),
            query_episodes=task_config.get('query_episodes', 20),
            difficulty=task_config.get('difficulty', 0.5),
            environment_config=task_config.get('environment_config', {}),
            success_threshold=task_config.get('success_threshold', 0.8)
        )
        
        self.current_tasks.append(task)
        return task
        
    def adapt_to_task(self, task: MetaTask, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model to a new task using meta-learning"""
        
        if not self.is_enabled or not self.meta_optimizer:
            return {'success': False, 'error': 'Meta-learning not initialized'}
            
        adaptation_start = time.time()
        
        # Get similar tasks for few-shot learning
        similar_tasks = self.learning_buffer.get_similar_tasks(task)
        
        # Prepare support and query sets
        support_data = training_data.get('support_set', {})
        query_data = training_data.get('query_set', {})
        
        try:
            # Perform meta-adaptation
            adaptation_result = self._perform_adaptation(task, support_data, query_data, similar_tasks)
            
            # Update adaptive learning rate
            performance_metrics = adaptation_result.get('performance_metrics', {})
            new_lr = self.adaptive_lr.update_learning_rate(performance_metrics)
            
            # Store adaptation experience
            adaptation_experience = {
                'task_id': task.task_id,
                'adaptation_time': time.time() - adaptation_start,
                'performance_improvement': adaptation_result.get('performance_improvement', 0),
                'final_performance': adaptation_result.get('final_performance', 0),
                'adaptation_steps': self.adaptation_steps,
                'learning_rate': new_lr
            }
            
            self.adaptation_history.append(adaptation_experience)
            self.learning_buffer.add_experience(task.task_id, adaptation_experience)
            
            return {
                'success': True,
                'adaptation_result': adaptation_result,
                'new_learning_rate': new_lr,
                'adaptation_time': adaptation_experience['adaptation_time']
            }
            
        except Exception as e:
            logger.error(f"Meta-adaptation failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def _perform_adaptation(self, task: MetaTask, support_data: Dict[str, Any], 
                          query_data: Dict[str, Any], similar_tasks: List[str]) -> Dict[str, Any]:
        """Perform the actual meta-adaptation"""
        
        # Initialize performance tracking
        initial_performance = self._evaluate_task_performance(task, query_data)
        
        # Create adaptation batches from similar tasks
        support_tasks = []
        query_tasks = []
        
        # Add current task data
        support_tasks.append(support_data)
        query_tasks.append(query_data)
        
        # Add similar task data for meta-learning
        for similar_task_id in similar_tasks:
            similar_experiences = self.learning_buffer.get_task_experiences(similar_task_id, 5)
            if similar_experiences:
                # Use the most recent successful experience
                best_exp = max(similar_experiences, 
                             key=lambda x: x['performance_metrics'].get('success_rate', 0))
                
                if 'support_data' in best_exp['episode_data']:
                    support_tasks.append(best_exp['episode_data']['support_data'])
                    query_tasks.append(best_exp['episode_data']['query_data'])
                    
        # Perform MAML adaptation
        if len(support_tasks) > 1:
            maml_result = self.meta_optimizer.meta_update(support_tasks, query_tasks)
        else:
            # Fallback to simple adaptation
            maml_result = self._simple_adaptation(task, support_data, query_data)
            
        # Evaluate final performance
        final_performance = self._evaluate_task_performance(task, query_data)
        
        return {
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'performance_improvement': final_performance - initial_performance,
            'meta_loss': maml_result.get('meta_loss', 0),
            'adaptation_steps': self.adaptation_steps,
            'performance_metrics': {
                'success_rate': final_performance,
                'adaptation_efficiency': final_performance / max(1, self.adaptation_steps)
            }
        }
        
    def _simple_adaptation(self, task: MetaTask, support_data: Dict[str, Any], 
                          query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple adaptation when meta-learning data is insufficient"""
        
        # Basic gradient descent adaptation
        if self.meta_model:
            optimizer = optim.SGD(self.meta_model.parameters(), lr=self.inner_lr)
            
            for step in range(self.adaptation_steps):
                loss = self._compute_task_loss(self.meta_model, support_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        return {'meta_loss': 0.1, 'adaptation_method': 'simple'}
        
    def _evaluate_task_performance(self, task: MetaTask, evaluation_data: Dict[str, Any]) -> float:
        """Evaluate performance on task"""
        # Placeholder evaluation - would implement actual task-specific evaluation
        return np.random.uniform(0.5, 0.9)  # Mock performance score
        
    def _compute_task_loss(self, model: nn.Module, task_data: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for task-specific data"""
        # Placeholder loss computation
        return torch.tensor(0.5, requires_grad=True)
        
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights from meta-learning experience"""
        
        if not self.adaptation_history:
            return {'insights': [], 'recommendations': []}
            
        # Analyze adaptation patterns
        adaptation_times = [exp['adaptation_time'] for exp in self.adaptation_history]
        performance_improvements = [exp['performance_improvement'] for exp in self.adaptation_history]
        
        insights = {
            'total_adaptations': len(self.adaptation_history),
            'average_adaptation_time': np.mean(adaptation_times),
            'average_performance_improvement': np.mean(performance_improvements),
            'best_performance_improvement': max(performance_improvements) if performance_improvements else 0,
            'adaptation_success_rate': sum(1 for imp in performance_improvements if imp > 0) / len(performance_improvements),
            'learning_efficiency_trend': self._compute_efficiency_trend(),
        }
        
        recommendations = self._generate_meta_learning_recommendations(insights)
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'adaptation_history': self.adaptation_history[-10:],  # Recent history
            'current_learning_rate': self.adaptive_lr.current_lr
        }
        
    def _compute_efficiency_trend(self) -> List[float]:
        """Compute learning efficiency trend over time"""
        
        if len(self.adaptation_history) < 5:
            return []
            
        # Compute efficiency as performance improvement per adaptation step
        efficiencies = []
        for exp in self.adaptation_history:
            efficiency = exp['performance_improvement'] / max(1, exp['adaptation_steps'])
            efficiencies.append(efficiency)
            
        return efficiencies
        
    def _generate_meta_learning_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on meta-learning insights"""
        
        recommendations = []
        
        # Adaptation time recommendations
        if insights['average_adaptation_time'] > 5.0:
            recommendations.append("Consider reducing adaptation steps to improve efficiency")
            
        # Performance improvement recommendations
        if insights['average_performance_improvement'] < 0.1:
            recommendations.append("Increase meta-learning rate or provide more diverse training tasks")
            
        # Success rate recommendations
        if insights['adaptation_success_rate'] < 0.7:
            recommendations.append("Review task similarity metrics for better meta-task selection")
            
        # Learning rate recommendations
        current_lr = self.adaptive_lr.current_lr
        if current_lr < self.meta_lr * 0.5:
            recommendations.append("Learning rate has decreased significantly - consider task difficulty review")
        elif current_lr > self.meta_lr * 2.0:
            recommendations.append("High learning rate detected - monitor for overfitting")
            
        return recommendations
        
    def export_meta_learning_data(self) -> Dict[str, Any]:
        """Export meta-learning data for analysis"""
        
        return {
            'config': self.meta_config,
            'adaptation_history': self.adaptation_history,
            'task_performance': dict(self.task_performance),
            'current_learning_rate': self.adaptive_lr.current_lr,
            'insights': self.get_meta_learning_insights(),
            'buffer_size': len(self.learning_buffer.experiences),
            'episodic_memory_size': len(self.episodic_memory.memory)
        }

# Global meta-learning manager instance
_meta_manager: Optional[MetaLearningManager] = None

def get_meta_learning_manager() -> MetaLearningManager:
    """Get the global meta-learning manager"""
    global _meta_manager
    if _meta_manager is None:
        _meta_manager = MetaLearningManager({})
    return _meta_manager

def initialize_meta_learning_manager(config: Dict[str, Any]) -> MetaLearningManager:
    """Initialize meta-learning manager with configuration"""
    global _meta_manager
    _meta_manager = MetaLearningManager(config)
    return _meta_manager

async def test_meta_learning_system():
    """Test meta-learning system"""
    
    config = {
        'meta_learning': {
            'enabled': True,
            'adaptation_steps': 5,
            'meta_lr': 0.01,
            'inner_lr': 0.1
        }
    }
    
    manager = initialize_meta_learning_manager(config)
    
    # Create a mock model
    mock_model = nn.Linear(10, 5)
    manager.initialize_meta_model(mock_model)
    
    # Create and adapt to a task
    task_config = {
        'task_id': 'test_coordination',
        'description': 'Test coordination task',
        'support_episodes': 20,
        'query_episodes': 10,
        'difficulty': 0.6
    }
    
    task = manager.create_meta_task(task_config)
    
    # Mock training data
    training_data = {
        'support_set': {'episodes': 20, 'performance': 0.6},
        'query_set': {'episodes': 10, 'performance': 0.7}
    }
    
    result = manager.adapt_to_task(task, training_data)
    print(f"Meta-adaptation result: {result['success']}")
    
    # Get insights
    insights = manager.get_meta_learning_insights()
    print(f"Meta-learning insights: {len(insights['recommendations'])} recommendations")

if __name__ == "__main__":
    asyncio.run(test_meta_learning_system())