"""
Breakthrough Detection System
Identifies and analyzes breakthrough events in agent behavior
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from .shared_memory import SharedMemorySystem, VectorType, get_shared_memory

logger = logging.getLogger(__name__)

class BreakthroughType(Enum):
    PATTERN_RECOGNITION = "pattern_recognition"
    COORDINATION_IMPROVEMENT = "coordination_improvement"
    EFFICIENCY_GAIN = "efficiency_gain"
    NOVEL_STRATEGY = "novel_strategy"
    COMMUNICATION_PROTOCOL = "communication_protocol"
    MEMORY_OPTIMIZATION = "memory_optimization"
    SPATIAL_AWARENESS = "spatial_awareness"

@dataclass
class BreakthroughEvent:
    """Represents a detected breakthrough event"""
    event_id: str
    agent_id: str
    breakthrough_type: BreakthroughType
    confidence: float
    description: str
    context_vector: torch.Tensor
    timestamp: float
    coordinates: Tuple[int, int, int]
    related_agents: List[str]
    impact_score: float
    validation_status: str = "pending"  # pending, validated, rejected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'agent_id': self.agent_id,
            'breakthrough_type': self.breakthrough_type.value,
            'confidence': self.confidence,
            'description': self.description,
            'timestamp': self.timestamp,
            'coordinates': self.coordinates,
            'related_agents': self.related_agents,
            'impact_score': self.impact_score,
            'validation_status': self.validation_status
        }

class BreakthroughDetectionNetwork(nn.Module):
    """Neural network for detecting breakthrough patterns"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Breakthrough type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, len(BreakthroughType)),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Impact score predictor
        self.impact_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for breakthrough detection
        
        Returns:
            type_probs: Breakthrough type probabilities
            confidence: Confidence score
            impact: Impact score
        """
        features = self.feature_extractor(x)
        
        type_probs = self.type_classifier(features)
        confidence = self.confidence_estimator(features)
        impact = self.impact_predictor(features)
        
        return type_probs, confidence, impact

class BreakthroughDetector:
    """Main breakthrough detection system"""
    
    def __init__(self, vector_dim: int = 256, detection_threshold: float = 0.7):
        self.vector_dim = vector_dim
        self.detection_threshold = detection_threshold
        
        # Neural network for detection
        self.detection_network = BreakthroughDetectionNetwork(vector_dim)
        
        # Memory system reference
        self.shared_memory = get_shared_memory()
        
        # Breakthrough tracking
        self.breakthrough_history: List[BreakthroughEvent] = []
        self.agent_baselines: Dict[str, torch.Tensor] = {}
        self.recent_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Statistical analysis
        self.breakthrough_stats = {
            'total_detected': 0,
            'by_type': defaultdict(int),
            'by_agent': defaultdict(int),
            'validation_rate': 0.0,
            'average_confidence': 0.0,
            'average_impact': 0.0
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks for breakthrough events
        self.breakthrough_callbacks: List[callable] = []
        
        logger.info("BreakthroughDetector initialized")
    
    def add_breakthrough_callback(self, callback: callable):
        """Add callback for breakthrough events"""
        self.breakthrough_callbacks.append(callback)
    
    def remove_breakthrough_callback(self, callback: callable):
        """Remove callback for breakthrough events"""
        if callback in self.breakthrough_callbacks:
            self.breakthrough_callbacks.remove(callback)
    
    def start_monitoring(self):
        """Start real-time breakthrough monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Breakthrough monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time breakthrough monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Breakthrough monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get recent memory vectors
                memory_stats = self.shared_memory.get_memory_statistics()
                
                # Check for breakthrough patterns
                self._check_memory_patterns()
                
                # Update agent baselines
                self._update_agent_baselines()
                
                time.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def detect_breakthrough(self, agent_id: str, behavior_vector: torch.Tensor,
                          coordinates: Tuple[int, int, int],
                          context: Optional[Dict[str, Any]] = None) -> Optional[BreakthroughEvent]:
        """
        Detect if a behavior vector represents a breakthrough
        
        Args:
            agent_id: ID of the agent
            behavior_vector: Vector representing agent behavior
            coordinates: Agent's position
            context: Additional context information
            
        Returns:
            BreakthroughEvent if breakthrough detected, None otherwise
        """
        
        # Ensure vector is the right shape
        if behavior_vector.dim() == 1:
            behavior_vector = behavior_vector.unsqueeze(0)
        
        # Run through detection network
        type_probs, confidence, impact = self.detection_network(behavior_vector)
        
        confidence_score = confidence.item()
        impact_score = impact.item()
        
        # Check if confidence exceeds threshold
        if confidence_score < self.detection_threshold:
            return None
        
        # Determine breakthrough type
        type_idx = torch.argmax(type_probs).item()
        breakthrough_type = list(BreakthroughType)[type_idx]
        
        # Check against agent baseline
        if agent_id in self.agent_baselines:
            baseline = self.agent_baselines[agent_id]
            novelty_score = self._calculate_novelty(behavior_vector.squeeze(), baseline)
            
            # Adjust confidence based on novelty
            confidence_score = min(1.0, confidence_score * (1.0 + novelty_score))
        
        # Create breakthrough event
        event_id = f"breakthrough_{agent_id}_{int(time.time() * 1000)}"
        
        breakthrough_event = BreakthroughEvent(
            event_id=event_id,
            agent_id=agent_id,
            breakthrough_type=breakthrough_type,
            confidence=confidence_score,
            description=self._generate_description(breakthrough_type, context),
            context_vector=behavior_vector.squeeze().clone(),
            timestamp=time.time(),
            coordinates=coordinates,
            related_agents=self._find_related_agents(agent_id, coordinates),
            impact_score=impact_score
        )
        
        # Store in memory
        self.breakthrough_history.append(breakthrough_event)
        
        # Update statistics
        self._update_statistics(breakthrough_event)
        
        # Store in shared memory
        self.shared_memory.store_vector(
            vector_id=event_id,
            content=behavior_vector.squeeze(),
            vector_type=VectorType.BREAKTHROUGH,
            coordinates=coordinates,
            importance=confidence_score,
            tags=[breakthrough_type.value, agent_id]
        )
        
        # Trigger callbacks
        for callback in self.breakthrough_callbacks:
            try:
                callback(breakthrough_event)
            except Exception as e:
                logger.error(f"Error in breakthrough callback: {e}")
        
        logger.info(f"Breakthrough detected: {agent_id} - {breakthrough_type.value} (conf: {confidence_score:.3f})")
        
        return breakthrough_event
    
    def _calculate_novelty(self, current_vector: torch.Tensor, baseline: torch.Tensor) -> float:
        """Calculate novelty score compared to baseline"""
        
        # Cosine similarity
        similarity = F.cosine_similarity(current_vector, baseline, dim=0)
        
        # Novelty is inverse of similarity
        novelty = 1.0 - similarity.item()
        
        return max(0.0, novelty)
    
    def _generate_description(self, breakthrough_type: BreakthroughType, 
                            context: Optional[Dict[str, Any]]) -> str:
        """Generate human-readable description of breakthrough"""
        
        descriptions = {
            BreakthroughType.PATTERN_RECOGNITION: "Agent discovered a new pattern in the environment",
            BreakthroughType.COORDINATION_IMPROVEMENT: "Agent improved coordination with neighbors",
            BreakthroughType.EFFICIENCY_GAIN: "Agent achieved significant efficiency improvement",
            BreakthroughType.NOVEL_STRATEGY: "Agent developed a novel problem-solving strategy",
            BreakthroughType.COMMUNICATION_PROTOCOL: "Agent evolved new communication protocol",
            BreakthroughType.MEMORY_OPTIMIZATION: "Agent optimized memory usage patterns",
            BreakthroughType.SPATIAL_AWARENESS: "Agent enhanced spatial awareness capabilities"
        }
        
        base_description = descriptions.get(breakthrough_type, "Unknown breakthrough type")
        
        if context:
            # Add context-specific details
            if 'performance_gain' in context:
                base_description += f" (performance gain: {context['performance_gain']:.2f})"
            if 'efficiency_improvement' in context:
                base_description += f" (efficiency: +{context['efficiency_improvement']:.1f}%)"
        
        return base_description
    
    def _find_related_agents(self, agent_id: str, coordinates: Tuple[int, int, int]) -> List[str]:
        """Find agents related to the breakthrough"""
        
        # Find agents in spatial proximity
        nearby_vectors = self.shared_memory.search_vectors_by_coordinates(coordinates, radius=2.0)
        
        related_agents = set()
        for vector_id in nearby_vectors:
            vector = self.shared_memory.retrieve_vector(vector_id)
            if vector is not None:
                # Extract agent IDs from vector tags or metadata
                # This is a simplified approach - in practice, you'd have better metadata
                if 'agent_' in vector_id:
                    related_agent_id = vector_id.split('_')[1]
                    if related_agent_id != agent_id:
                        related_agents.add(related_agent_id)
        
        return list(related_agents)
    
    def _update_statistics(self, breakthrough_event: BreakthroughEvent):
        """Update breakthrough statistics"""
        
        self.breakthrough_stats['total_detected'] += 1
        self.breakthrough_stats['by_type'][breakthrough_event.breakthrough_type.value] += 1
        self.breakthrough_stats['by_agent'][breakthrough_event.agent_id] += 1
        
        # Update averages
        total = self.breakthrough_stats['total_detected']
        self.breakthrough_stats['average_confidence'] = (
            self.breakthrough_stats['average_confidence'] * (total - 1) + 
            breakthrough_event.confidence
        ) / total
        
        self.breakthrough_stats['average_impact'] = (
            self.breakthrough_stats['average_impact'] * (total - 1) + 
            breakthrough_event.impact_score
        ) / total
    
    def _check_memory_patterns(self):
        """Check shared memory for breakthrough patterns"""
        
        # Get recent breakthrough vectors
        breakthrough_vectors = self.shared_memory.search_vectors_by_type(VectorType.BREAKTHROUGH)
        
        # Analyze patterns in recent breakthroughs
        if len(breakthrough_vectors) > 5:
            # Look for clustering of breakthrough types
            recent_types = []
            for vector_id in breakthrough_vectors[-10:]:  # Last 10 breakthroughs
                vector = self.shared_memory.retrieve_vector(vector_id)
                if vector is not None:
                    # Extract type from vector metadata (simplified)
                    pass
    
    def _update_agent_baselines(self):
        """Update baseline behavior vectors for agents"""
        
        # Get recent context vectors for each agent
        context_vectors = self.shared_memory.search_vectors_by_type(VectorType.CONTEXT)
        
        agent_vectors = defaultdict(list)
        for vector_id in context_vectors:
            vector = self.shared_memory.retrieve_vector(vector_id)
            if vector is not None:
                # Extract agent ID from vector ID (simplified)
                if 'agent_' in vector_id:
                    agent_id = vector_id.split('_')[1]
                    agent_vectors[agent_id].append(vector)
        
        # Update baselines
        for agent_id, vectors in agent_vectors.items():
            if len(vectors) >= 3:  # Need minimum vectors for stable baseline
                baseline = torch.stack(vectors[-5:]).mean(dim=0)  # Use last 5 vectors
                self.agent_baselines[agent_id] = baseline
    
    def validate_breakthrough(self, event_id: str, is_valid: bool, 
                            feedback: Optional[str] = None) -> bool:
        """Validate a breakthrough event"""
        
        for event in self.breakthrough_history:
            if event.event_id == event_id:
                event.validation_status = "validated" if is_valid else "rejected"
                
                # Update validation rate
                total_validated = sum(1 for e in self.breakthrough_history 
                                    if e.validation_status != "pending")
                if total_validated > 0:
                    valid_count = sum(1 for e in self.breakthrough_history 
                                    if e.validation_status == "validated")
                    self.breakthrough_stats['validation_rate'] = valid_count / total_validated
                
                logger.info(f"Breakthrough {event_id} {'validated' if is_valid else 'rejected'}")
                return True
        
        return False
    
    def get_breakthrough_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of breakthrough events"""
        
        events = self.breakthrough_history
        if time_window:
            cutoff_time = time.time() - time_window
            events = [e for e in events if e.timestamp >= cutoff_time]
        
        if not events:
            return {
                'total_breakthroughs': 0,
                'by_type': {},
                'by_agent': {},
                'average_confidence': 0.0,
                'average_impact': 0.0,
                'recent_events': []
            }
        
        # Aggregate by type
        by_type = defaultdict(int)
        for event in events:
            by_type[event.breakthrough_type.value] += 1
        
        # Aggregate by agent
        by_agent = defaultdict(int)
        for event in events:
            by_agent[event.agent_id] += 1
        
        # Calculate averages
        avg_confidence = np.mean([e.confidence for e in events])
        avg_impact = np.mean([e.impact_score for e in events])
        
        return {
            'total_breakthroughs': len(events),
            'by_type': dict(by_type),
            'by_agent': dict(by_agent),
            'average_confidence': avg_confidence,
            'average_impact': avg_impact,
            'recent_events': [e.to_dict() for e in events[-10:]]
        }
    
    def get_agent_breakthrough_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get breakthrough profile for specific agent"""
        
        agent_events = [e for e in self.breakthrough_history if e.agent_id == agent_id]
        
        if not agent_events:
            return {
                'agent_id': agent_id,
                'total_breakthroughs': 0,
                'breakthrough_types': {},
                'average_confidence': 0.0,
                'trend': 'stable'
            }
        
        # Calculate breakthrough frequency over time
        recent_events = [e for e in agent_events if e.timestamp >= time.time() - 3600]  # Last hour
        
        # Determine trend
        if len(recent_events) > len(agent_events) * 0.5:
            trend = 'increasing'
        elif len(recent_events) < len(agent_events) * 0.2:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Aggregate by type
        breakthrough_types = defaultdict(int)
        for event in agent_events:
            breakthrough_types[event.breakthrough_type.value] += 1
        
        return {
            'agent_id': agent_id,
            'total_breakthroughs': len(agent_events),
            'breakthrough_types': dict(breakthrough_types),
            'average_confidence': np.mean([e.confidence for e in agent_events]),
            'recent_frequency': len(recent_events),
            'trend': trend,
            'latest_breakthrough': agent_events[-1].to_dict() if agent_events else None
        }
    
    def export_breakthrough_data(self) -> Dict[str, Any]:
        """Export all breakthrough data for analysis"""
        
        return {
            'breakthrough_history': [e.to_dict() for e in self.breakthrough_history],
            'statistics': dict(self.breakthrough_stats),
            'agent_baselines': {
                agent_id: baseline.tolist() 
                for agent_id, baseline in self.agent_baselines.items()
            },
            'detection_threshold': self.detection_threshold,
            'monitoring_active': self.monitoring_active
        }
    
    def shutdown(self):
        """Gracefully shutdown the breakthrough detector"""
        
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
        
        logger.info("BreakthroughDetector shutdown complete")

# Global breakthrough detector instance
breakthrough_detector = BreakthroughDetector()

def get_breakthrough_detector() -> BreakthroughDetector:
    """Get the global breakthrough detector instance"""
    return breakthrough_detector
