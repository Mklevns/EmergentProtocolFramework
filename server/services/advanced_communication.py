"""
Advanced Communication & Memory Enhancement Module
Implements sophisticated communication mechanisms inspired by recent MARL research
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import uuid
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import math

from .shared_memory import SharedMemorySystem, VectorType, MemoryVector
from .communication_protocol import Message, MessageType, MessagePriority

logger = logging.getLogger(__name__)

class MessageEmbeddingType(Enum):
    """Enhanced message embedding types for complex communication"""
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    EMOTIONAL = "emotional"
    STRATEGIC = "strategic"

class AttentionMechanism(Enum):
    """Different attention mechanisms for message processing"""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"
    SPARSE_ATTENTION = "sparse_attention"

@dataclass
class MessageEmbedding:
    """Advanced message embedding with multiple vector representations"""
    embedding_id: str
    semantic_vector: torch.Tensor
    contextual_vector: torch.Tensor
    temporal_vector: torch.Tensor
    spatial_vector: torch.Tensor
    attention_weights: torch.Tensor
    confidence_score: float
    embedding_type: MessageEmbeddingType
    creation_time: float = field(default_factory=time.time)

@dataclass
class CommunicationContext:
    """Rich communication context for enhanced message understanding"""
    context_id: str
    participants: Set[str]
    conversation_history: List[str]
    topic_vectors: Dict[str, torch.Tensor]
    urgency_level: float
    coherence_score: float
    active_strategies: List[str]
    environmental_factors: Dict[str, Any]

class AdaptiveBandwidthManager:
    """Manages communication bandwidth dynamically based on importance and network state"""
    
    def __init__(self, max_bandwidth: int = 10000, adaptation_rate: float = 0.1):
        self.max_bandwidth = max_bandwidth
        self.adaptation_rate = adaptation_rate
        self.current_usage = 0.0
        self.agent_allocations: Dict[str, float] = {}
        self.priority_weights = {
            MessagePriority.URGENT: 1.0,
            MessagePriority.HIGH: 0.7,
            MessagePriority.NORMAL: 0.4,
            MessagePriority.LOW: 0.1
        }
        self.congestion_history: deque = deque(maxlen=100)
        self.adaptation_history: deque = deque(maxlen=50)
        
    def allocate_bandwidth(self, agent_id: str, message_size: int, priority: MessagePriority, 
                          context: Optional[CommunicationContext] = None) -> Tuple[bool, float]:
        """Dynamically allocate bandwidth based on message importance and network state"""
        
        # Calculate base allocation
        base_allocation = message_size * self.priority_weights[priority]
        
        # Apply context-based adjustments
        context_multiplier = 1.0
        if context:
            # Increase allocation for high-urgency contexts
            context_multiplier *= (1.0 + context.urgency_level * 0.5)
            # Increase for high-coherence conversations
            context_multiplier *= (1.0 + context.coherence_score * 0.3)
        
        adjusted_allocation = base_allocation * context_multiplier
        
        # Check if allocation is possible
        if self.current_usage + adjusted_allocation > self.max_bandwidth:
            # Try adaptive compression
            compression_ratio = self._calculate_compression_ratio(adjusted_allocation)
            adjusted_allocation *= compression_ratio
            
            if self.current_usage + adjusted_allocation > self.max_bandwidth:
                return False, 0.0
        
        # Update allocations
        self.current_usage += adjusted_allocation
        if agent_id not in self.agent_allocations:
            self.agent_allocations[agent_id] = 0.0
        self.agent_allocations[agent_id] += adjusted_allocation
        
        # Record congestion metrics
        congestion_level = self.current_usage / self.max_bandwidth
        self.congestion_history.append(congestion_level)
        
        return True, adjusted_allocation
    
    def _calculate_compression_ratio(self, required_allocation: float) -> float:
        """Calculate adaptive compression ratio based on network congestion"""
        congestion_level = self.current_usage / self.max_bandwidth
        
        if congestion_level < 0.7:
            return 1.0  # No compression needed
        elif congestion_level < 0.85:
            return 0.8  # Light compression
        elif congestion_level < 0.95:
            return 0.6  # Medium compression
        else:
            return 0.4  # Heavy compression
    
    def release_bandwidth(self, agent_id: str, allocation: float):
        """Release allocated bandwidth"""
        self.current_usage = max(0.0, self.current_usage - allocation)
        if agent_id in self.agent_allocations:
            self.agent_allocations[agent_id] = max(0.0, self.agent_allocations[agent_id] - allocation)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network utilization statistics"""
        avg_congestion = np.mean(list(self.congestion_history)) if self.congestion_history else 0.0
        peak_congestion = max(self.congestion_history) if self.congestion_history else 0.0
        
        return {
            'current_usage': self.current_usage,
            'max_bandwidth': self.max_bandwidth,
            'utilization_percentage': (self.current_usage / self.max_bandwidth) * 100,
            'average_congestion': avg_congestion,
            'peak_congestion': peak_congestion,
            'active_agents': len(self.agent_allocations),
            'agent_allocations': dict(self.agent_allocations)
        }

class MessageEmbeddingNetwork(nn.Module):
    """Advanced neural network for generating sophisticated message embeddings"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-modal embedding networks
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.contextual_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),  # +64 for temporal features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for spatial coordinates
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Cross-modal fusion
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Confidence and importance estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.importance_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, content_vector: torch.Tensor, temporal_features: torch.Tensor,
                spatial_coords: torch.Tensor, context_vectors: Optional[torch.Tensor] = None) -> MessageEmbedding:
        """Generate sophisticated message embedding"""
        
        # Generate multi-modal embeddings
        semantic_emb = self.semantic_encoder(content_vector)
        contextual_emb = self.contextual_encoder(content_vector)
        
        # Add temporal information
        temporal_input = torch.cat([content_vector, temporal_features], dim=-1)
        temporal_emb = self.temporal_encoder(temporal_input)
        
        # Add spatial information
        spatial_input = torch.cat([content_vector, spatial_coords], dim=-1)
        spatial_emb = self.spatial_encoder(spatial_input)
        
        # Stack embeddings for attention
        embeddings = torch.stack([semantic_emb, contextual_emb, temporal_emb, spatial_emb], dim=1)
        
        # Apply cross-modal attention
        attended_embs, attention_weights = self.attention(embeddings, embeddings, embeddings)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            attended_embs = layer(attended_embs)
        
        # Extract individual embeddings
        semantic_final = attended_embs[:, 0, :]
        contextual_final = attended_embs[:, 1, :]
        temporal_final = attended_embs[:, 2, :]
        spatial_final = attended_embs[:, 3, :]
        
        # Concatenate for confidence estimation
        fused_representation = torch.cat([semantic_final, contextual_final, temporal_final, spatial_final], dim=-1)
        confidence_score = self.confidence_estimator(fused_representation).item()
        
        return MessageEmbedding(
            embedding_id=str(uuid.uuid4()),
            semantic_vector=semantic_final.squeeze(0),
            contextual_vector=contextual_final.squeeze(0),
            temporal_vector=temporal_final.squeeze(0),
            spatial_vector=spatial_final.squeeze(0),
            attention_weights=attention_weights.squeeze(0),
            confidence_score=confidence_score,
            embedding_type=MessageEmbeddingType.SEMANTIC
        )

class AdvancedMemoryIndexing:
    """Sophisticated memory indexing system with multiple retrieval strategies"""
    
    def __init__(self, vector_dim: int = 256, max_clusters: int = 50):
        self.vector_dim = vector_dim
        self.max_clusters = max_clusters
        
        # Multiple indexing structures
        self.semantic_index: Dict[str, List[str]] = defaultdict(list)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.spatial_index: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        self.importance_index: Dict[str, List[str]] = defaultdict(list)
        
        # Advanced indexing structures
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)
        self.access_pattern_clusters: Dict[str, List[str]] = defaultdict(list)
        self.concept_hierarchy: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Dynamic clustering
        self.cluster_centers: torch.Tensor = torch.randn(max_clusters, vector_dim)
        self.cluster_assignments: Dict[str, int] = {}
        self.cluster_update_threshold = 0.1
        
        # Retrieval statistics
        self.retrieval_stats = {
            'semantic_queries': 0,
            'temporal_queries': 0,
            'spatial_queries': 0,
            'associative_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Query cache
        self.query_cache: Dict[str, Tuple[List[str], float]] = {}
        self.cache_max_size = 1000
        self.cache_ttl = 300  # 5 minutes
    
    def index_memory_vector(self, vector_id: str, vector: MemoryVector):
        """Add vector to multiple indexing structures"""
        
        # Semantic indexing by type and tags
        self.semantic_index[vector.vector_type.value].append(vector_id)
        for tag in vector.tags:
            self.semantic_index[tag].append(vector_id)
        
        # Temporal indexing by creation time
        time_bucket = str(int(vector.created_at // 3600))  # Hour buckets
        self.temporal_index[time_bucket].append(vector_id)
        
        # Spatial indexing
        self.spatial_index[vector.coordinates].append(vector_id)
        
        # Importance-based indexing
        importance_level = self._categorize_importance(vector.importance)
        self.importance_index[importance_level].append(vector_id)
        
        # Dynamic clustering
        self._update_clusters(vector_id, vector.content)
        
        # Update concept hierarchy
        self._update_concept_hierarchy(vector_id, vector)
    
    def query_by_semantic_similarity(self, query_vector: torch.Tensor, 
                                   top_k: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Retrieve vectors by semantic similarity with caching"""
        
        # Generate cache key
        cache_key = f"semantic_{hash(query_vector.data.tobytes())}_{top_k}_{threshold}"
        
        # Check cache
        if cache_key in self.query_cache:
            cached_results, cache_time = self.query_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                self.retrieval_stats['cache_hits'] += 1
                return cached_results
        
        self.retrieval_stats['cache_misses'] += 1
        self.retrieval_stats['semantic_queries'] += 1
        
        # Perform similarity search (simplified implementation)
        results = []
        # In practice, this would use efficient vector similarity search
        # For now, we'll return placeholder results
        
        # Cache results
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest cache entry
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k][1])
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = (results, time.time())
        return results
    
    def query_by_temporal_patterns(self, time_range: Tuple[float, float], 
                                 pattern_type: str = "sequential") -> List[str]:
        """Retrieve vectors based on temporal access patterns"""
        
        self.retrieval_stats['temporal_queries'] += 1
        
        start_bucket = str(int(time_range[0] // 3600))
        end_bucket = str(int(time_range[1] // 3600))
        
        results = []
        for bucket in self.temporal_index:
            if start_bucket <= bucket <= end_bucket:
                results.extend(self.temporal_index[bucket])
        
        return results
    
    def query_by_association(self, vector_id: str, max_depth: int = 3) -> List[str]:
        """Retrieve vectors through associative connections"""
        
        self.retrieval_stats['associative_queries'] += 1
        
        visited = set()
        queue = deque([(vector_id, 0)])
        results = []
        
        while queue and len(results) < 50:  # Limit results
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth >= max_depth:
                continue
                
            visited.add(current_id)
            if current_id != vector_id:
                results.append(current_id)
            
            # Add associated vectors
            for associated_id in self.association_graph[current_id]:
                if associated_id not in visited:
                    queue.append((associated_id, depth + 1))
        
        return results
    
    def query_by_spatial_proximity(self, center_coords: Tuple[int, int, int], 
                                 radius: int = 2) -> List[str]:
        """Retrieve vectors within spatial proximity"""
        
        self.retrieval_stats['spatial_queries'] += 1
        
        results = []
        cx, cy, cz = center_coords
        
        for (x, y, z), vector_ids in self.spatial_index.items():
            distance = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            if distance <= radius:
                results.extend(vector_ids)
        
        return results
    
    def _categorize_importance(self, importance: float) -> str:
        """Categorize importance level for indexing"""
        if importance >= 0.8:
            return "critical"
        elif importance >= 0.6:
            return "high"
        elif importance >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _update_clusters(self, vector_id: str, vector: torch.Tensor):
        """Update dynamic clustering structure"""
        
        # Find nearest cluster
        distances = torch.norm(self.cluster_centers - vector, dim=1)
        nearest_cluster = torch.argmin(distances).item()
        min_distance = distances[nearest_cluster].item()
        
        # Assign to cluster or create new one
        if min_distance < self.cluster_update_threshold:
            self.cluster_assignments[vector_id] = nearest_cluster
            # Update cluster center (moving average)
            alpha = 0.1
            self.cluster_centers[nearest_cluster] = (
                (1 - alpha) * self.cluster_centers[nearest_cluster] + alpha * vector
            )
        else:
            # Create new cluster if we haven't reached max
            if len(set(self.cluster_assignments.values())) < self.max_clusters:
                new_cluster_id = max(self.cluster_assignments.values(), default=-1) + 1
                self.cluster_assignments[vector_id] = new_cluster_id
                if new_cluster_id < self.max_clusters:
                    self.cluster_centers[new_cluster_id] = vector
    
    def _update_concept_hierarchy(self, vector_id: str, vector: MemoryVector):
        """Update hierarchical concept relationships"""
        
        # Build concept relationships based on tags and type
        concepts = [vector.vector_type.value] + vector.tags
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i != j:
                    # Strengthen relationship
                    if concept2 not in self.concept_hierarchy[concept1]:
                        self.concept_hierarchy[concept1][concept2] = 0.0
                    self.concept_hierarchy[concept1][concept2] += 0.1
                    
                    # Normalize to keep values in reasonable range
                    if self.concept_hierarchy[concept1][concept2] > 1.0:
                        self.concept_hierarchy[concept1][concept2] = 1.0
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get comprehensive indexing statistics"""
        
        return {
            'retrieval_stats': dict(self.retrieval_stats),
            'index_sizes': {
                'semantic': len(self.semantic_index),
                'temporal': len(self.temporal_index),
                'spatial': len(self.spatial_index),
                'importance': len(self.importance_index)
            },
            'cluster_info': {
                'total_clusters': len(set(self.cluster_assignments.values())),
                'max_clusters': self.max_clusters,
                'cluster_distribution': len(self.cluster_assignments)
            },
            'cache_stats': {
                'cache_size': len(self.query_cache),
                'cache_max_size': self.cache_max_size,
                'hit_rate': (self.retrieval_stats['cache_hits'] / 
                           (self.retrieval_stats['cache_hits'] + self.retrieval_stats['cache_misses'] + 1))
            },
            'concept_hierarchy_size': len(self.concept_hierarchy)
        }

class EnhancedCommunicationProtocol:
    """Enhanced communication protocol with advanced embedding and bandwidth management"""
    
    def __init__(self, embedding_dim: int = 512, max_bandwidth: int = 10000):
        self.embedding_network = MessageEmbeddingNetwork(embedding_dim)
        self.bandwidth_manager = AdaptiveBandwidthManager(max_bandwidth)
        self.memory_indexing = AdvancedMemoryIndexing()
        
        # Communication contexts
        self.active_contexts: Dict[str, CommunicationContext] = {}
        self.context_history: deque = deque(maxlen=1000)
        
        # Enhanced message processing
        self.message_embeddings: Dict[str, MessageEmbedding] = {}
        self.conversation_graphs: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Performance metrics
        self.metrics = {
            'total_messages_processed': 0,
            'average_embedding_time': 0.0,
            'bandwidth_efficiency': 0.0,
            'context_coherence_avg': 0.0,
            'retrieval_accuracy': 0.0
        }
        
        logger.info("Enhanced Communication Protocol initialized")
    
    def create_communication_context(self, participants: Set[str], 
                                   topic: str, urgency: float = 0.5) -> str:
        """Create a new communication context for enhanced message understanding"""
        
        context_id = str(uuid.uuid4())
        context = CommunicationContext(
            context_id=context_id,
            participants=participants,
            conversation_history=[],
            topic_vectors={topic: torch.randn(256)},  # Simplified topic vector
            urgency_level=urgency,
            coherence_score=1.0,
            active_strategies=[],
            environmental_factors={}
        )
        
        self.active_contexts[context_id] = context
        return context_id
    
    def process_enhanced_message(self, message: Message, context_id: Optional[str] = None) -> MessageEmbedding:
        """Process message with advanced embedding and context awareness"""
        
        start_time = time.time()
        
        # Prepare input vectors
        content_vector = self._message_to_vector(message)
        temporal_features = self._extract_temporal_features(message)
        spatial_coords = self._extract_spatial_coords(message)
        
        # Generate enhanced embedding
        embedding = self.embedding_network(
            content_vector.unsqueeze(0),
            temporal_features.unsqueeze(0),
            spatial_coords.unsqueeze(0)
        )
        
        # Store embedding
        self.message_embeddings[message.message_id] = embedding
        
        # Update context if provided
        if context_id and context_id in self.active_contexts:
            context = self.active_contexts[context_id]
            context.conversation_history.append(message.message_id)
            
            # Update coherence score based on embedding similarity
            if len(context.conversation_history) > 1:
                prev_msg_id = context.conversation_history[-2]
                if prev_msg_id in self.message_embeddings:
                    prev_embedding = self.message_embeddings[prev_msg_id]
                    similarity = F.cosine_similarity(
                        embedding.semantic_vector.unsqueeze(0),
                        prev_embedding.semantic_vector.unsqueeze(0)
                    ).item()
                    context.coherence_score = 0.9 * context.coherence_score + 0.1 * similarity
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics['total_messages_processed'] += 1
        self.metrics['average_embedding_time'] = (
            0.9 * self.metrics['average_embedding_time'] + 0.1 * processing_time
        )
        
        return embedding
    
    def allocate_communication_bandwidth(self, message: Message, embedding: MessageEmbedding,
                                       context_id: Optional[str] = None) -> bool:
        """Allocate bandwidth for message transmission with advanced prioritization"""
        
        # Estimate message size based on embedding complexity
        base_size = len(json.dumps(message.content))
        embedding_size = embedding.semantic_vector.numel() * 4  # Assume float32
        total_size = base_size + embedding_size
        
        # Get context for enhanced prioritization
        context = self.active_contexts.get(context_id) if context_id else None
        
        # Allocate bandwidth
        success, allocation = self.bandwidth_manager.allocate_bandwidth(
            message.source_agent, total_size, message.priority, context
        )
        
        if success:
            # Update bandwidth efficiency metric
            theoretical_min_size = base_size
            efficiency = theoretical_min_size / allocation if allocation > 0 else 0
            self.metrics['bandwidth_efficiency'] = (
                0.9 * self.metrics['bandwidth_efficiency'] + 0.1 * efficiency
            )
        
        return success
    
    def query_memory_with_embedding(self, query_embedding: MessageEmbedding, 
                                  query_type: str = "semantic", 
                                  max_results: int = 10) -> List[Tuple[str, float]]:
        """Query memory system using advanced embedding-based retrieval"""
        
        if query_type == "semantic":
            return self.memory_indexing.query_by_semantic_similarity(
                query_embedding.semantic_vector, max_results
            )
        elif query_type == "temporal":
            # Use temporal vector for time-based queries
            current_time = time.time()
            time_range = (current_time - 3600, current_time)  # Last hour
            results = self.memory_indexing.query_by_temporal_patterns(time_range)
            return [(r, 1.0) for r in results[:max_results]]
        elif query_type == "spatial":
            # Extract spatial information from embedding
            # This is simplified - in practice, you'd decode spatial coords from the embedding
            center_coords = (0, 0, 0)  # Placeholder
            results = self.memory_indexing.query_by_spatial_proximity(center_coords)
            return [(r, 1.0) for r in results[:max_results]]
        else:
            return []
    
    def _message_to_vector(self, message: Message) -> torch.Tensor:
        """Convert message content to vector representation"""
        # Simplified: In practice, this would use sophisticated NLP embedding
        content_str = json.dumps(message.content)
        # Create a simple hash-based vector
        hash_val = hash(content_str)
        vector = torch.randn(512)  # Placeholder embedding
        vector[0] = float(hash_val % 1000) / 1000.0  # Inject some content dependency
        return vector
    
    def _extract_temporal_features(self, message: Message) -> torch.Tensor:
        """Extract temporal features from message"""
        current_time = time.time()
        time_since_created = current_time - message.timestamp
        
        # Create temporal feature vector
        features = torch.zeros(64)
        features[0] = message.timestamp / 1e9  # Normalized timestamp
        features[1] = time_since_created / 3600  # Hours since creation
        features[2] = math.sin(2 * math.pi * (message.timestamp % 86400) / 86400)  # Daily cycle
        features[3] = math.cos(2 * math.pi * (message.timestamp % 86400) / 86400)
        
        return features
    
    def _extract_spatial_coords(self, message: Message) -> torch.Tensor:
        """Extract spatial coordinates from message"""
        # In practice, this would extract from agent network or message routing
        coords = torch.zeros(3)
        # Placeholder spatial extraction
        coords[0] = hash(message.source_agent) % 10 / 10.0
        coords[1] = hash(message.target_agent) % 10 / 10.0
        coords[2] = 0.5  # Default z-coordinate
        
        return coords
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive communication and memory metrics"""
        
        base_metrics = dict(self.metrics)
        bandwidth_stats = self.bandwidth_manager.get_network_stats()
        indexing_stats = self.memory_indexing.get_indexing_stats()
        
        context_metrics = {
            'active_contexts': len(self.active_contexts),
            'average_context_coherence': np.mean([
                ctx.coherence_score for ctx in self.active_contexts.values()
            ]) if self.active_contexts else 0.0,
            'average_urgency': np.mean([
                ctx.urgency_level for ctx in self.active_contexts.values()
            ]) if self.active_contexts else 0.0
        }
        
        return {
            'communication_metrics': base_metrics,
            'bandwidth_metrics': bandwidth_stats,
            'indexing_metrics': indexing_stats,
            'context_metrics': context_metrics,
            'total_embeddings': len(self.message_embeddings)
        }


# Global instance for integration with existing system
_enhanced_protocol = None

def get_enhanced_communication_protocol() -> EnhancedCommunicationProtocol:
    """Get global enhanced communication protocol instance"""
    global _enhanced_protocol
    if _enhanced_protocol is None:
        _enhanced_protocol = EnhancedCommunicationProtocol()
    return _enhanced_protocol