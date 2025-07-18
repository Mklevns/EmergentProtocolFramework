"""
Shared Vectorized Memory System
Implements efficient pointer-based communication and memory management
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import uuid
import json
import logging
import math
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VectorType(Enum):
    BREAKTHROUGH = "breakthrough"
    CONTEXT = "context"
    COORDINATION = "coordination"
    MEMORY_TRACE = "memory_trace"
    PATTERN = "pattern"

class MemoryAccessPattern(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"

@dataclass
class MemoryVector:
    """A memory vector with metadata"""
    vector_id: str
    content: torch.Tensor
    vector_type: VectorType
    coordinates: Tuple[int, int, int]
    importance: float
    access_count: int
    created_at: float
    last_accessed: float
    ttl: Optional[float] = None  # Time to live
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Check if memory vector has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

@dataclass
class MemoryPointer:
    """A pointer to a memory vector"""
    pointer_id: str
    vector_id: str
    source_agent: str
    target_agent: str
    message_type: str
    metadata: Dict[str, Any]
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pointer_id': self.pointer_id,
            'vector_id': self.vector_id,
            'source_agent': self.source_agent,
            'target_agent': self.target_agent,
            'message_type': self.message_type,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

class MemoryCompressionUnit(nn.Module):
    """Neural network for memory compression and decompression"""
    
    def __init__(self, input_dim: int, compressed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        
        # Encoder for compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, compressed_dim),
            nn.Tanh()
        )
        
        # Decoder for decompression
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim),
            nn.Tanh()
        )
        
        # Quality assessment network
        self.quality_assessor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input vector"""
        return self.encoder(x)
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress vector"""
        return self.decoder(compressed)
    
    def assess_quality(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Assess reconstruction quality"""
        diff = torch.abs(original - reconstructed)
        return self.quality_assessor(diff)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full compression-decompression cycle"""
        compressed = self.compress(x)
        reconstructed = self.decompress(compressed)
        quality = self.assess_quality(x, reconstructed)
        return compressed, reconstructed, quality

class SharedMemorySystem:
    """Advanced shared memory system with vectorized storage and pointer-based access"""
    
    def __init__(self, vector_dim: int = 256, max_vectors: int = 1000, 
                 compression_ratio: float = 0.5, enable_clustering: bool = True):
        self.vector_dim = vector_dim
        self.max_vectors = max_vectors
        self.compression_ratio = compression_ratio
        self.enable_clustering = enable_clustering
        
        # Core memory storage
        self.memory_vectors: Dict[str, MemoryVector] = {}
        self.memory_pointers: Dict[str, MemoryPointer] = {}
        
        # Enhanced indexing structures
        self.spatial_index: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        self.type_index: Dict[VectorType, List[str]] = defaultdict(list)
        self.access_queue: deque = deque(maxlen=max_vectors)
        
        # Advanced indexing for sophisticated retrieval
        self.importance_index: Dict[str, List[str]] = defaultdict(list)  # 'high', 'medium', 'low'
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)   # Time-based buckets
        self.tag_index: Dict[str, List[str]] = defaultdict(list)        # Tag-based indexing
        self.association_graph: Dict[str, Set[str]] = defaultdict(set)  # Associative connections
        self.concept_hierarchy: Dict[str, Dict[str, float]] = defaultdict(dict)  # Concept relationships
        
        # Advanced access pattern tracking
        self.access_sequences: deque = deque(maxlen=1000)  # Track access sequences
        self.co_access_patterns: Dict[Tuple[str, str], int] = defaultdict(int)  # Co-occurrence tracking
        self.prediction_cache: Dict[str, List[str]] = {}  # Predictive prefetching
        
        # Compression system
        compressed_dim = int(vector_dim * compression_ratio)
        self.compression_unit = MemoryCompressionUnit(vector_dim, compressed_dim)
        
        # Access pattern analysis
        self.access_patterns: Dict[str, MemoryAccessPattern] = {}
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        
        # Memory management
        self.memory_lock = threading.RLock()
        self.cleanup_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Enhanced statistics
        self.stats = {
            'total_stores': 0,
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_ratio': 0.0,
            'average_access_time': 0.0,
            'prediction_accuracy': 0.0,
            'index_efficiency': 0.0,
            'associative_retrievals': 0,
            'temporal_retrievals': 0,
            'spatial_retrievals': 0
        }
        
        # Vector similarity cache with enhanced management
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_max_size = 10000
        self.cache_ttl = 3600  # 1 hour TTL
        
        self._start_cleanup_thread()
        
        logger.info(f"Enhanced SharedMemorySystem initialized with {max_vectors} capacity")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                time.sleep(60)  # Cleanup every minute
                self._cleanup_expired_vectors()
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def store_vector(self, vector_id: str, content: torch.Tensor, 
                    vector_type: VectorType, coordinates: Tuple[int, int, int],
                    importance: float = 0.5, ttl: Optional[float] = None,
                    tags: Optional[List[str]] = None, compress: bool = True) -> bool:
        """Store a vector in shared memory"""
        
        with self.memory_lock:
            start_time = time.time()
            
            # Check if we need to make space
            if len(self.memory_vectors) >= self.max_vectors:
                self._evict_least_important()
            
            # Optionally compress the vector
            stored_content = content.clone()
            if compress and content.numel() > 64:  # Only compress larger vectors
                compressed, _, quality = self.compression_unit(content.unsqueeze(0))
                if quality.item() > 0.8:  # Only store if quality is good
                    stored_content = compressed.squeeze(0)
            
            # Create memory vector
            memory_vector = MemoryVector(
                vector_id=vector_id,
                content=stored_content,
                vector_type=vector_type,
                coordinates=coordinates,
                importance=importance,
                access_count=0,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                tags=tags or []
            )
            
            # Store in main memory
            self.memory_vectors[vector_id] = memory_vector
            
            # Update indices
            self.spatial_index[coordinates].append(vector_id)
            self.type_index[vector_type].append(vector_id)
            self.access_queue.append(vector_id)
            
            # Update statistics
            self.stats['total_stores'] += 1
            self.stats['average_access_time'] = (
                self.stats['average_access_time'] * (self.stats['total_stores'] - 1) + 
                (time.time() - start_time)
            ) / self.stats['total_stores']
            
            logger.debug(f"Stored vector {vector_id} at {coordinates}")
            return True
    
    def retrieve_vector(self, vector_id: str) -> Optional[torch.Tensor]:
        """Retrieve a vector from shared memory"""
        
        with self.memory_lock:
            start_time = time.time()
            
            if vector_id not in self.memory_vectors:
                self.stats['cache_misses'] += 1
                return None
            
            memory_vector = self.memory_vectors[vector_id]
            
            # Check if expired
            if memory_vector.is_expired():
                self._remove_vector(vector_id)
                self.stats['cache_misses'] += 1
                return None
            
            # Update access statistics
            memory_vector.update_access()
            self.access_history[vector_id].append(time.time())
            
            # Decompress if needed
            content = memory_vector.content
            if content.numel() != self.vector_dim:  # Likely compressed
                decompressed = self.compression_unit.decompress(content.unsqueeze(0))
                content = decompressed.squeeze(0)
            
            # Update statistics
            self.stats['total_retrievals'] += 1
            self.stats['cache_hits'] += 1
            self.stats['average_access_time'] = (
                self.stats['average_access_time'] * (self.stats['total_retrievals'] - 1) + 
                (time.time() - start_time)
            ) / self.stats['total_retrievals']
            
            logger.debug(f"Retrieved vector {vector_id}")
            return content.clone()
    
    def create_pointer(self, vector_id: str, source_agent: str, 
                      target_agent: str, message_type: str = "pointer",
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a pointer to a memory vector"""
        
        if vector_id not in self.memory_vectors:
            return None
        
        pointer_id = str(uuid.uuid4())
        pointer = MemoryPointer(
            pointer_id=pointer_id,
            vector_id=vector_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=message_type,
            metadata=metadata or {},
            created_at=time.time()
        )
        
        self.memory_pointers[pointer_id] = pointer
        logger.debug(f"Created pointer {pointer_id} for vector {vector_id}")
        return pointer_id
    
    def resolve_pointer(self, pointer_id: str) -> Optional[torch.Tensor]:
        """Resolve a pointer to retrieve the referenced vector"""
        
        if pointer_id not in self.memory_pointers:
            return None
        
        pointer = self.memory_pointers[pointer_id]
        return self.retrieve_vector(pointer.vector_id)
    
    def get_pointer_metadata(self, pointer_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a pointer"""
        
        if pointer_id not in self.memory_pointers:
            return None
        
        return self.memory_pointers[pointer_id].to_dict()
    
    def search_vectors_by_coordinates(self, coordinates: Tuple[int, int, int],
                                    radius: float = 1.0) -> List[str]:
        """Search for vectors near given coordinates"""
        
        found_vectors = []
        x, y, z = coordinates
        
        for coord, vector_ids in self.spatial_index.items():
            cx, cy, cz = coord
            distance = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            
            if distance <= radius:
                found_vectors.extend(vector_ids)
        
        return found_vectors
    
    def search_vectors_by_type(self, vector_type: VectorType) -> List[str]:
        """Search for vectors of a specific type"""
        return self.type_index[vector_type].copy()
    
    def search_vectors_by_similarity(self, query_vector: torch.Tensor, 
                                   top_k: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Search for vectors similar to query vector"""
        
        results = []
        
        for vector_id, memory_vector in self.memory_vectors.items():
            if memory_vector.is_expired():
                continue
            
            # Get the stored vector
            stored_content = memory_vector.content
            
            # Decompress if needed
            if stored_content.numel() != self.vector_dim:
                stored_content = self.compression_unit.decompress(stored_content.unsqueeze(0)).squeeze(0)
            
            # Calculate similarity
            similarity = F.cosine_similarity(query_vector, stored_content, dim=0).item()
            
            if similarity >= threshold:
                results.append((vector_id, similarity))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_vectors_by_agent(self, agent_id: str) -> List[str]:
        """Get vectors associated with a specific agent"""
        
        # Search through pointers to find vectors associated with agent
        associated_vectors = set()
        
        for pointer in self.memory_pointers.values():
            if pointer.source_agent == agent_id or pointer.target_agent == agent_id:
                associated_vectors.add(pointer.vector_id)
        
        return list(associated_vectors)
    
    def analyze_access_patterns(self, vector_id: str) -> MemoryAccessPattern:
        """Analyze access patterns for a vector"""
        
        if vector_id not in self.access_history:
            return MemoryAccessPattern.RANDOM
        
        access_times = self.access_history[vector_id]
        
        if len(access_times) < 3:
            return MemoryAccessPattern.RANDOM
        
        # Calculate intervals between accesses
        intervals = np.diff(access_times)
        
        # Determine pattern based on interval analysis
        if np.std(intervals) < 0.1 * np.mean(intervals):
            return MemoryAccessPattern.SEQUENTIAL
        elif np.all(np.diff(intervals) < 0):
            return MemoryAccessPattern.TEMPORAL
        else:
            return MemoryAccessPattern.RANDOM
    
    def _evict_least_important(self):
        """Evict the least important vector to make space"""
        
        if not self.memory_vectors:
            return
        
        # Find vector with lowest importance score
        min_importance = float('inf')
        min_vector_id = None
        
        for vector_id, memory_vector in self.memory_vectors.items():
            # Calculate dynamic importance
            age_factor = time.time() - memory_vector.created_at
            recency_factor = time.time() - memory_vector.last_accessed
            
            dynamic_importance = memory_vector.importance * (1.0 / (age_factor + 1)) * (1.0 / (recency_factor + 1))
            
            if dynamic_importance < min_importance:
                min_importance = dynamic_importance
                min_vector_id = vector_id
        
        if min_vector_id:
            self._remove_vector(min_vector_id)
            logger.debug(f"Evicted vector {min_vector_id} with importance {min_importance}")
    
    def _remove_vector(self, vector_id: str):
        """Remove a vector from memory"""
        
        if vector_id not in self.memory_vectors:
            return
        
        memory_vector = self.memory_vectors[vector_id]
        
        # Remove from indices
        self.spatial_index[memory_vector.coordinates].remove(vector_id)
        if not self.spatial_index[memory_vector.coordinates]:
            del self.spatial_index[memory_vector.coordinates]
        
        self.type_index[memory_vector.vector_type].remove(vector_id)
        if not self.type_index[memory_vector.vector_type]:
            del self.type_index[memory_vector.vector_type]
        
        # Remove from access queue
        if vector_id in self.access_queue:
            self.access_queue.remove(vector_id)
        
        # Remove from main memory
        del self.memory_vectors[vector_id]
        
        # Remove associated pointers
        pointers_to_remove = []
        for pointer_id, pointer in self.memory_pointers.items():
            if pointer.vector_id == vector_id:
                pointers_to_remove.append(pointer_id)
        
        for pointer_id in pointers_to_remove:
            del self.memory_pointers[pointer_id]
        
        # Clean up access history
        if vector_id in self.access_history:
            del self.access_history[vector_id]
    
    def _cleanup_expired_vectors(self):
        """Clean up expired vectors"""
        
        with self.memory_lock:
            expired_vectors = []
            
            for vector_id, memory_vector in self.memory_vectors.items():
                if memory_vector.is_expired():
                    expired_vectors.append(vector_id)
            
            for vector_id in expired_vectors:
                self._remove_vector(vector_id)
                logger.debug(f"Removed expired vector {vector_id}")
    
    def retrieve_by_importance_ranking(self, min_importance: float = 0.5, max_results: int = 20) -> List[Tuple[str, float]]:
        """Retrieve vectors ranked by importance with sophisticated scoring"""
        
        with self.memory_lock:
            results = []
            current_time = time.time()
            
            for vector_id, memory_vector in self.memory_vectors.items():
                if memory_vector.importance < min_importance or memory_vector.is_expired():
                    continue
                
                # Calculate dynamic importance score
                age_factor = 1.0 / (1.0 + (current_time - memory_vector.created_at) / 86400)  # Days
                recency_factor = 1.0 / (1.0 + (current_time - memory_vector.last_accessed) / 3600)  # Hours
                access_frequency = memory_vector.access_count / max(1, len(self.access_history.get(vector_id, [1])))
                
                dynamic_score = (
                    memory_vector.importance * 0.4 +
                    age_factor * 0.2 +
                    recency_factor * 0.2 +
                    min(access_frequency, 1.0) * 0.2
                )
                
                results.append((vector_id, dynamic_score))
            
            # Sort by dynamic score and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            self.stats['importance_retrievals'] = self.stats.get('importance_retrievals', 0) + 1
            return results[:max_results]
    
    def retrieve_by_temporal_locality(self, time_window: float = 3600, pattern_type: str = "recent") -> List[str]:
        """Retrieve vectors based on temporal access patterns"""
        
        with self.memory_lock:
            current_time = time.time()
            results = []
            
            if pattern_type == "recent":
                # Get recently accessed vectors
                for vector_id, memory_vector in self.memory_vectors.items():
                    if current_time - memory_vector.last_accessed <= time_window:
                        results.append(vector_id)
            
            elif pattern_type == "periodic":
                # Detect periodic access patterns
                for vector_id, access_times in self.access_history.items():
                    if len(access_times) >= 3:
                        intervals = np.diff(access_times)
                        if np.std(intervals) < 0.2 * np.mean(intervals):  # Low variance indicates periodicity
                            results.append(vector_id)
            
            elif pattern_type == "bursty":
                # Detect bursty access patterns
                for vector_id, access_times in self.access_history.items():
                    if len(access_times) >= 3:
                        recent_accesses = [t for t in access_times if current_time - t <= time_window]
                        if len(recent_accesses) >= 3:  # Multiple accesses in window
                            results.append(vector_id)
            
            self.stats['temporal_retrievals'] = self.stats.get('temporal_retrievals', 0) + 1
            return results
    
    def retrieve_by_associative_connections(self, seed_vector_id: str, max_depth: int = 3, max_results: int = 15) -> List[Tuple[str, float]]:
        """Retrieve vectors through associative connections with relevance scoring"""
        
        with self.memory_lock:
            if seed_vector_id not in self.memory_vectors:
                return []
            
            visited = set()
            queue = deque([(seed_vector_id, 1.0, 0)])  # (vector_id, relevance_score, depth)
            results = []
            
            while queue and len(results) < max_results:
                current_id, relevance, depth = queue.popleft()
                
                if current_id in visited or depth >= max_depth:
                    continue
                
                visited.add(current_id)
                if current_id != seed_vector_id:
                    results.append((current_id, relevance))
                
                # Find associated vectors through various connections
                associated = set()
                
                # 1. Direct associations in graph
                associated.update(self.association_graph.get(current_id, set()))
                
                # 2. Co-access patterns
                for (vec1, vec2), count in self.co_access_patterns.items():
                    if vec1 == current_id and count >= 3:
                        associated.add(vec2)
                    elif vec2 == current_id and count >= 3:
                        associated.add(vec1)
                
                # 3. Spatial proximity
                if current_id in self.memory_vectors:
                    current_coords = self.memory_vectors[current_id].coordinates
                    for coords, vector_ids in self.spatial_index.items():
                        if self._spatial_distance(current_coords, coords) <= 2:
                            associated.update(vector_ids)
                
                # Add to queue with reduced relevance
                decay_factor = 0.7
                for assoc_id in associated:
                    if assoc_id not in visited:
                        queue.append((assoc_id, relevance * decay_factor, depth + 1))
            
            # Sort by relevance
            results.sort(key=lambda x: x[1], reverse=True)
            self.stats['associative_retrievals'] = self.stats.get('associative_retrievals', 0) + 1
            return results
    
    def retrieve_by_concept_hierarchy(self, concept: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """Retrieve vectors based on concept hierarchy relationships"""
        
        with self.memory_lock:
            results = []
            
            # Find vectors with matching tags or types
            candidate_vectors = set()
            
            # Direct concept matches
            if concept in self.tag_index:
                candidate_vectors.update(self.tag_index[concept])
            
            # Check for VectorType match
            for vector_type in VectorType:
                if vector_type.value == concept:
                    candidate_vectors.update(self.type_index[vector_type])
            
            # Hierarchical concept matches
            if concept in self.concept_hierarchy:
                for related_concept, strength in self.concept_hierarchy[concept].items():
                    if strength >= 0.5:  # Threshold for relevance
                        if related_concept in self.tag_index:
                            candidate_vectors.update(self.tag_index[related_concept])
            
            # Score candidates
            for vector_id in candidate_vectors:
                if vector_id not in self.memory_vectors:
                    continue
                
                memory_vector = self.memory_vectors[vector_id]
                score = 0.0
                
                # Direct match bonus
                if concept in memory_vector.tags or concept == memory_vector.vector_type.value:
                    score += 1.0
                
                # Hierarchical relationship bonus
                if concept in self.concept_hierarchy:
                    for tag in memory_vector.tags:
                        if tag in self.concept_hierarchy[concept]:
                            score += self.concept_hierarchy[concept][tag] * 0.5
                
                # Importance bonus
                score += memory_vector.importance * 0.3
                
                if score > 0:
                    results.append((vector_id, score))
            
            # Sort and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
    
    def predictive_prefetch(self, recent_access_sequence: List[str], max_predictions: int = 5) -> List[str]:
        """Predict and prefetch likely next memory accesses"""
        
        with self.memory_lock:
            if len(recent_access_sequence) < 2:
                return []
            
            predictions = defaultdict(float)
            
            # Pattern 1: Sequential pattern detection
            for i in range(len(recent_access_sequence) - 1):
                current = recent_access_sequence[i]
                next_item = recent_access_sequence[i + 1]
                
                # Look for this pattern in access sequences
                for j in range(len(self.access_sequences) - 1):
                    if (self.access_sequences[j] == current and 
                        self.access_sequences[j + 1] == next_item):
                        
                        # Predict the item after the pattern
                        if j + 2 < len(self.access_sequences):
                            predicted = self.access_sequences[j + 2]
                            predictions[predicted] += 1.0
            
            # Pattern 2: Co-access pattern prediction
            last_accessed = recent_access_sequence[-1]
            for (vec1, vec2), count in self.co_access_patterns.items():
                if vec1 == last_accessed:
                    predictions[vec2] += count / 10.0  # Normalize co-access strength
                elif vec2 == last_accessed:
                    predictions[vec1] += count / 10.0
            
            # Pattern 3: Associative prediction
            for assoc_id in self.association_graph.get(last_accessed, set()):
                predictions[assoc_id] += 0.5
            
            # Sort predictions and return top candidates
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Update prediction accuracy metrics
            if hasattr(self, '_last_predictions'):
                hits = sum(1 for pred in self._last_predictions if pred in recent_access_sequence[-3:])
                accuracy = hits / max(1, len(self._last_predictions))
                self.stats['prediction_accuracy'] = 0.9 * self.stats['prediction_accuracy'] + 0.1 * accuracy
            
            predicted_vectors = [pred[0] for pred in sorted_predictions[:max_predictions]]
            self._last_predictions = predicted_vectors  # Store for accuracy calculation
            
            return predicted_vectors
    
    def _spatial_distance(self, coords1: Tuple[int, int, int], coords2: Tuple[int, int, int]) -> float:
        """Calculate spatial distance between two coordinate points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coords1, coords2)))
    
    def _update_access_patterns(self, vector_id: str):
        """Update sophisticated access pattern tracking"""
        
        current_time = time.time()
        
        # Update access sequence
        self.access_sequences.append(vector_id)
        
        # Update co-access patterns
        if len(self.access_sequences) >= 2:
            prev_access = self.access_sequences[-2]
            self.co_access_patterns[(prev_access, vector_id)] += 1
        
        # Update association graph based on recent accesses
        recent_window = list(self.access_sequences)[-5:]  # Last 5 accesses
        for other_id in recent_window:
            if other_id != vector_id:
                self.association_graph[vector_id].add(other_id)
                self.association_graph[other_id].add(vector_id)
        
        # Update concept hierarchy
        if vector_id in self.memory_vectors:
            self._update_concept_relationships(vector_id)
    
    def _update_concept_relationships(self, vector_id: str):
        """Update concept hierarchy based on access patterns"""
        
        memory_vector = self.memory_vectors[vector_id]
        concepts = [memory_vector.vector_type.value] + memory_vector.tags
        
        # Strengthen relationships between co-occurring concepts
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i != j:
                    if concept2 not in self.concept_hierarchy[concept1]:
                        self.concept_hierarchy[concept1][concept2] = 0.0
                    
                    # Strengthen relationship
                    self.concept_hierarchy[concept1][concept2] = min(1.0, 
                        self.concept_hierarchy[concept1][concept2] + 0.05)
    
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
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        with self.memory_lock:
            total_size = sum(v.content.numel() for v in self.memory_vectors.values())
            
            type_distribution = {}
            for vector_type in VectorType:
                type_distribution[vector_type.value] = len(self.type_index[vector_type])
            
            # Calculate average importance
            avg_importance = np.mean([v.importance for v in self.memory_vectors.values()]) if self.memory_vectors else 0
            
            # Calculate compression efficiency
            compressed_count = sum(1 for v in self.memory_vectors.values() if v.content.numel() != self.vector_dim)
            compression_efficiency = compressed_count / max(1, len(self.memory_vectors))
            
            # Advanced statistics
            importance_distribution = {}
            for importance_level in ["critical", "high", "medium", "low"]:
                importance_distribution[importance_level] = len(self.importance_index[importance_level])
            
            return {
                'total_vectors': len(self.memory_vectors),
                'total_pointers': len(self.memory_pointers),
                'max_capacity': self.max_vectors,
                'usage_percentage': len(self.memory_vectors) / self.max_vectors * 100,
                'total_memory_size': total_size,
                'average_vector_size': total_size / max(1, len(self.memory_vectors)),
                'type_distribution': type_distribution,
                'importance_distribution': importance_distribution,
                'average_importance': avg_importance,
                'compression_efficiency': compression_efficiency,
                'cache_hit_ratio': self.stats['cache_hits'] / max(1, self.stats['total_retrievals']),
                'average_access_time': self.stats['average_access_time'],
                'spatial_localities': len(self.spatial_index),
                'temporal_buckets': len(self.temporal_index),
                'tag_categories': len(self.tag_index),
                'association_connections': sum(len(assocs) for assocs in self.association_graph.values()),
                'co_access_patterns': len(self.co_access_patterns),
                'concept_relationships': len(self.concept_hierarchy),
                'access_patterns': {k: v.value for k, v in self.access_patterns.items()},
                'advanced_stats': dict(self.stats)
            }
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for memory visualization"""
        
        with self.memory_lock:
            vector_data = []
            
            for vector_id, memory_vector in self.memory_vectors.items():
                vector_data.append({
                    'id': vector_id,
                    'type': memory_vector.vector_type.value,
                    'coordinates': memory_vector.coordinates,
                    'importance': memory_vector.importance,
                    'access_count': memory_vector.access_count,
                    'age': time.time() - memory_vector.created_at,
                    'last_accessed': memory_vector.last_accessed,
                    'size': memory_vector.content.numel(),
                    'tags': memory_vector.tags
                })
            
            pointer_data = []
            for pointer_id, pointer in self.memory_pointers.items():
                pointer_data.append({
                    'id': pointer_id,
                    'vector_id': pointer.vector_id,
                    'source_agent': pointer.source_agent,
                    'target_agent': pointer.target_agent,
                    'message_type': pointer.message_type,
                    'created_at': pointer.created_at
                })
            
            return {
                'vectors': vector_data,
                'pointers': pointer_data,
                'spatial_index': {
                    str(coord): vector_ids for coord, vector_ids in self.spatial_index.items()
                },
                'statistics': self.get_memory_statistics()
            }
    
    def optimize_memory_layout(self):
        """Optimize memory layout for better performance"""
        
        with self.memory_lock:
            # Reorder vectors by access frequency
            sorted_vectors = sorted(
                self.memory_vectors.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )
            
            # Rebuild indices
            self.spatial_index.clear()
            self.type_index.clear()
            
            for vector_id, memory_vector in sorted_vectors:
                self.spatial_index[memory_vector.coordinates].append(vector_id)
                self.type_index[memory_vector.vector_type].append(vector_id)
            
            logger.info("Memory layout optimized")
    
    def export_memory_dump(self) -> Dict[str, Any]:
        """Export complete memory dump for analysis"""
        
        return {
            'vectors': {
                vector_id: {
                    'content': memory_vector.content.tolist(),
                    'type': memory_vector.vector_type.value,
                    'coordinates': memory_vector.coordinates,
                    'importance': memory_vector.importance,
                    'access_count': memory_vector.access_count,
                    'created_at': memory_vector.created_at,
                    'last_accessed': memory_vector.last_accessed,
                    'tags': memory_vector.tags
                }
                for vector_id, memory_vector in self.memory_vectors.items()
            },
            'pointers': {
                pointer_id: pointer.to_dict()
                for pointer_id, pointer in self.memory_pointers.items()
            },
            'statistics': self.get_memory_statistics()
        }
    
    def shutdown(self):
        """Gracefully shutdown the memory system"""
        
        logger.info("Shutting down shared memory system")
        
        # Stop cleanup thread
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear all data
        with self.memory_lock:
            self.memory_vectors.clear()
            self.memory_pointers.clear()
            self.spatial_index.clear()
            self.type_index.clear()
            self.access_queue.clear()

# Global shared memory instance
shared_memory = SharedMemorySystem()

def get_shared_memory() -> SharedMemorySystem:
    """Get the global shared memory instance"""
    return shared_memory
