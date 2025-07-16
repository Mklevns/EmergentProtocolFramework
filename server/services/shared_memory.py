"""
Shared Vectorized Memory System
Implements efficient pointer-based communication and memory management
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid
import json
import logging
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
        
        # Indexing structures
        self.spatial_index: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        self.type_index: Dict[VectorType, List[str]] = defaultdict(list)
        self.access_queue: deque = deque(maxlen=max_vectors)
        
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
        
        # Statistics
        self.stats = {
            'total_stores': 0,
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_ratio': 0.0,
            'average_access_time': 0.0
        }
        
        # Vector similarity cache
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        
        self._start_cleanup_thread()
        
        logger.info(f"SharedMemorySystem initialized with {max_vectors} capacity")
    
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
            
            return {
                'total_vectors': len(self.memory_vectors),
                'total_pointers': len(self.memory_pointers),
                'max_capacity': self.max_vectors,
                'usage_percentage': len(self.memory_vectors) / self.max_vectors * 100,
                'total_memory_size': total_size,
                'average_vector_size': total_size / max(1, len(self.memory_vectors)),
                'type_distribution': type_distribution,
                'average_importance': avg_importance,
                'compression_efficiency': compression_efficiency,
                'cache_hit_ratio': self.stats['cache_hits'] / max(1, self.stats['total_retrievals']),
                'average_access_time': self.stats['average_access_time'],
                'spatial_localities': len(self.spatial_index),
                'access_patterns': {k: v.value for k, v in self.access_patterns.items()}
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
