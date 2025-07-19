"""
Enhanced Shared Memory System with Database Persistence
Combines in-memory vector operations with robust database storage
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import uuid
import logging
import math
import time
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    get_db_manager, 
    MemoryVectorData,
    MessageData
)

logger = logging.getLogger(__name__)

class VectorType(Enum):
    BREAKTHROUGH = "breakthrough"
    CONTEXT = "context"
    COORDINATION = "coordination"
    MEMORY_TRACE = "memory_trace"
    PATTERN = "pattern"
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"

class MemoryAccessPattern(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"

@dataclass
class EnhancedMemoryVector:
    """Enhanced memory vector with persistence capabilities"""
    vector_id: str
    content: torch.Tensor
    vector_type: VectorType
    coordinates: Tuple[int, int, int]
    importance: float
    access_count: int
    created_at: float
    last_accessed: float
    ttl: Optional[float] = None
    tags: List[str] = None
    persistence_level: str = "memory_only"  # "memory_only", "database_cached", "database_persistent"
    compression_ratio: float = 1.0
    quality_score: float = 1.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def should_persist_to_database(self) -> bool:
        """Determine if this vector should be persisted to database"""
        # Persist if high importance, frequently accessed, or explicitly marked
        return (
            self.importance > 0.7 or 
            self.access_count > 10 or 
            self.persistence_level in ["database_cached", "database_persistent"] or
            self.vector_type in [VectorType.BREAKTHROUGH, VectorType.KNOWLEDGE]
        )
    
    def to_database_format(self) -> MemoryVectorData:
        """Convert to database format"""
        # Serialize tensor content
        content_dict = {
            'tensor_data': self.content.detach().cpu().numpy().tolist(),
            'tensor_shape': list(self.content.shape),
            'tensor_dtype': str(self.content.dtype),
            'tags': self.tags,
            'persistence_level': self.persistence_level,
            'compression_ratio': self.compression_ratio,
            'quality_score': self.quality_score,
            'ttl': self.ttl
        }
        
        return MemoryVectorData(
            vector_id=self.vector_id,
            content=content_dict,
            vector_type=self.vector_type.value,
            coordinates=f"{self.coordinates[0]},{self.coordinates[1]},{self.coordinates[2]}",
            importance=self.importance,
            access_count=self.access_count
        )
    
    @classmethod
    def from_database_format(cls, db_vector: MemoryVectorData) -> 'EnhancedMemoryVector':
        """Create from database format"""
        content_dict = db_vector.content
        
        # Reconstruct tensor
        tensor_data = torch.tensor(content_dict['tensor_data'], dtype=getattr(torch, content_dict['tensor_dtype'].split('.')[-1]))
        tensor_data = tensor_data.reshape(content_dict['tensor_shape'])
        
        # Parse coordinates
        coords = tuple(map(int, db_vector.coordinates.split(','))) if db_vector.coordinates else (0, 0, 0)
        
        return cls(
            vector_id=db_vector.vector_id,
            content=tensor_data,
            vector_type=VectorType(db_vector.vector_type),
            coordinates=coords,
            importance=db_vector.importance,
            access_count=db_vector.access_count or 0,
            created_at=db_vector.created_at.timestamp() if db_vector.created_at else time.time(),
            last_accessed=db_vector.last_accessed.timestamp() if db_vector.last_accessed else time.time(),
            tags=content_dict.get('tags', []),
            persistence_level=content_dict.get('persistence_level', 'memory_only'),
            compression_ratio=content_dict.get('compression_ratio', 1.0),
            quality_score=content_dict.get('quality_score', 1.0),
            ttl=content_dict.get('ttl')
        )

class EnhancedMemoryCompressionUnit(nn.Module):
    """Advanced neural network for memory compression with quality preservation"""
    
    def __init__(self, input_dim: int, compressed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        
        # Multi-layer encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, compressed_dim),
            nn.Tanh()
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim),
            nn.Tanh()
        )
        
        # Quality assessment network
        self.quality_assessor = nn.Sequential(
            nn.Linear(input_dim + compressed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Compress input vector and return quality score"""
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        
        # Calculate quality score
        quality_input = torch.cat([x, compressed], dim=-1)
        quality_score = self.quality_assessor(quality_input).item()
        
        return compressed, quality_score
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress vector"""
        return self.decoder(compressed)

class PersistentSharedMemory:
    """Enhanced shared memory system with database persistence"""
    
    def __init__(self, capacity: int = 10000, compression_threshold: float = 0.8, use_database: bool = True):
        self.capacity = capacity
        self.compression_threshold = compression_threshold
        self.use_database = use_database
        
        # In-memory storage
        self.vectors: Dict[str, EnhancedMemoryVector] = {}
        self.vector_index = defaultdict(list)  # Type-based indexing
        self.spatial_index = defaultdict(list)  # Spatial indexing
        self.access_history = deque(maxlen=1000)
        
        # Database connection
        self.db = get_db_manager() if use_database else None
        
        # Compression and quality management
        self.compression_unit = None
        self.compressed_vectors = {}
        
        # Threading and synchronization
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.metrics = {
            'total_vectors': 0,
            'database_hits': 0,
            'memory_hits': 0,
            'compression_saves': 0,
            'quality_degradation': 0.0
        }
        
        logger.info(f"Enhanced shared memory initialized - Capacity: {capacity}, Database: {use_database}")
        
        # Load vectors from database on initialization
        if self.use_database:
            self._load_vectors_from_database()
    
    def _initialize_compression_unit(self, input_dim: int):
        """Initialize compression unit based on vector dimensions"""
        if self.compression_unit is None:
            compressed_dim = max(input_dim // 4, 32)
            self.compression_unit = EnhancedMemoryCompressionUnit(input_dim, compressed_dim)
            logger.info(f"Memory compression unit initialized - Input: {input_dim}, Compressed: {compressed_dim}")
    
    def _load_vectors_from_database(self):
        """Load high-importance vectors from database"""
        if not self.db:
            return
        
        try:
            # Load all vectors from database
            db_vectors = self.db.get_all_memory_vectors()
            loaded_count = 0
            
            for db_vector in db_vectors:
                try:
                    # Convert database vector to enhanced format
                    enhanced_vector = EnhancedMemoryVector.from_database_format(db_vector)
                    
                    # Only load high-importance or recently accessed vectors into memory
                    if enhanced_vector.importance > 0.5 or enhanced_vector.access_count > 5:
                        with self.lock:
                            self.vectors[enhanced_vector.vector_id] = enhanced_vector
                            self._update_indices(enhanced_vector)
                        loaded_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to load vector {db_vector.vector_id} from database: {e}")
            
            logger.info(f"Loaded {loaded_count} vectors from database into memory")
            
        except Exception as e:
            logger.error(f"Failed to load vectors from database: {e}")
    
    def _update_indices(self, vector: EnhancedMemoryVector):
        """Update internal indices for fast lookup"""
        # Type-based index
        self.vector_index[vector.vector_type.value].append(vector.vector_id)
        
        # Spatial index (simplified grid-based)
        spatial_key = f"{vector.coordinates[0]//2}_{vector.coordinates[1]//2}_{vector.coordinates[2]//2}"
        self.spatial_index[spatial_key].append(vector.vector_id)
    
    def store_vector(self, vector_id: str, content: torch.Tensor, vector_type: VectorType, 
                    coordinates: Tuple[int, int, int], importance: float = 0.5, 
                    tags: List[str] = None, persistence_level: str = "memory_only") -> bool:
        """Store a vector in shared memory with optional database persistence"""
        
        if tags is None:
            tags = []
        
        try:
            with self.lock:
                # Initialize compression unit if needed
                if len(content.shape) > 0:
                    self._initialize_compression_unit(content.shape[-1])
                
                # Check if compression is needed
                quality_score = 1.0
                compression_ratio = 1.0
                
                if (self.compression_unit is not None and 
                    len(self.vectors) > self.capacity * self.compression_threshold):
                    
                    compressed_content, quality_score = self.compression_unit.compress(content.unsqueeze(0))
                    content = compressed_content.squeeze(0)
                    compression_ratio = content.numel() / content.numel()  # Simplified
                    self.metrics['compression_saves'] += 1
                
                # Create enhanced memory vector
                enhanced_vector = EnhancedMemoryVector(
                    vector_id=vector_id,
                    content=content,
                    vector_type=vector_type,
                    coordinates=coordinates,
                    importance=importance,
                    access_count=0,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    tags=tags,
                    persistence_level=persistence_level,
                    compression_ratio=compression_ratio,
                    quality_score=quality_score
                )
                
                # Store in memory
                self.vectors[vector_id] = enhanced_vector
                self._update_indices(enhanced_vector)
                self.metrics['total_vectors'] += 1
                
                # Persist to database if needed
                if self.db and enhanced_vector.should_persist_to_database():
                    self.executor.submit(self._persist_vector_to_database, enhanced_vector)
                
                # Memory management - remove least important vectors if over capacity
                if len(self.vectors) > self.capacity:
                    self._cleanup_memory()
                
                logger.debug(f"Stored vector {vector_id} - Type: {vector_type.value}, Importance: {importance:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store vector {vector_id}: {e}")
            return False
    
    def _persist_vector_to_database(self, vector: EnhancedMemoryVector):
        """Persist vector to database in background"""
        try:
            db_vector = vector.to_database_format()
            self.db.create_memory_vector(db_vector)
            logger.debug(f"Persisted vector {vector.vector_id} to database")
            
        except Exception as e:
            logger.error(f"Failed to persist vector {vector.vector_id} to database: {e}")
    
    def retrieve_vector(self, vector_id: str) -> Optional[torch.Tensor]:
        """Retrieve vector content by ID"""
        try:
            with self.lock:
                # Check memory first
                if vector_id in self.vectors:
                    vector = self.vectors[vector_id]
                    vector.access_count += 1
                    vector.last_accessed = time.time()
                    self.access_history.append((vector_id, time.time()))
                    self.metrics['memory_hits'] += 1
                    
                    # Decompress if needed
                    content = vector.content
                    if vector.compression_ratio < 1.0 and self.compression_unit:
                        content = self.compression_unit.decompress(content.unsqueeze(0)).squeeze(0)
                    
                    return content
                
                # Check database if enabled
                if self.db:
                    db_vector = self.db.get_memory_vector(vector_id)
                    if db_vector:
                        # Load from database and cache in memory
                        enhanced_vector = EnhancedMemoryVector.from_database_format(db_vector)
                        self.vectors[vector_id] = enhanced_vector
                        self._update_indices(enhanced_vector)
                        
                        # Update access statistics
                        enhanced_vector.access_count += 1
                        enhanced_vector.last_accessed = time.time()
                        self.db.update_memory_access(vector_id)
                        
                        self.metrics['database_hits'] += 1
                        return enhanced_vector.content
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve vector {vector_id}: {e}")
            return None
    
    def get_vectors_by_type(self, vector_type: VectorType, limit: int = 50) -> List[Tuple[str, torch.Tensor]]:
        """Get vectors by type"""
        try:
            with self.lock:
                vector_ids = self.vector_index.get(vector_type.value, [])
                results = []
                
                for vector_id in vector_ids[:limit]:
                    content = self.retrieve_vector(vector_id)
                    if content is not None:
                        results.append((vector_id, content))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get vectors by type {vector_type}: {e}")
            return []
    
    def get_nearby_vectors(self, coordinates: Tuple[int, int, int], radius: int = 2, 
                          limit: int = 20) -> List[Tuple[str, torch.Tensor, float]]:
        """Get vectors within spatial radius"""
        try:
            with self.lock:
                results = []
                
                # Search in spatial index
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            search_coords = (
                                (coordinates[0] + dx) // 2,
                                (coordinates[1] + dy) // 2,
                                (coordinates[2] + dz) // 2
                            )
                            spatial_key = f"{search_coords[0]}_{search_coords[1]}_{search_coords[2]}"
                            
                            for vector_id in self.spatial_index.get(spatial_key, []):
                                if vector_id in self.vectors:
                                    vector = self.vectors[vector_id]
                                    distance = math.sqrt(
                                        (vector.coordinates[0] - coordinates[0]) ** 2 +
                                        (vector.coordinates[1] - coordinates[1]) ** 2 +
                                        (vector.coordinates[2] - coordinates[2]) ** 2
                                    )
                                    
                                    if distance <= radius:
                                        content = self.retrieve_vector(vector_id)
                                        if content is not None:
                                            results.append((vector_id, content, distance))
                
                # Sort by distance and limit results
                results.sort(key=lambda x: x[2])
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get nearby vectors: {e}")
            return []
    
    def _cleanup_memory(self):
        """Remove least important vectors from memory"""
        try:
            # Calculate cleanup threshold
            target_size = int(self.capacity * 0.8)
            vectors_to_remove = len(self.vectors) - target_size
            
            if vectors_to_remove <= 0:
                return
            
            # Sort vectors by importance and access patterns
            vector_scores = []
            for vector_id, vector in self.vectors.items():
                # Score based on importance, access count, and recency
                recency_score = 1.0 / (1.0 + time.time() - vector.last_accessed)
                combined_score = (vector.importance * 0.4 + 
                                min(vector.access_count / 10.0, 1.0) * 0.4 + 
                                recency_score * 0.2)
                
                vector_scores.append((combined_score, vector_id))
            
            # Remove lowest scoring vectors
            vector_scores.sort()
            for _, vector_id in vector_scores[:vectors_to_remove]:
                vector = self.vectors[vector_id]
                
                # Persist to database before removal if not already persisted
                if self.db and vector.should_persist_to_database():
                    self.executor.submit(self._persist_vector_to_database, vector)
                
                # Remove from memory and indices
                del self.vectors[vector_id]
                
                # Clean up indices
                for type_vectors in self.vector_index.values():
                    if vector_id in type_vectors:
                        type_vectors.remove(vector_id)
                
                for spatial_vectors in self.spatial_index.values():
                    if vector_id in spatial_vectors:
                        spatial_vectors.remove(vector_id)
            
            logger.info(f"Cleaned up {vectors_to_remove} vectors from memory")
            
        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage and performance statistics"""
        with self.lock:
            total_vectors = len(self.vectors)
            high_importance_count = sum(1 for v in self.vectors.values() if v.importance > 0.7)
            avg_quality = np.mean([v.quality_score for v in self.vectors.values()]) if self.vectors else 0.0
            
            return {
                'total_vectors': total_vectors,
                'capacity': self.capacity,
                'utilization': total_vectors / self.capacity,
                'high_importance_vectors': high_importance_count,
                'average_quality_score': avg_quality,
                'database_enabled': self.db is not None,
                'performance_metrics': self.metrics.copy(),
                'recent_access_count': len(self.access_history),
                'vector_types': dict(self.vector_index.keys())
            }
    
    def create_memory_pointer(self, source_agent: str, target_agent: str, vector_id: str, 
                            message_type: str, metadata: Dict[str, Any] = None) -> str:
        """Create a memory pointer for inter-agent communication"""
        if metadata is None:
            metadata = {}
        
        pointer_id = str(uuid.uuid4())
        
        # Create message with memory pointer
        if self.db:
            message_data = MessageData(
                from_agent_id=source_agent,
                to_agent_id=target_agent,
                message_type=message_type,
                content={
                    'pointer_id': pointer_id,
                    'vector_id': vector_id,
                    'metadata': metadata,
                    'creation_time': time.time()
                },
                memory_pointer=vector_id
            )
            
            self.db.create_message(message_data)
        
        return pointer_id
    
    def close(self):
        """Clean up resources"""
        try:
            self.executor.shutdown(wait=True)
            if self.db:
                # Final persistence of important vectors
                for vector in self.vectors.values():
                    if vector.should_persist_to_database():
                        try:
                            self._persist_vector_to_database(vector)
                        except Exception as e:
                            logger.error(f"Failed to persist vector {vector.vector_id} on close: {e}")
            
            logger.info("Enhanced shared memory system closed")
            
        except Exception as e:
            logger.error(f"Error closing shared memory system: {e}")

# Global instance
_shared_memory_instance = None

def get_shared_memory(capacity: int = 10000, use_database: bool = True) -> PersistentSharedMemory:
    """Get or create shared memory instance"""
    global _shared_memory_instance
    if _shared_memory_instance is None:
        _shared_memory_instance = PersistentSharedMemory(capacity=capacity, use_database=use_database)
    return _shared_memory_instance

def close_shared_memory():
    """Close shared memory instance"""
    global _shared_memory_instance
    if _shared_memory_instance:
        _shared_memory_instance.close()
        _shared_memory_instance = None