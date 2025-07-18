#!/usr/bin/env python3
"""
Enhanced API Handler for Advanced Communication and Memory Features
Provides REST API endpoints for sophisticated MARL communication and memory operations
"""

import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torch, use fallbacks if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using fallback implementations")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using basic implementations")

def get_enhanced_communication_stats() -> Dict[str, Any]:
    """Get enhanced communication protocol statistics"""
    try:
        # Try to import the enhanced communication protocol
        from advanced_communication import get_enhanced_communication_protocol
        
        enhanced_protocol = get_enhanced_communication_protocol()
        stats = enhanced_protocol.get_enhanced_metrics()
        
        return {
            'success': True,
            'data': stats,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting enhanced communication stats: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': {
                'communication_metrics': {
                    'total_messages_processed': 0,
                    'average_embedding_time': 0.0,
                    'bandwidth_efficiency': 0.0,
                    'context_coherence_avg': 0.0,
                    'retrieval_accuracy': 0.0
                },
                'bandwidth_metrics': {
                    'current_usage': 0,
                    'max_bandwidth': 10000,
                    'utilization_percentage': 0,
                    'average_congestion': 0,
                    'peak_congestion': 0,
                    'active_agents': 0,
                    'agent_allocations': {}
                },
                'indexing_metrics': {
                    'retrieval_stats': {
                        'semantic_queries': 0,
                        'temporal_queries': 0,
                        'spatial_queries': 0,
                        'associative_queries': 0,
                        'cache_hits': 0,
                        'cache_misses': 0
                    },
                    'index_sizes': {
                        'semantic': 0,
                        'temporal': 0,
                        'spatial': 0,
                        'importance': 0
                    },
                    'cluster_info': {
                        'total_clusters': 0,
                        'max_clusters': 50,
                        'cluster_distribution': 0
                    },
                    'cache_stats': {
                        'cache_size': 0,
                        'cache_max_size': 1000,
                        'hit_rate': 0.0
                    },
                    'concept_hierarchy_size': 0
                },
                'context_metrics': {
                    'active_contexts': 0,
                    'average_context_coherence': 0.0,
                    'average_urgency': 0.0
                },
                'total_embeddings': 0
            }
        }

def get_advanced_memory_stats() -> Dict[str, Any]:
    """Get advanced memory system statistics"""
    try:
        from shared_memory import get_shared_memory
        
        memory_system = get_shared_memory()
        stats = memory_system.get_memory_statistics()
        
        return {
            'success': True,
            'data': stats,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting advanced memory stats: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': {
                'total_vectors': 0,
                'total_pointers': 0,
                'max_capacity': 1000,
                'usage_percentage': 0,
                'total_memory_size': 0,
                'average_vector_size': 0,
                'type_distribution': {
                    'breakthrough': 0,
                    'context': 0,
                    'coordination': 0,
                    'memory_trace': 0,
                    'pattern': 0
                },
                'importance_distribution': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'average_importance': 0,
                'compression_efficiency': 0,
                'cache_hit_ratio': 0,
                'average_access_time': 0,
                'spatial_localities': 0,
                'temporal_buckets': 0,
                'tag_categories': 0,
                'association_connections': 0,
                'co_access_patterns': 0,
                'concept_relationships': 0,
                'access_patterns': {},
                'advanced_stats': {
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
            }
        }

def semantic_memory_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute semantic memory query"""
    try:
        from shared_memory import get_shared_memory
        
        memory_system = get_shared_memory()
        query_text = query_data.get('query_text', '')
        max_results = query_data.get('max_results', 10)
        threshold = query_data.get('threshold', 0.7)
        
        # Create a simple query vector from text (in practice, you'd use proper NLP embeddings)
        if TORCH_AVAILABLE:
            query_vector = torch.randn(256)  # Placeholder semantic vector
        else:
            # Simple fallback vector representation
            query_vector = [0.1] * 256
        
        # Use the semantic similarity query
        if hasattr(memory_system, 'query_similar_vectors'):
            results = memory_system.query_similar_vectors(query_vector, max_results, threshold)
        else:
            results = []
        
        return {
            'success': True,
            'data': {
                'query_text': query_text,
                'results': results,
                'total_found': len(results),
                'query_vector_dim': query_vector.shape[0]
            }
        }
    except Exception as e:
        logger.error(f"Error in semantic memory query: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': {
                'query_text': query_data.get('query_text', ''),
                'results': [],
                'total_found': 0
            }
        }

def associative_memory_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute associative memory query"""
    try:
        from shared_memory import get_shared_memory
        
        memory_system = get_shared_memory()
        vector_id = query_data.get('vector_id', '')
        max_depth = query_data.get('max_depth', 3)
        max_results = query_data.get('max_results', 15)
        
        # Use associative retrieval if available
        if hasattr(memory_system, 'retrieve_by_associative_connections'):
            results = memory_system.retrieve_by_associative_connections(vector_id, max_depth, max_results)
        else:
            # Fallback to basic vector retrieval
            results = []
        
        return {
            'success': True,
            'data': {
                'seed_vector_id': vector_id,
                'max_depth': max_depth,
                'results': results,
                'total_found': len(results)
            }
        }
    except Exception as e:
        logger.error(f"Error in associative memory query: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': {
                'seed_vector_id': query_data.get('vector_id', ''),
                'results': [],
                'total_found': 0
            }
        }

def get_bandwidth_usage() -> Dict[str, Any]:
    """Get communication bandwidth usage statistics"""
    try:
        from advanced_communication import get_enhanced_communication_protocol
        
        enhanced_protocol = get_enhanced_communication_protocol()
        bandwidth_stats = enhanced_protocol.bandwidth_manager.get_network_stats()
        
        return {
            'success': True,
            'data': bandwidth_stats,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting bandwidth usage: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': {
                'current_usage': 0,
                'max_bandwidth': 10000,
                'utilization_percentage': 0,
                'average_congestion': 0,
                'peak_congestion': 0,
                'active_agents': 0,
                'agent_allocations': {}
            }
        }

def predictive_prefetch(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute predictive prefetching"""
    try:
        from shared_memory import get_shared_memory
        
        memory_system = get_shared_memory()
        access_sequence = query_data.get('access_sequence', [])
        max_predictions = query_data.get('max_predictions', 5)
        
        # Use predictive prefetch if available
        if hasattr(memory_system, 'predictive_prefetch'):
            predictions = memory_system.predictive_prefetch(access_sequence, max_predictions)
        else:
            # Fallback to empty predictions
            predictions = []
        
        return {
            'success': True,
            'data': {
                'access_sequence': access_sequence,
                'predictions': predictions,
                'prediction_count': len(predictions),
                'accuracy_score': getattr(memory_system.stats, 'prediction_accuracy', 0.0) if hasattr(memory_system, 'stats') else 0.0
            }
        }
    except Exception as e:
        logger.error(f"Error in predictive prefetch: {e}")
        return {
            'success': False,
            'error': str(e),
            'data': {
                'access_sequence': query_data.get('access_sequence', []),
                'predictions': [],
                'prediction_count': 0
            }
        }

def main():
    """Main API handler"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No operation specified'}))
        sys.exit(1)
    
    operation = sys.argv[1]
    
    try:
        if operation == 'get_enhanced_communication_stats':
            result = get_enhanced_communication_stats()
        elif operation == 'get_advanced_memory_stats':
            result = get_advanced_memory_stats()
        elif operation == 'semantic_memory_query':
            # Read query data from stdin
            query_data = json.loads(sys.stdin.read())
            result = semantic_memory_query(query_data)
        elif operation == 'associative_memory_query':
            # Read query data from stdin
            query_data = json.loads(sys.stdin.read())
            result = associative_memory_query(query_data)
        elif operation == 'get_bandwidth_usage':
            result = get_bandwidth_usage()
        elif operation == 'predictive_prefetch':
            # Read query data from stdin
            query_data = json.loads(sys.stdin.read())
            result = predictive_prefetch(query_data)
        else:
            result = {'error': f'Unknown operation: {operation}'}
        
        print(json.dumps(result))
    
    except Exception as e:
        logger.error(f"Error in main API handler: {e}")
        print(json.dumps({
            'success': False,
            'error': str(e),
            'operation': operation
        }))

if __name__ == '__main__':
    main()