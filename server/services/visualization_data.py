"""
Visualization Data Service
Prepares and formats data for 3D visualization and real-time monitoring
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import threading

from .marl_framework import get_framework
from .shared_memory import get_shared_memory
from .agent_network import get_network_manager
from .communication_protocol import get_communication_protocol
from .breakthrough_detector import get_breakthrough_detector

logger = logging.getLogger(__name__)

@dataclass
class AgentVisualizationData:
    """Data for visualizing a single agent"""
    agent_id: str
    position: Tuple[float, float, float]
    agent_type: str
    status: str
    activity_level: float
    communication_strength: float
    memory_usage: float
    breakthrough_count: int
    coordinator_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CommunicationFlowData:
    """Data for visualizing communication flows"""
    source_agent: str
    target_agent: str
    message_count: int
    flow_strength: float
    message_type: str
    latency: float
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MemoryVectorVisualizationData:
    """Data for visualizing memory vectors"""
    vector_id: str
    position: Tuple[float, float, float]
    vector_type: str
    importance: float
    access_count: int
    age: float
    size: int
    connected_agents: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class VisualizationDataService:
    """Service for preparing visualization data"""
    
    def __init__(self, update_interval: float = 0.5):
        self.update_interval = update_interval
        
        # Service references
        self.framework = get_framework()
        self.shared_memory = get_shared_memory()
        self.network_manager = get_network_manager()
        self.communication_protocol = get_communication_protocol()
        self.breakthrough_detector = get_breakthrough_detector()
        
        # Cached data
        self.cached_data: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_lock = threading.Lock()
        
        # Real-time tracking
        self.activity_tracker = defaultdict(lambda: defaultdict(float))
        self.communication_tracker = defaultdict(lambda: defaultdict(int))
        self.memory_access_tracker = defaultdict(int)
        
        # Update thread
        self.update_thread = None
        self.running = False
        
        logger.info("VisualizationDataService initialized")
    
    def start_updates(self):
        """Start real-time data updates"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Visualization data updates started")
    
    def stop_updates(self):
        """Stop real-time data updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        logger.info("Visualization data updates stopped")
    
    def _update_loop(self):
        """Main update loop for real-time data"""
        while self.running:
            try:
                # Update activity tracking
                self._update_activity_tracking()
                
                # Update communication tracking
                self._update_communication_tracking()
                
                # Update memory access tracking
                self._update_memory_tracking()
                
                # Clear old cache entries
                self._cleanup_cache()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in visualization update loop: {e}")
                time.sleep(1.0)
    
    def _update_activity_tracking(self):
        """Update agent activity tracking"""
        
        # Get recent breakthrough events
        breakthrough_summary = self.breakthrough_detector.get_breakthrough_summary(time_window=60)
        
        # Update activity levels based on breakthroughs
        for agent_id, count in breakthrough_summary.get('by_agent', {}).items():
            self.activity_tracker[agent_id]['breakthrough_activity'] = count
        
        # Decay activity levels over time
        decay_factor = 0.95
        for agent_id in self.activity_tracker:
            for metric in self.activity_tracker[agent_id]:
                self.activity_tracker[agent_id][metric] *= decay_factor
    
    def _update_communication_tracking(self):
        """Update communication flow tracking"""
        
        # Get recent network activity
        network_activity = self.communication_protocol.get_network_activity()
        
        # Update communication patterns
        for flow_key, count in network_activity.get('active_flows', {}).items():
            if ' -> ' in flow_key:
                source, target = flow_key.split(' -> ')
                self.communication_tracker[source][target] += count
        
        # Decay communication strength over time
        decay_factor = 0.9
        for source in self.communication_tracker:
            for target in self.communication_tracker[source]:
                self.communication_tracker[source][target] *= decay_factor
    
    def _update_memory_tracking(self):
        """Update memory access tracking"""
        
        # Get memory statistics
        memory_stats = self.shared_memory.get_memory_statistics()
        
        # Update access counts (simplified - in practice, you'd track individual accesses)
        for vector_id in memory_stats.get('recent_access', []):
            self.memory_access_tracker[vector_id] += 1
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        
        current_time = time.time()
        with self.cache_lock:
            expired_keys = []
            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > 60:  # Cache expires after 60 seconds
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cached_data[key]
                del self.cache_timestamps[key]
    
    def get_cached_or_compute(self, key: str, compute_func: callable, 
                            cache_duration: float = 5.0) -> Any:
        """Get cached data or compute if not available/expired"""
        
        current_time = time.time()
        
        with self.cache_lock:
            if (key in self.cached_data and 
                key in self.cache_timestamps and
                current_time - self.cache_timestamps[key] < cache_duration):
                return self.cached_data[key]
            
            # Compute new data
            data = compute_func()
            self.cached_data[key] = data
            self.cache_timestamps[key] = current_time
            
            return data
    
    def get_agent_visualization_data(self) -> List[Dict[str, Any]]:
        """Get visualization data for all agents"""
        
        def compute_agent_data():
            agent_data = []
            
            for agent_id, agent in self.framework.agents.items():
                # Calculate activity level
                activity_level = (
                    self.activity_tracker[agent_id].get('breakthrough_activity', 0) * 0.4 +
                    self.activity_tracker[agent_id].get('communication_activity', 0) * 0.3 +
                    self.activity_tracker[agent_id].get('memory_activity', 0) * 0.3
                )
                
                # Calculate communication strength
                total_communications = sum(
                    self.communication_tracker[agent_id].values()
                )
                communication_strength = min(1.0, total_communications / 10)
                
                # Get breakthrough count
                breakthrough_profile = self.breakthrough_detector.get_agent_breakthrough_profile(agent_id)
                breakthrough_count = breakthrough_profile.get('total_breakthroughs', 0)
                
                # Calculate memory usage (simplified)
                memory_vectors = self.shared_memory.get_vectors_by_agent(agent_id)
                memory_usage = len(memory_vectors) / 100  # Normalized to 0-1
                
                agent_viz_data = AgentVisualizationData(
                    agent_id=agent_id,
                    position=(
                        float(agent.position.x),
                        float(agent.position.y),
                        float(agent.position.z)
                    ),
                    agent_type=agent.agent_type.value,
                    status=agent.status,
                    activity_level=min(1.0, activity_level),
                    communication_strength=communication_strength,
                    memory_usage=min(1.0, memory_usage),
                    breakthrough_count=breakthrough_count,
                    coordinator_id=agent.coordinator_id
                )
                
                agent_data.append(agent_viz_data.to_dict())
            
            return agent_data
        
        return self.get_cached_or_compute("agent_data", compute_agent_data, 2.0)
    
    def get_communication_flow_data(self) -> List[Dict[str, Any]]:
        """Get visualization data for communication flows"""
        
        def compute_flow_data():
            flow_data = []
            
            # Get communication statistics
            comm_stats = self.communication_protocol.get_communication_stats()
            
            # Process communication patterns
            for source_agent, targets in comm_stats.get('communication_patterns', {}).items():
                for target_agent, count in targets.items():
                    if count > 0:
                        # Calculate flow strength
                        flow_strength = min(1.0, count / 20)
                        
                        # Get network activity for this flow
                        network_activity = self.communication_protocol.get_network_activity()
                        flow_key = f"{source_agent} -> {target_agent}"
                        recent_count = network_activity.get('active_flows', {}).get(flow_key, 0)
                        
                        # Calculate success rate (simplified)
                        success_rate = 0.95 if recent_count > 0 else 0.8
                        
                        # Calculate latency (simplified)
                        latency = np.random.uniform(0.01, 0.1)  # Simulated latency
                        
                        flow_viz_data = CommunicationFlowData(
                            source_agent=source_agent,
                            target_agent=target_agent,
                            message_count=int(count),
                            flow_strength=flow_strength,
                            message_type="mixed",
                            latency=latency,
                            success_rate=success_rate
                        )
                        
                        flow_data.append(flow_viz_data.to_dict())
            
            return flow_data
        
        return self.get_cached_or_compute("communication_flows", compute_flow_data, 1.0)
    
    def get_memory_visualization_data(self) -> List[Dict[str, Any]]:
        """Get visualization data for memory vectors"""
        
        def compute_memory_data():
            memory_data = []
            
            # Get memory visualization data from shared memory
            memory_viz = self.shared_memory.get_visualization_data()
            
            for vector_info in memory_viz.get('vectors', []):
                # Find connected agents (those who access this vector)
                connected_agents = []
                for pointer_info in memory_viz.get('pointers', []):
                    if pointer_info.get('vector_id') == vector_info.get('id'):
                        connected_agents.extend([
                            pointer_info.get('source_agent'),
                            pointer_info.get('target_agent')
                        ])
                
                connected_agents = list(set(filter(None, connected_agents)))
                
                memory_viz_data = MemoryVectorVisualizationData(
                    vector_id=vector_info.get('id'),
                    position=vector_info.get('coordinates'),
                    vector_type=vector_info.get('type'),
                    importance=vector_info.get('importance'),
                    access_count=vector_info.get('access_count'),
                    age=vector_info.get('age'),
                    size=vector_info.get('size'),
                    connected_agents=connected_agents
                )
                
                memory_data.append(memory_viz_data.to_dict())
            
            return memory_data
        
        return self.get_cached_or_compute("memory_vectors", compute_memory_data, 3.0)
    
    def get_network_topology_data(self) -> Dict[str, Any]:
        """Get network topology data for visualization"""
        
        def compute_topology_data():
            # Get network topology from network manager
            topology = self.network_manager.get_network_topology_data()
            
            # Enhance with real-time activity
            enhanced_nodes = []
            for node in topology.get('nodes', []):
                node_id = node.get('id')
                
                # Add activity information
                activity = self.activity_tracker.get(node_id, {})
                node['activity_level'] = sum(activity.values())
                
                # Add communication load
                comm_load = sum(self.communication_tracker.get(node_id, {}).values())
                node['communication_load'] = min(1.0, comm_load / 50)
                
                enhanced_nodes.append(node)
            
            # Enhance edges with flow information
            enhanced_edges = []
            for edge in topology.get('spatial_edges', []):
                source = edge.get('source')
                target = edge.get('target')
                
                # Add flow strength
                flow_strength = (
                    self.communication_tracker.get(source, {}).get(target, 0) +
                    self.communication_tracker.get(target, {}).get(source, 0)
                )
                edge['flow_strength'] = min(1.0, flow_strength / 10)
                
                enhanced_edges.append(edge)
            
            return {
                'nodes': enhanced_nodes,
                'edges': enhanced_edges,
                'hierarchical_edges': topology.get('hierarchical_edges', []),
                'network_metrics': topology.get('network_metrics', {})
            }
        
        return self.get_cached_or_compute("network_topology", compute_topology_data, 5.0)
    
    def get_breakthrough_visualization_data(self) -> Dict[str, Any]:
        """Get breakthrough events data for visualization"""
        
        def compute_breakthrough_data():
            # Get breakthrough summary
            summary = self.breakthrough_detector.get_breakthrough_summary(time_window=300)  # Last 5 minutes
            
            # Get agent breakthrough profiles
            agent_profiles = {}
            for agent_id in self.framework.agents.keys():
                profile = self.breakthrough_detector.get_agent_breakthrough_profile(agent_id)
                agent_profiles[agent_id] = profile
            
            # Format recent events for visualization
            recent_events = []
            for event in summary.get('recent_events', []):
                event_viz = {
                    'id': event['event_id'],
                    'agent_id': event['agent_id'],
                    'type': event['breakthrough_type'],
                    'confidence': event['confidence'],
                    'timestamp': event['timestamp'],
                    'position': event['coordinates'],
                    'impact': event['impact_score']
                }
                recent_events.append(event_viz)
            
            return {
                'summary': summary,
                'agent_profiles': agent_profiles,
                'recent_events': recent_events,
                'breakthrough_heatmap': self._generate_breakthrough_heatmap()
            }
        
        return self.get_cached_or_compute("breakthrough_data", compute_breakthrough_data, 2.0)
    
    def _generate_breakthrough_heatmap(self) -> Dict[str, Any]:
        """Generate heatmap data for breakthrough density"""
        
        # Create 3D grid for heatmap
        grid_size = self.framework.grid_size
        heatmap = np.zeros(grid_size)
        
        # Get recent breakthrough events
        breakthrough_data = self.breakthrough_detector.get_breakthrough_summary(time_window=3600)  # Last hour
        
        for event in breakthrough_data.get('recent_events', []):
            x, y, z = event['coordinates']
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
                heatmap[x, y, z] += event['confidence']
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return {
            'data': heatmap.tolist(),
            'max_value': float(heatmap.max()),
            'grid_size': grid_size
        }
    
    def get_training_metrics_data(self) -> Dict[str, Any]:
        """Get training metrics for visualization"""
        
        def compute_metrics_data():
            # Get training metrics from framework
            metrics = self.framework.get_training_metrics()
            
            # Add communication efficiency over time
            comm_stats = self.communication_protocol.get_communication_stats()
            delivery_rate = comm_stats.get('delivery_stats', {}).get('total_delivered', 0) / max(1, comm_stats.get('delivery_stats', {}).get('total_sent', 1))
            
            # Add memory efficiency
            memory_stats = self.shared_memory.get_memory_statistics()
            memory_efficiency = memory_stats.get('cache_hit_ratio', 0.0)
            
            # Add network efficiency
            network_metrics = self.network_manager.analyze_network_properties()
            network_efficiency = network_metrics.communication_efficiency
            
            enhanced_metrics = {
                **metrics,
                'communication_delivery_rate': delivery_rate,
                'memory_efficiency': memory_efficiency,
                'network_efficiency': network_efficiency,
                'system_health': self._calculate_system_health()
            }
            
            return enhanced_metrics
        
        return self.get_cached_or_compute("training_metrics", compute_metrics_data, 1.0)
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        
        # Get component health scores
        comm_health = min(1.0, self.communication_protocol.get_communication_stats().get('delivery_stats', {}).get('total_delivered', 0) / max(1, self.communication_protocol.get_communication_stats().get('delivery_stats', {}).get('total_sent', 1)))
        
        memory_health = self.shared_memory.get_memory_statistics().get('cache_hit_ratio', 0.0)
        
        network_health = self.network_manager.analyze_network_properties().communication_efficiency
        
        # Calculate weighted average
        system_health = (comm_health * 0.4 + memory_health * 0.3 + network_health * 0.3)
        
        return system_health
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive real-time dashboard data"""
        
        return {
            'agents': self.get_agent_visualization_data(),
            'communication_flows': self.get_communication_flow_data(),
            'memory_vectors': self.get_memory_visualization_data(),
            'network_topology': self.get_network_topology_data(),
            'breakthrough_data': self.get_breakthrough_visualization_data(),
            'training_metrics': self.get_training_metrics_data(),
            'timestamp': time.time()
        }
    
    def get_historical_data(self, time_range: Tuple[float, float]) -> Dict[str, Any]:
        """Get historical data for a specific time range"""
        
        start_time, end_time = time_range
        
        # Filter breakthrough events
        breakthrough_events = [
            event.to_dict() for event in self.breakthrough_detector.breakthrough_history
            if start_time <= event.timestamp <= end_time
        ]
        
        # Filter communication messages
        communication_messages = [
            msg.to_dict() for msg in self.communication_protocol.message_history
            if start_time <= msg.timestamp <= end_time
        ]
        
        return {
            'breakthrough_events': breakthrough_events,
            'communication_messages': communication_messages,
            'time_range': time_range,
            'total_events': len(breakthrough_events) + len(communication_messages)
        }
    
    def export_visualization_data(self) -> Dict[str, Any]:
        """Export all visualization data for external analysis"""
        
        return {
            'current_state': self.get_real_time_dashboard_data(),
            'activity_tracking': dict(self.activity_tracker),
            'communication_tracking': dict(self.communication_tracker),
            'memory_access_tracking': dict(self.memory_access_tracker),
            'cache_info': {
                'cached_keys': list(self.cached_data.keys()),
                'cache_timestamps': dict(self.cache_timestamps)
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the visualization data service"""
        
        self.stop_updates()
        logger.info("VisualizationDataService shutdown complete")

# Global visualization data service instance
visualization_service = VisualizationDataService()

def get_visualization_service() -> VisualizationDataService:
    """Get the global visualization data service instance"""
    return visualization_service
