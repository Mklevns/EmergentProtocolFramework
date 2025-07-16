"""
Agent Network Management
Handles 3D grid topology and hierarchical communication
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import json
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CommunicationRange(Enum):
    IMMEDIATE = 1.0    # Direct neighbors
    LOCAL = 2.0        # Local neighborhood
    REGIONAL = 3.0     # Regional communication
    GLOBAL = float('inf')  # Global broadcast

@dataclass
class NetworkMetrics:
    """Metrics for network analysis"""
    clustering_coefficient: float
    average_path_length: float
    network_diameter: int
    centrality_measures: Dict[str, float]
    communication_efficiency: float

class AgentNetworkManager:
    """Manages the 3D agent network topology and communication routing"""
    
    def __init__(self, grid_size: Tuple[int, int, int] = (4, 3, 3)):
        self.grid_size = grid_size
        self.network = nx.Graph()
        self.spatial_graph = nx.Graph()  # For spatial relationships
        self.hierarchical_graph = nx.DiGraph()  # For hierarchical communication
        
        # Communication routing tables
        self.routing_tables: Dict[str, Dict[str, str]] = {}
        self.communication_costs: Dict[Tuple[str, str], float] = {}
        
        # Network analysis cache
        self.network_metrics_cache: Optional[NetworkMetrics] = None
        self.cache_valid = False
        
    def build_spatial_network(self, agents: Dict[str, any]) -> None:
        """Build spatial network based on agent positions"""
        
        self.network.clear()
        self.spatial_graph.clear()
        
        # Add all agents as nodes
        for agent_id, agent in agents.items():
            self.network.add_node(agent_id, **asdict(agent))
            self.spatial_graph.add_node(agent_id, **asdict(agent))
        
        # Create spatial connections
        for agent_id1, agent1 in agents.items():
            for agent_id2, agent2 in agents.items():
                if agent_id1 != agent_id2:
                    distance = self._calculate_3d_distance(agent1.position, agent2.position)
                    
                    # Add edge if within communication range
                    if distance <= CommunicationRange.LOCAL.value:
                        self.network.add_edge(agent_id1, agent_id2, 
                                            distance=distance, 
                                            type='spatial')
                        self.spatial_graph.add_edge(agent_id1, agent_id2, 
                                                  distance=distance)
        
        self.cache_valid = False
        logger.info(f"Built spatial network with {len(self.network.nodes)} nodes and {len(self.network.edges)} edges")
    
    def build_hierarchical_network(self, agents: Dict[str, any]) -> None:
        """Build hierarchical communication network"""
        
        self.hierarchical_graph.clear()
        
        # Add all agents
        for agent_id, agent in agents.items():
            self.hierarchical_graph.add_node(agent_id, **asdict(agent))
        
        # Create hierarchical connections
        for agent_id, agent in agents.items():
            if agent.agent_type.value == 'regular' and agent.coordinator_id:
                # Regular agent to coordinator
                self.hierarchical_graph.add_edge(agent_id, agent.coordinator_id,
                                               type='upward',
                                               cost=1.0)
                
                # Coordinator to regular agent (for broadcasts)
                self.hierarchical_graph.add_edge(agent.coordinator_id, agent_id,
                                               type='downward',
                                               cost=0.5)
        
        # Inter-coordinator connections
        coordinators = [aid for aid, agent in agents.items() 
                       if agent.agent_type.value == 'coordinator']
        
        for i, coord1 in enumerate(coordinators):
            for coord2 in coordinators[i+1:]:
                self.hierarchical_graph.add_edge(coord1, coord2,
                                               type='inter_coordinator',
                                               cost=2.0)
                self.hierarchical_graph.add_edge(coord2, coord1,
                                               type='inter_coordinator',
                                               cost=2.0)
    
    def _calculate_3d_distance(self, pos1: any, pos2: any) -> float:
        """Calculate 3D Euclidean distance between two positions"""
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)
    
    def find_communication_path(self, source: str, target: str, 
                              prefer_hierarchical: bool = False) -> Optional[List[str]]:
        """Find optimal communication path between two agents"""
        
        if prefer_hierarchical and self.hierarchical_graph.has_node(source) and self.hierarchical_graph.has_node(target):
            try:
                return nx.shortest_path(self.hierarchical_graph, source, target, weight='cost')
            except nx.NetworkXNoPath:
                pass
        
        # Fall back to spatial network
        try:
            return nx.shortest_path(self.network, source, target, weight='distance')
        except nx.NetworkXNoPath:
            return None
    
    def get_neighbors(self, agent_id: str, range_type: CommunicationRange = CommunicationRange.LOCAL) -> List[str]:
        """Get neighbors within specified communication range"""
        
        if agent_id not in self.network:
            return []
        
        if range_type == CommunicationRange.IMMEDIATE:
            return list(self.network.neighbors(agent_id))
        
        # For other ranges, use distance-based filtering
        neighbors = []
        for neighbor in self.network.nodes():
            if neighbor != agent_id and self.network.has_edge(agent_id, neighbor):
                edge_data = self.network.get_edge_data(agent_id, neighbor)
                if edge_data and edge_data.get('distance', float('inf')) <= range_type.value:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def get_regional_coordinators(self, agent_id: str) -> List[str]:
        """Get coordinators responsible for an agent's region"""
        
        if agent_id not in self.network:
            return []
        
        agent_data = self.network.nodes[agent_id]
        if agent_data.get('agent_type') == 'coordinator':
            return [agent_id]
        
        # Find supervising coordinator
        coordinator_id = agent_data.get('coordinator_id')
        if coordinator_id:
            return [coordinator_id]
        
        return []
    
    def calculate_communication_cost(self, source: str, target: str) -> float:
        """Calculate cost of communication between two agents"""
        
        path = self.find_communication_path(source, target)
        if not path:
            return float('inf')
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            current, next_node = path[i], path[i + 1]
            
            # Check hierarchical graph first
            if self.hierarchical_graph.has_edge(current, next_node):
                edge_data = self.hierarchical_graph.get_edge_data(current, next_node)
                total_cost += edge_data.get('cost', 1.0)
            else:
                # Use spatial network
                edge_data = self.network.get_edge_data(current, next_node)
                total_cost += edge_data.get('distance', 1.0)
        
        return total_cost
    
    def route_message(self, source: str, target: str, message_type: str = 'normal') -> Dict[str, any]:
        """Route a message from source to target"""
        
        # Determine routing strategy based on message type
        if message_type == 'breakthrough':
            # Breakthroughs use hierarchical routing
            path = self.find_communication_path(source, target, prefer_hierarchical=True)
        elif message_type == 'broadcast':
            # Broadcasts use hierarchical dissemination
            path = self._find_broadcast_path(source)
        else:
            # Normal messages use shortest path
            path = self.find_communication_path(source, target)
        
        if not path:
            return {'success': False, 'error': 'No path found'}
        
        cost = self.calculate_communication_cost(source, target)
        
        return {
            'success': True,
            'path': path,
            'cost': cost,
            'hops': len(path) - 1,
            'routing_strategy': message_type
        }
    
    def _find_broadcast_path(self, source: str) -> List[str]:
        """Find path for broadcast messages"""
        
        if source not in self.network:
            return []
        
        # If source is coordinator, broadcast to all supervised agents
        agent_data = self.network.nodes[source]
        if agent_data.get('agent_type') == 'coordinator':
            supervised_agents = [
                agent_id for agent_id, data in self.network.nodes(data=True)
                if data.get('coordinator_id') == source
            ]
            return [source] + supervised_agents
        
        # If source is regular agent, route through coordinator
        coordinator_id = agent_data.get('coordinator_id')
        if coordinator_id:
            return [source, coordinator_id]
        
        return [source]
    
    def analyze_network_properties(self) -> NetworkMetrics:
        """Analyze network properties and return metrics"""
        
        if self.cache_valid and self.network_metrics_cache:
            return self.network_metrics_cache
        
        if len(self.network.nodes) == 0:
            return NetworkMetrics(0.0, 0.0, 0, {}, 0.0)
        
        # Calculate clustering coefficient
        clustering = nx.average_clustering(self.network)
        
        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(self.network)
        except nx.NetworkXError:
            avg_path_length = float('inf')
        
        # Calculate network diameter
        try:
            diameter = nx.diameter(self.network)
        except nx.NetworkXError:
            diameter = 0
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(self.network)
        closeness = nx.closeness_centrality(self.network)
        degree = nx.degree_centrality(self.network)
        
        centrality_measures = {
            'betweenness': betweenness,
            'closeness': closeness,
            'degree': degree
        }
        
        # Calculate communication efficiency
        comm_efficiency = self._calculate_global_efficiency()
        
        metrics = NetworkMetrics(
            clustering_coefficient=clustering,
            average_path_length=avg_path_length,
            network_diameter=diameter,
            centrality_measures=centrality_measures,
            communication_efficiency=comm_efficiency
        )
        
        self.network_metrics_cache = metrics
        self.cache_valid = True
        
        return metrics
    
    def _calculate_global_efficiency(self) -> float:
        """Calculate global efficiency of the network"""
        
        if len(self.network.nodes) < 2:
            return 0.0
        
        total_efficiency = 0.0
        node_count = 0
        
        for source in self.network.nodes:
            for target in self.network.nodes:
                if source != target:
                    try:
                        path_length = nx.shortest_path_length(self.network, source, target)
                        if path_length > 0:
                            total_efficiency += 1.0 / path_length
                        node_count += 1
                    except nx.NetworkXNoPath:
                        pass
        
        if node_count == 0:
            return 0.0
        
        return total_efficiency / node_count
    
    def get_network_topology_data(self) -> Dict[str, any]:
        """Get network topology data for visualization"""
        
        nodes = []
        edges = []
        
        # Prepare node data
        for node_id, node_data in self.network.nodes(data=True):
            nodes.append({
                'id': node_id,
                'type': node_data.get('agent_type', 'unknown'),
                'position': {
                    'x': node_data.get('position', {}).get('x', 0),
                    'y': node_data.get('position', {}).get('y', 0),
                    'z': node_data.get('position', {}).get('z', 0)
                },
                'coordinator_id': node_data.get('coordinator_id'),
                'status': node_data.get('status', 'idle')
            })
        
        # Prepare edge data
        for source, target, edge_data in self.network.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'distance': edge_data.get('distance', 0),
                'type': edge_data.get('type', 'spatial'),
                'cost': edge_data.get('cost', 1.0)
            })
        
        # Add hierarchical edges
        hierarchical_edges = []
        for source, target, edge_data in self.hierarchical_graph.edges(data=True):
            hierarchical_edges.append({
                'source': source,
                'target': target,
                'type': edge_data.get('type', 'hierarchical'),
                'cost': edge_data.get('cost', 1.0)
            })
        
        return {
            'nodes': nodes,
            'spatial_edges': edges,
            'hierarchical_edges': hierarchical_edges,
            'network_metrics': asdict(self.analyze_network_properties())
        }
    
    def get_communication_matrix(self) -> np.ndarray:
        """Get communication adjacency matrix"""
        
        nodes = list(self.network.nodes)
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if self.network.has_edge(node1, node2):
                    edge_data = self.network.get_edge_data(node1, node2)
                    matrix[i, j] = 1.0 / (edge_data.get('distance', 1.0) + 1e-6)
        
        return matrix
    
    def find_optimal_coordinator_positions(self, num_coordinators: int = 3) -> List[Tuple[int, int, int]]:
        """Find optimal positions for coordinators to minimize communication costs"""
        
        # Simple heuristic: place coordinators to minimize average distance to regular agents
        regular_agents = [
            (node_id, data) for node_id, data in self.network.nodes(data=True)
            if data.get('agent_type') == 'regular'
        ]
        
        if not regular_agents:
            return [(1, 1, 1), (2, 1, 1), (3, 1, 1)]
        
        # Extract positions
        positions = []
        for _, data in regular_agents:
            pos = data.get('position', {})
            positions.append((pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)))
        
        positions = np.array(positions)
        
        # Use k-means-like approach to find optimal coordinator positions
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_coordinators, random_state=42)
        kmeans.fit(positions)
        
        # Round to grid positions
        optimal_positions = []
        for center in kmeans.cluster_centers_:
            x, y, z = center
            optimal_positions.append((
                int(round(np.clip(x, 0, self.grid_size[0] - 1))),
                int(round(np.clip(y, 0, self.grid_size[1] - 1))),
                int(round(np.clip(z, 0, self.grid_size[2] - 1)))
            ))
        
        return optimal_positions
    
    def validate_network_connectivity(self) -> Dict[str, any]:
        """Validate network connectivity and return issues"""
        
        issues = []
        
        # Check for disconnected components
        components = list(nx.connected_components(self.network))
        if len(components) > 1:
            issues.append({
                'type': 'disconnected_components',
                'count': len(components),
                'components': [list(comp) for comp in components]
            })
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(self.network))
        if isolated_nodes:
            issues.append({
                'type': 'isolated_nodes',
                'nodes': isolated_nodes
            })
        
        # Check for agents without coordinators
        agents_without_coordinators = [
            node_id for node_id, data in self.network.nodes(data=True)
            if data.get('agent_type') == 'regular' and not data.get('coordinator_id')
        ]
        
        if agents_without_coordinators:
            issues.append({
                'type': 'agents_without_coordinators',
                'nodes': agents_without_coordinators
            })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_nodes': len(self.network.nodes),
            'total_edges': len(self.network.edges),
            'connectivity_score': 1.0 - (len(issues) / max(1, len(self.network.nodes)))
        }

# Global network manager instance
network_manager = AgentNetworkManager()

def get_network_manager() -> AgentNetworkManager:
    """Get the global network manager instance"""
    return network_manager
