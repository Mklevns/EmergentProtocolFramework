"""
Bio-Inspired Multi-Agent Reinforcement Learning Framework
Core framework for hierarchical agent coordination with shared memory
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict
import asyncio
import aiohttp
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    REGULAR = "regular"
    COORDINATOR = "coordinator"

class MessageType(Enum):
    POINTER = "pointer"
    BROADCAST = "broadcast"
    BREAKTHROUGH = "breakthrough"
    COORDINATION = "coordination"

@dataclass
class GridPosition:
    x: int
    y: int
    z: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def distance_to(self, other: 'GridPosition') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class Agent:
    agent_id: str
    agent_type: AgentType
    position: GridPosition
    coordinator_id: Optional[str]
    hidden_dim: int
    status: str = "idle"
    is_active: bool = True

class PheromoneAttentionNetwork(nn.Module):
    """Bio-inspired attention mechanism for agent communication"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for pheromone detection
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Pheromone encoder creates "chemical" signals
        self.pheromone_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded like chemical concentrations
        )
        
        # Spatial encoding for 3D positioning
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Local neighborhood gating
        self.neighborhood_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                positions: torch.Tensor, distance_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatial awareness
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]
            value: Value tensor [batch_size, seq_len, hidden_dim]
            positions: 3D positions [batch_size, seq_len, 3]
            distance_mask: Mask for local neighborhoods [batch_size, seq_len, seq_len]
            
        Returns:
            attended_output: Attended features
            attention_weights: Attention weights for analysis
        """
        
        # Encode spatial positions
        spatial_features = self.spatial_encoder(positions)
        
        # Enhance query and key with spatial information
        enhanced_query = query + spatial_features
        enhanced_key = key + spatial_features
        
        # Apply pheromone encoding
        pheromone_signals = self.pheromone_encoder(enhanced_key)
        
        # Multi-head attention with distance masking
        attended_output, attention_weights = self.attention(
            enhanced_query, pheromone_signals, value
        )
        
        # Apply local neighborhood gating
        gate_input = torch.cat([attended_output, query], dim=-1)
        gate = self.neighborhood_gate(gate_input)
        
        # Apply distance mask to attention weights
        masked_attention = attention_weights * distance_mask
        
        # Gated output
        gated_output = gate * attended_output + (1 - gate) * query
        
        return gated_output, masked_attention

class NeuralPlasticityMemory(nn.Module):
    """Neural plasticity-inspired memory system"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU for memory dynamics
        self.memory_cell = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Plasticity gate determines learning rate
        self.plasticity_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Signal strength encoder for adaptive plasticity
        self.signal_strength_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Initialize GRU with bias towards remembering
        for name, param in self.memory_cell.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.1)
        
    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive plasticity
        
        Args:
            inputs: Input tensor [batch_size, seq_len, hidden_dim]
            hidden_state: Previous hidden state [batch_size, hidden_dim]
            
        Returns:
            new_hidden_state: Updated hidden state
        """
        
        # Compute new memory through GRU
        new_memory, _ = self.memory_cell(inputs, hidden_state.unsqueeze(0))
        new_memory = new_memory.squeeze(0)
        
        # Compute plasticity gate
        gate_input = torch.cat([inputs.squeeze(1), hidden_state], dim=-1)
        plasticity_gate = self.plasticity_gate(gate_input)
        
        # Adaptive plasticity based on signal strength
        signal_strength = self.signal_strength_encoder(inputs.squeeze(1))
        adaptive_rate = plasticity_gate * signal_strength
        
        # Weighted update (smooth gradient flow)
        new_hidden_state = (1 - adaptive_rate) * hidden_state + adaptive_rate * new_memory
        
        return new_hidden_state

class SharedMemorySystem:
    """Shared vectorized memory system for agent coordination"""
    
    def __init__(self, vector_dim: int = 256, max_vectors: int = 1000):
        self.vector_dim = vector_dim
        self.max_vectors = max_vectors
        self.memory_table: Dict[str, torch.Tensor] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.vector_types: Dict[str, str] = {}
        self.coordinates: Dict[str, GridPosition] = {}
        
    def store_vector(self, vector_id: str, vector: torch.Tensor, 
                    vector_type: str, coordinates: GridPosition) -> bool:
        """Store a vector in shared memory"""
        
        if len(self.memory_table) >= self.max_vectors:
            # Remove least accessed vector
            least_accessed = min(self.access_counts, key=self.access_counts.get)
            self.remove_vector(least_accessed)
        
        self.memory_table[vector_id] = vector.clone()
        self.vector_types[vector_id] = vector_type
        self.coordinates[vector_id] = coordinates
        self.access_counts[vector_id] = 0
        
        return True
    
    def retrieve_vector(self, vector_id: str) -> Optional[torch.Tensor]:
        """Retrieve a vector from shared memory"""
        
        if vector_id in self.memory_table:
            self.access_counts[vector_id] += 1
            return self.memory_table[vector_id].clone()
        return None
    
    def remove_vector(self, vector_id: str) -> bool:
        """Remove a vector from shared memory"""
        
        if vector_id in self.memory_table:
            del self.memory_table[vector_id]
            del self.access_counts[vector_id]
            del self.vector_types[vector_id]
            del self.coordinates[vector_id]
            return True
        return False
    
    def get_vectors_by_type(self, vector_type: str) -> List[Tuple[str, torch.Tensor]]:
        """Get all vectors of a specific type"""
        
        result = []
        for vector_id, v_type in self.vector_types.items():
            if v_type == vector_type:
                result.append((vector_id, self.memory_table[vector_id]))
        return result
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        
        return {
            'total_capacity': self.max_vectors,
            'used_capacity': len(self.memory_table),
            'usage_percentage': len(self.memory_table) / self.max_vectors * 100,
            'vector_types': dict(self.vector_types),
            'access_counts': dict(self.access_counts)
        }

class MARLFramework:
    """Main Bio-Inspired Multi-Agent Reinforcement Learning Framework"""
    
    def __init__(self, grid_size: Tuple[int, int, int] = (4, 3, 3), 
                 hidden_dim: int = 256, num_coordinators: int = 3):
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_coordinators = num_coordinators
        
        # Initialize components
        self.agents: Dict[str, Agent] = {}
        self.communication_graph = nx.Graph()
        self.shared_memory = SharedMemorySystem(hidden_dim)
        
        # Neural networks
        self.pheromone_attention = PheromoneAttentionNetwork(hidden_dim)
        self.neural_plasticity = NeuralPlasticityMemory(hidden_dim)
        
        # Communication tracking
        self.communication_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.breakthrough_history: List[Dict[str, Any]] = []
        
        # Training state
        self.current_episode = 0
        self.current_step = 0
        self.is_training = False
        
        logger.info("MARL Framework initialized")
    
    def initialize_agent_grid(self) -> None:
        """Initialize 4x3x3 grid with 27 regular agents and 3 coordinators"""
        
        agent_counter = 0
        
        # Create regular agents
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    if agent_counter < 27:  # 27 regular agents
                        agent_id = f"agent_{agent_counter}"
                        coordinator_id = f"coordinator_{agent_counter // 9}"
                        
                        agent = Agent(
                            agent_id=agent_id,
                            agent_type=AgentType.REGULAR,
                            position=GridPosition(x, y, z),
                            coordinator_id=coordinator_id,
                            hidden_dim=self.hidden_dim
                        )
                        
                        self.agents[agent_id] = agent
                        self.communication_graph.add_node(agent_id, **agent.__dict__)
                        agent_counter += 1
        
        # Create 3 coordinator agents at strategic positions
        coordinator_positions = [
            GridPosition(1, 1, 1),
            GridPosition(2, 1, 1),
            GridPosition(3, 1, 1)
        ]
        
        for i in range(3):
            coordinator_id = f"coordinator_{i}"
            agent = Agent(
                agent_id=coordinator_id,
                agent_type=AgentType.COORDINATOR,
                position=coordinator_positions[i],
                coordinator_id=None,
                hidden_dim=self.hidden_dim * 2  # Coordinators have larger capacity
            )
            
            self.agents[coordinator_id] = agent
            self.communication_graph.add_node(coordinator_id, **agent.__dict__)
        
        # Establish communication links
        self._establish_communication_links()
        
        logger.info(f"Initialized {len(self.agents)} agents in {self.grid_size} grid")
    
    def _establish_communication_links(self) -> None:
        """Establish communication links based on spatial proximity and hierarchy"""
        
        # Direct communication between nearby agents
        for agent_id1, agent1 in self.agents.items():
            for agent_id2, agent2 in self.agents.items():
                if agent_id1 != agent_id2:
                    distance = agent1.position.distance_to(agent2.position)
                    
                    # Connect agents within communication range
                    if distance <= 2.0:  # Communication range
                        self.communication_graph.add_edge(agent_id1, agent_id2, distance=distance)
        
        # Hierarchical communication: regular agents to coordinators
        for agent_id, agent in self.agents.items():
            if agent.agent_type == AgentType.REGULAR and agent.coordinator_id:
                if agent.coordinator_id in self.agents:
                    self.communication_graph.add_edge(agent_id, agent.coordinator_id, 
                                                    distance=0.0, type="hierarchical")
        
        # Inter-coordinator communication
        coordinators = [aid for aid, agent in self.agents.items() 
                       if agent.agent_type == AgentType.COORDINATOR]
        
        for i, coord1 in enumerate(coordinators):
            for coord2 in coordinators[i+1:]:
                self.communication_graph.add_edge(coord1, coord2, 
                                                distance=0.0, type="inter_coordinator")
    
    def process_agent_communication(self, agent_id: str, message_data: Dict[str, Any]) -> None:
        """Process communication from an agent"""
        
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        message_type = MessageType(message_data.get('type', 'pointer'))
        
        if message_type == MessageType.BREAKTHROUGH:
            self._handle_breakthrough_message(agent_id, message_data)
        elif message_type == MessageType.COORDINATION:
            self._handle_coordination_message(agent_id, message_data)
        elif message_type == MessageType.POINTER:
            self._handle_pointer_message(agent_id, message_data)
        
        # Update communication patterns
        target_id = message_data.get('target_id')
        if target_id:
            self.communication_patterns[agent_id][target_id] = \
                self.communication_patterns[agent_id].get(target_id, 0) + 1
    
    def _handle_breakthrough_message(self, agent_id: str, message_data: Dict[str, Any]) -> None:
        """Handle breakthrough detection and sharing"""
        
        breakthrough_vector = torch.tensor(message_data['vector'], dtype=torch.float32)
        vector_id = f"breakthrough_{agent_id}_{self.current_step}"
        
        # Store breakthrough in shared memory
        agent = self.agents[agent_id]
        self.shared_memory.store_vector(vector_id, breakthrough_vector, 
                                      "breakthrough", agent.position)
        
        # Record breakthrough event
        breakthrough_event = {
            'agent_id': agent_id,
            'vector_id': vector_id,
            'timestamp': self.current_step,
            'type': message_data.get('breakthrough_type', 'unknown'),
            'confidence': message_data.get('confidence', 0.0)
        }
        
        self.breakthrough_history.append(breakthrough_event)
        
        # Share with coordinator if regular agent
        if agent.agent_type == AgentType.REGULAR and agent.coordinator_id:
            self._propagate_breakthrough_to_coordinator(agent.coordinator_id, breakthrough_event)
        
        logger.info(f"Breakthrough detected by {agent_id}: {breakthrough_event['type']}")
    
    def _handle_coordination_message(self, agent_id: str, message_data: Dict[str, Any]) -> None:
        """Handle coordination messages between agents"""
        
        if self.agents[agent_id].agent_type == AgentType.COORDINATOR:
            # Coordinator broadcasting to supervised agents
            supervised_agents = [aid for aid, agent in self.agents.items() 
                               if agent.coordinator_id == agent_id]
            
            for supervised_id in supervised_agents:
                self._send_coordination_signal(supervised_id, message_data)
    
    def _handle_pointer_message(self, agent_id: str, message_data: Dict[str, Any]) -> None:
        """Handle pointer-based communication"""
        
        pointer_id = message_data.get('pointer_id')
        if pointer_id:
            vector = self.shared_memory.retrieve_vector(pointer_id)
            if vector is not None:
                # Process retrieved vector
                self._process_retrieved_vector(agent_id, vector, pointer_id)
    
    def _propagate_breakthrough_to_coordinator(self, coordinator_id: str, 
                                             breakthrough_event: Dict[str, Any]) -> None:
        """Propagate breakthrough to coordinator for potential system-wide sharing"""
        
        # Coordinator decides whether to share breakthrough system-wide
        if self._should_share_breakthrough(breakthrough_event):
            # Share with all coordinators
            for coord_id in [aid for aid, agent in self.agents.items() 
                           if agent.agent_type == AgentType.COORDINATOR]:
                if coord_id != coordinator_id:
                    self._send_breakthrough_to_coordinator(coord_id, breakthrough_event)
    
    def _should_share_breakthrough(self, breakthrough_event: Dict[str, Any]) -> bool:
        """Determine if breakthrough should be shared system-wide"""
        
        confidence = breakthrough_event.get('confidence', 0.0)
        return confidence > 0.7  # Threshold for sharing
    
    def _send_breakthrough_to_coordinator(self, coordinator_id: str, 
                                        breakthrough_event: Dict[str, Any]) -> None:
        """Send breakthrough information to coordinator"""
        
        # Implementation for inter-coordinator communication
        pass
    
    def _process_retrieved_vector(self, agent_id: str, vector: torch.Tensor, 
                                pointer_id: str) -> None:
        """Process a vector retrieved from shared memory"""
        
        # Apply neural plasticity to update agent's memory
        agent = self.agents[agent_id]
        
        # Simulate memory update (in real implementation, this would be part of the policy)
        current_state = torch.randn(1, 1, self.hidden_dim)  # Placeholder
        hidden_state = torch.randn(1, self.hidden_dim)      # Placeholder
        
        new_hidden_state = self.neural_plasticity(current_state, hidden_state)
        
        logger.info(f"Agent {agent_id} processed vector {pointer_id}")
    
    def _send_coordination_signal(self, target_id: str, message_data: Dict[str, Any]) -> None:
        """Send coordination signal to target agent"""
        
        # Implementation for sending coordination signals
        pass
    
    def get_communication_network_data(self) -> Dict[str, Any]:
        """Get current communication network data for visualization"""
        
        nodes = []
        edges = []
        
        # Prepare node data
        for agent_id, agent in self.agents.items():
            nodes.append({
                'id': agent_id,
                'type': agent.agent_type.value,
                'position': {
                    'x': agent.position.x,
                    'y': agent.position.y,
                    'z': agent.position.z
                },
                'status': agent.status,
                'coordinator_id': agent.coordinator_id
            })
        
        # Prepare edge data
        for edge in self.communication_graph.edges(data=True):
            edges.append({
                'source': edge[0],
                'target': edge[1],
                'distance': edge[2].get('distance', 0),
                'type': edge[2].get('type', 'spatial')
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'communication_patterns': dict(self.communication_patterns),
            'breakthrough_history': self.breakthrough_history[-10:]  # Last 10 breakthroughs
        }
    
    def get_memory_visualization_data(self) -> Dict[str, Any]:
        """Get shared memory data for visualization"""
        
        return {
            'usage_stats': self.shared_memory.get_memory_usage(),
            'vector_locations': {
                vector_id: {
                    'position': {
                        'x': pos.x,
                        'y': pos.y,
                        'z': pos.z
                    },
                    'type': self.shared_memory.vector_types[vector_id],
                    'access_count': self.shared_memory.access_counts[vector_id]
                }
                for vector_id, pos in self.shared_memory.coordinates.items()
            }
        }
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        
        return {
            'current_episode': self.current_episode,
            'current_step': self.current_step,
            'total_breakthroughs': len(self.breakthrough_history),
            'communication_efficiency': self._calculate_communication_efficiency(),
            'memory_utilization': self.shared_memory.get_memory_usage()['usage_percentage'],
            'active_agents': sum(1 for agent in self.agents.values() if agent.is_active)
        }
    
    def _calculate_communication_efficiency(self) -> float:
        """Calculate overall communication efficiency"""
        
        if not self.communication_patterns:
            return 0.0
        
        total_communications = sum(
            sum(targets.values()) for targets in self.communication_patterns.values()
        )
        
        total_breakthroughs = len(self.breakthrough_history)
        
        if total_communications == 0:
            return 0.0
        
        return total_breakthroughs / total_communications
    
    def step(self) -> None:
        """Single step of the framework"""
        
        self.current_step += 1
        
        # Process any pending communications
        # Update agent states
        # Check for breakthroughs
        # Update memory
        
        # Simulate some activity for demo
        if self.current_step % 10 == 0:
            self._simulate_breakthrough()
    
    def _simulate_breakthrough(self) -> None:
        """Simulate a breakthrough event for demonstration"""
        
        # Select random agent
        agent_id = np.random.choice(list(self.agents.keys()))
        
        # Create simulated breakthrough
        breakthrough_data = {
            'type': 'breakthrough',
            'breakthrough_type': 'pattern_recognition',
            'vector': np.random.randn(self.hidden_dim).tolist(),
            'confidence': np.random.uniform(0.5, 1.0)
        }
        
        self.process_agent_communication(agent_id, breakthrough_data)

# Initialize global framework instance
framework = MARLFramework()

def initialize_framework():
    """Initialize the framework with demo data"""
    framework.initialize_agent_grid()
    return framework

def get_framework():
    """Get the current framework instance"""
    return framework
