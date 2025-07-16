#!/usr/bin/env python3
"""
Agent Initialization Service for Brain-Inspired MARL System
Handles creation and placement of agents in 3D grid structure
"""

import json
import math
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

class AgentType(Enum):
    REGULAR = "regular"
    COORDINATOR = "coordinator"

class AgentStatus(Enum):
    IDLE = "idle"
    COMMUNICATING = "communicating"
    PROCESSING = "processing"
    BREAKTHROUGH = "breakthrough"

@dataclass
class Agent3DPosition:
    x: int
    y: int
    z: int
    
    def distance_to(self, other: 'Agent3DPosition') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class AgentConfig:
    agent_id: str
    agent_type: AgentType
    position: Agent3DPosition
    coordinator_id: str | None
    region: str
    communication_range: float
    neighbors: List[str]
    is_active: bool = True
    status: AgentStatus = AgentStatus.IDLE
    activity_level: float = 0.0
    communication_load: float = 0.0

class AgentInitializer:
    def __init__(self):
        self.grid_dimensions = (4, 3, 3)  # 4x3x3 grid = 36 positions, using 30
        self.total_agents = 30
        self.coordinator_count = 3
        self.communication_range = 2.0  # Distance threshold for direct communication
        
    def get_3d_grid_positions(self) -> List[Agent3DPosition]:
        """Generate all possible positions in the 3D grid"""
        positions = []
        for x in range(self.grid_dimensions[0]):
            for y in range(self.grid_dimensions[1]):
                for z in range(self.grid_dimensions[2]):
                    positions.append(Agent3DPosition(x, y, z))
        return positions
    
    def select_agent_positions(self) -> List[Agent3DPosition]:
        """Select 30 positions from the 36 available positions"""
        all_positions = self.get_3d_grid_positions()
        # Remove 6 positions to get exactly 30 agents
        # Remove corners and some edge positions for better connectivity
        excluded_positions = [
            Agent3DPosition(0, 0, 0),  # Corner
            Agent3DPosition(3, 0, 0),  # Corner
            Agent3DPosition(0, 2, 0),  # Corner
            Agent3DPosition(3, 2, 0),  # Corner
            Agent3DPosition(0, 0, 2),  # Corner
            Agent3DPosition(3, 0, 2),  # Corner
        ]
        
        selected_positions = [pos for pos in all_positions if pos not in excluded_positions]
        return selected_positions[:30]  # Ensure exactly 30 positions
    
    def select_coordinator_positions(self, positions: List[Agent3DPosition]) -> List[Agent3DPosition]:
        """Select 3 strategic positions for coordinators"""
        # Place coordinators in central positions for better oversight
        coordinator_positions = [
            Agent3DPosition(1, 1, 1),  # Center of grid
            Agent3DPosition(2, 1, 1),  # Center-right
            Agent3DPosition(1, 1, 0),  # Center-bottom
        ]
        
        # Ensure these positions are in our selected positions
        valid_coordinators = []
        for coord_pos in coordinator_positions:
            if coord_pos in positions:
                valid_coordinators.append(coord_pos)
        
        # If we don't have enough valid positions, add more from available positions
        while len(valid_coordinators) < 3:
            for pos in positions:
                if pos not in valid_coordinators:
                    valid_coordinators.append(pos)
                    break
        
        return valid_coordinators[:3]
    
    def assign_regions(self, positions: List[Agent3DPosition], coordinator_positions: List[Agent3DPosition]) -> Dict[str, str]:
        """Assign each agent to a coordinator's region"""
        regions = {}
        
        # Define region names
        region_names = ["Alpha", "Beta", "Gamma"]
        
        # Assign each position to the nearest coordinator
        for i, pos in enumerate(positions):
            agent_id = f"agent_{i:02d}"
            
            if pos in coordinator_positions:
                # This is a coordinator
                coord_index = coordinator_positions.index(pos)
                regions[agent_id] = region_names[coord_index]
            else:
                # Find nearest coordinator
                min_distance = float('inf')
                nearest_region = region_names[0]
                
                for j, coord_pos in enumerate(coordinator_positions):
                    distance = pos.distance_to(coord_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_region = region_names[j]
                
                regions[agent_id] = nearest_region
        
        return regions
    
    def find_neighbors(self, agent_pos: Agent3DPosition, all_positions: List[Agent3DPosition]) -> List[str]:
        """Find all agents within communication range"""
        neighbors = []
        
        for i, pos in enumerate(all_positions):
            if pos != agent_pos:  # Don't include self
                distance = agent_pos.distance_to(pos)
                if distance <= self.communication_range:
                    neighbors.append(f"agent_{i:02d}")
        
        return neighbors
    
    def create_agent_configs(self) -> List[AgentConfig]:
        """Create complete agent configurations"""
        positions = self.select_agent_positions()
        coordinator_positions = self.select_coordinator_positions(positions)
        regions = self.assign_regions(positions, coordinator_positions)
        
        agents = []
        
        for i, pos in enumerate(positions):
            agent_id = f"agent_{i:02d}"
            is_coordinator = pos in coordinator_positions
            
            # Determine coordinator assignment
            coordinator_id = None
            if not is_coordinator:
                # Find the coordinator for this agent's region
                agent_region = regions[agent_id]
                for j, coord_pos in enumerate(coordinator_positions):
                    coord_agent_id = f"agent_{positions.index(coord_pos):02d}"
                    if regions[coord_agent_id] == agent_region:
                        coordinator_id = coord_agent_id
                        break
            
            agent_config = AgentConfig(
                agent_id=agent_id,
                agent_type=AgentType.COORDINATOR if is_coordinator else AgentType.REGULAR,
                position=pos,
                coordinator_id=coordinator_id,
                region=regions[agent_id],
                communication_range=self.communication_range,
                neighbors=self.find_neighbors(pos, positions),
                is_active=True,
                status=AgentStatus.IDLE,
                activity_level=random.uniform(0.1, 0.3),  # Small initial activity
                communication_load=0.0
            )
            
            agents.append(agent_config)
        
        return agents
    
    def agents_to_json(self, agents: List[AgentConfig]) -> str:
        """Convert agent configurations to JSON format for Express backend"""
        json_data = []
        
        for agent in agents:
            agent_data = {
                "id": agent.agent_id,
                "type": agent.agent_type.value,
                "position": asdict(agent.position),
                "coordinatorId": agent.coordinator_id,
                "region": agent.region,
                "communicationRange": agent.communication_range,
                "neighbors": agent.neighbors,
                "isActive": agent.is_active,
                "status": agent.status.value,
                "activityLevel": agent.activity_level,
                "communicationLoad": agent.communication_load
            }
            json_data.append(agent_data)
        
        return json.dumps(json_data, indent=2)
    
    def initialize_system(self) -> Dict[str, Any]:
        """Main initialization function that returns complete system state"""
        agents = self.create_agent_configs()
        
        # Generate initial communication patterns (empty but structured)
        communication_patterns = []
        
        # Generate initial memory state
        memory_state = {
            "vectors": [],
            "usage": {
                "total": 1000,
                "used": 0,
                "efficiency": 0.0
            },
            "recentAccess": []
        }
        
        # Generate initial experiment configuration
        experiment_config = {
            "name": "Brain-Inspired MARL Training",
            "description": "Multi-agent reinforcement learning with bio-inspired communication",
            "totalEpisodes": 1000,
            "learningRate": 0.001,
            "batchSize": 32,
            "hiddenDim": 256,
            "breakthroughThreshold": 0.7,
            "status": "initialized"
        }
        
        # Convert agents to JSON-serializable format
        agents_json = []
        for agent in agents:
            agent_dict = asdict(agent)
            agent_dict['agent_type'] = agent.agent_type.value
            agent_dict['status'] = agent.status.value
            agents_json.append(agent_dict)
        
        return {
            "agents": agents_json,
            "communicationPatterns": communication_patterns,
            "memoryState": memory_state,
            "experimentConfig": experiment_config,
            "gridDimensions": self.grid_dimensions,
            "totalAgents": self.total_agents,
            "coordinatorCount": self.coordinator_count
        }

def main():
    """Main function for standalone testing and Express backend integration"""
    initializer = AgentInitializer()
    system_state = initializer.initialize_system()
    
    # Output JSON for Express backend integration
    print(json.dumps(system_state, indent=2))
    
    return system_state

if __name__ == "__main__":
    main()