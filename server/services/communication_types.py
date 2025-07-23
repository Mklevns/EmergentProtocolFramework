
"""
Shared Communication Types
Contains all data structures used across communication modules to avoid circular imports
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

class MessageType(Enum):
    POINTER = "pointer"
    BROADCAST = "broadcast"
    BREAKTHROUGH = "breakthrough"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"

class CommunicationPattern(Enum):
    HIERARCHICAL = "hierarchical"
    DIRECT = "direct"
    BROADCAST = "broadcast"
    EMERGENT = "emergent"

@dataclass
class Position3D:
    x: int
    y: int
    z: int
    
    def distance_to(self, other: 'Position3D') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5

@dataclass
class Agent:
    agent_id: str
    agent_type: str
    position: Position3D
    coordinator_id: Optional[str]
    region: str
    communication_range: float
    neighbors: List[str]
    is_active: bool
    status: str
    activity_level: float
    communication_load: float

@dataclass
class Message:
    message_id: str
    from_agent_id: str
    to_agent_id: str
    message_type: MessageType
    content: Dict[str, Any]
    memory_pointer: Optional[str]
    priority: float
    timestamp: float
    ttl: int  # time to live
    hop_count: int

@dataclass
class CommunicationMetrics:
    total_messages: int
    successful_deliveries: int
    failed_deliveries: int
    average_latency: float
    network_efficiency: float
    emergent_patterns: List[str]

@dataclass
class CommunicationEvent:
    from_agent_id: str
    to_agent_id: str
    message_type: str
    timestamp: float
    route: List[str]
    success: bool
    latency: float

@dataclass
class NetworkTopology:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    efficiency: float
    patterns: List[str]
