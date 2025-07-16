"""
Communication Protocol Implementation
Handles hierarchical communication and message routing
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import time
import asyncio
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .shared_memory import SharedMemorySystem, VectorType, get_shared_memory
from .agent_network import AgentNetworkManager, get_network_manager

logger = logging.getLogger(__name__)

class MessageType(Enum):
    POINTER = "pointer"
    BROADCAST = "broadcast"
    BREAKTHROUGH = "breakthrough"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    QUERY = "query"
    RESPONSE = "response"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class CommunicationMode(Enum):
    DIRECT = "direct"
    HIERARCHICAL = "hierarchical"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"

@dataclass
class Message:
    """Communication message between agents"""
    message_id: str
    source_agent: str
    target_agent: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    memory_pointer: Optional[str] = None
    timestamp: float = 0.0
    ttl: Optional[float] = None
    route: List[str] = None
    delivery_attempts: int = 0
    max_attempts: int = 3
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.route is None:
            self.route = []
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'source_agent': self.source_agent,
            'target_agent': self.target_agent,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'content': self.content,
            'memory_pointer': self.memory_pointer,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'route': self.route,
            'delivery_attempts': self.delivery_attempts
        }

class MessageQueue:
    """Priority-based message queue"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {
            MessagePriority.URGENT: deque(),
            MessagePriority.HIGH: deque(),
            MessagePriority.NORMAL: deque(),
            MessagePriority.LOW: deque()
        }
        self.lock = threading.Lock()
        self.total_size = 0
    
    def enqueue(self, message: Message) -> bool:
        """Add message to appropriate priority queue"""
        with self.lock:
            if self.total_size >= self.max_size:
                # Remove lowest priority message
                self._remove_lowest_priority()
            
            self.queues[message.priority].append(message)
            self.total_size += 1
            return True
    
    def dequeue(self) -> Optional[Message]:
        """Get highest priority message"""
        with self.lock:
            # Check queues in priority order
            for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                if self.queues[priority]:
                    message = self.queues[priority].popleft()
                    self.total_size -= 1
                    return message
            return None
    
    def _remove_lowest_priority(self):
        """Remove lowest priority message to make space"""
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL, 
                        MessagePriority.HIGH, MessagePriority.URGENT]:
            if self.queues[priority]:
                self.queues[priority].popleft()
                self.total_size -= 1
                break
    
    def size(self) -> int:
        return self.total_size
    
    def get_stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                priority.name: len(queue) 
                for priority, queue in self.queues.items()
            }

class CommunicationProtocol:
    """Main communication protocol handler"""
    
    def __init__(self):
        self.shared_memory = get_shared_memory()
        self.network_manager = get_network_manager()
        
        # Message queues for each agent
        self.agent_queues: Dict[str, MessageQueue] = {}
        
        # Message routing and delivery
        self.message_history: List[Message] = []
        self.delivery_stats = {
            'total_sent': 0,
            'total_delivered': 0,
            'failed_deliveries': 0,
            'average_latency': 0.0,
            'by_type': defaultdict(int),
            'by_priority': defaultdict(int)
        }
        
        # Communication patterns
        self.communication_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.bandwidth_usage: Dict[str, float] = defaultdict(float)
        
        # Protocol state
        self.active_agents: set = set()
        self.protocol_running = False
        self.processing_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Callbacks
        self.message_callbacks: Dict[MessageType, List[callable]] = defaultdict(list)
        
        logger.info("CommunicationProtocol initialized")
    
    def register_agent(self, agent_id: str):
        """Register an agent with the communication protocol"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = MessageQueue()
            self.active_agents.add(agent_id)
            logger.debug(f"Registered agent {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the communication protocol"""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
            self.active_agents.discard(agent_id)
            logger.debug(f"Unregistered agent {agent_id}")
    
    def add_message_callback(self, message_type: MessageType, callback: callable):
        """Add callback for specific message type"""
        self.message_callbacks[message_type].append(callback)
    
    def start_protocol(self):
        """Start the communication protocol"""
        if self.protocol_running:
            return
        
        self.protocol_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Communication protocol started")
    
    def stop_protocol(self):
        """Stop the communication protocol"""
        self.protocol_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        logger.info("Communication protocol stopped")
    
    def _processing_loop(self):
        """Main message processing loop"""
        while self.protocol_running:
            try:
                # Process messages for all agents
                for agent_id in list(self.active_agents):
                    if agent_id in self.agent_queues:
                        message = self.agent_queues[agent_id].dequeue()
                        if message:
                            self._process_message(message)
                
                # Clean up expired messages
                self._cleanup_expired_messages()
                
                time.sleep(0.01)  # Process every 10ms
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def send_message(self, source_agent: str, target_agent: str, 
                    message_type: MessageType, content: Dict[str, Any],
                    priority: MessagePriority = MessagePriority.NORMAL,
                    memory_pointer: Optional[str] = None,
                    ttl: Optional[float] = None) -> str:
        """Send a message from source to target agent"""
        
        # Generate message ID
        message_id = f"msg_{source_agent}_{target_agent}_{int(time.time() * 1000)}"
        
        # Create message
        message = Message(
            message_id=message_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=message_type,
            priority=priority,
            content=content,
            memory_pointer=memory_pointer,
            ttl=ttl
        )
        
        # Route message
        success = self._route_message(message)
        
        # Update statistics
        self.delivery_stats['total_sent'] += 1
        self.delivery_stats['by_type'][message_type.value] += 1
        self.delivery_stats['by_priority'][priority.value] += 1
        
        if success:
            logger.debug(f"Message sent: {message_id} ({source_agent} -> {target_agent})")
        else:
            logger.warning(f"Failed to send message: {message_id}")
            self.delivery_stats['failed_deliveries'] += 1
        
        return message_id
    
    def _route_message(self, message: Message) -> bool:
        """Route message based on communication mode"""
        
        if message.message_type == MessageType.BROADCAST:
            return self._handle_broadcast(message)
        elif message.message_type == MessageType.BREAKTHROUGH:
            return self._handle_breakthrough_message(message)
        elif message.message_type == MessageType.COORDINATION:
            return self._handle_coordination_message(message)
        else:
            return self._handle_direct_message(message)
    
    def _handle_direct_message(self, message: Message) -> bool:
        """Handle direct point-to-point communication"""
        
        # Find route to target
        route = self.network_manager.find_communication_path(
            message.source_agent, message.target_agent
        )
        
        if not route:
            logger.warning(f"No route found for message {message.message_id}")
            return False
        
        message.route = route
        
        # If direct connection, deliver immediately
        if len(route) == 2:
            return self._deliver_message(message)
        
        # Otherwise, route through intermediate agents
        return self._route_through_intermediates(message)
    
    def _handle_broadcast(self, message: Message) -> bool:
        """Handle broadcast messages"""
        
        # Get all agents in the system
        if message.target_agent == "all":
            targets = list(self.active_agents)
        else:
            # Get agents supervised by coordinator
            targets = self._get_supervised_agents(message.source_agent)
        
        success_count = 0
        for target in targets:
            if target != message.source_agent:
                # Create individual message for each target
                individual_message = Message(
                    message_id=f"{message.message_id}_{target}",
                    source_agent=message.source_agent,
                    target_agent=target,
                    message_type=message.message_type,
                    priority=message.priority,
                    content=message.content,
                    memory_pointer=message.memory_pointer,
                    timestamp=message.timestamp,
                    ttl=message.ttl
                )
                
                if self._deliver_message(individual_message):
                    success_count += 1
        
        return success_count > 0
    
    def _handle_breakthrough_message(self, message: Message) -> bool:
        """Handle breakthrough messages using hierarchical routing"""
        
        # If from regular agent, route to coordinator
        if message.source_agent.startswith('agent_'):
            coordinators = self.network_manager.get_regional_coordinators(message.source_agent)
            if coordinators:
                coordinator_id = coordinators[0]
                
                # Create message to coordinator
                coord_message = Message(
                    message_id=f"{message.message_id}_to_coord",
                    source_agent=message.source_agent,
                    target_agent=coordinator_id,
                    message_type=message.message_type,
                    priority=MessagePriority.HIGH,
                    content=message.content,
                    memory_pointer=message.memory_pointer,
                    timestamp=message.timestamp
                )
                
                return self._deliver_message(coord_message)
        
        # If from coordinator, potentially broadcast to other coordinators
        elif message.source_agent.startswith('coordinator_'):
            # Check if breakthrough should be shared system-wide
            if self._should_share_breakthrough(message):
                return self._broadcast_to_coordinators(message)
        
        return False
    
    def _handle_coordination_message(self, message: Message) -> bool:
        """Handle coordination messages"""
        
        # Coordination messages use hierarchical routing
        route = self.network_manager.find_communication_path(
            message.source_agent, message.target_agent, prefer_hierarchical=True
        )
        
        if route:
            message.route = route
            return self._deliver_message(message)
        
        return False
    
    def _route_through_intermediates(self, message: Message) -> bool:
        """Route message through intermediate agents"""
        
        if len(message.route) < 2:
            return False
        
        # Send to next hop
        next_hop = message.route[1]
        
        # Update route (remove current agent)
        message.route = message.route[1:]
        
        # Add to next hop's queue
        if next_hop in self.agent_queues:
            return self.agent_queues[next_hop].enqueue(message)
        
        return False
    
    def _deliver_message(self, message: Message) -> bool:
        """Deliver message to target agent"""
        
        if message.target_agent not in self.agent_queues:
            return False
        
        # Check if message expired
        if message.is_expired():
            return False
        
        # Add to target agent's queue
        success = self.agent_queues[message.target_agent].enqueue(message)
        
        if success:
            # Update delivery statistics
            self.delivery_stats['total_delivered'] += 1
            latency = time.time() - message.timestamp
            self.delivery_stats['average_latency'] = (
                self.delivery_stats['average_latency'] * (self.delivery_stats['total_delivered'] - 1) + 
                latency
            ) / self.delivery_stats['total_delivered']
            
            # Update communication patterns
            self.communication_patterns[message.source_agent][message.target_agent] = \
                self.communication_patterns[message.source_agent].get(message.target_agent, 0) + 1
            
            # Store message in history
            self.message_history.append(message)
            
            # Trigger callbacks
            for callback in self.message_callbacks[message.message_type]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
        
        return success
    
    def _process_message(self, message: Message):
        """Process a message (routing or final delivery)"""
        
        # If message has a route and hasn't reached destination
        if message.route and len(message.route) > 1:
            # Continue routing
            self._route_through_intermediates(message)
        else:
            # Final delivery
            self._deliver_message(message)
    
    def _get_supervised_agents(self, coordinator_id: str) -> List[str]:
        """Get agents supervised by a coordinator"""
        
        supervised = []
        for agent_id in self.active_agents:
            if agent_id.startswith('agent_'):
                # Check if this coordinator supervises the agent
                # This is a simplified check - in practice, you'd query the agent network
                agent_num = int(agent_id.split('_')[1])
                coord_num = int(coordinator_id.split('_')[1])
                
                if agent_num // 9 == coord_num:  # 9 agents per coordinator
                    supervised.append(agent_id)
        
        return supervised
    
    def _should_share_breakthrough(self, message: Message) -> bool:
        """Determine if breakthrough should be shared system-wide"""
        
        # Check breakthrough confidence and impact
        content = message.content
        confidence = content.get('confidence', 0.0)
        impact = content.get('impact_score', 0.0)
        
        # Share if high confidence and impact
        return confidence > 0.8 and impact > 0.7
    
    def _broadcast_to_coordinators(self, message: Message) -> bool:
        """Broadcast message to all coordinators"""
        
        coordinators = [agent_id for agent_id in self.active_agents 
                       if agent_id.startswith('coordinator_')]
        
        success_count = 0
        for coordinator in coordinators:
            if coordinator != message.source_agent:
                coord_message = Message(
                    message_id=f"{message.message_id}_{coordinator}",
                    source_agent=message.source_agent,
                    target_agent=coordinator,
                    message_type=message.message_type,
                    priority=MessagePriority.HIGH,
                    content=message.content,
                    memory_pointer=message.memory_pointer,
                    timestamp=message.timestamp
                )
                
                if self._deliver_message(coord_message):
                    success_count += 1
        
        return success_count > 0
    
    def _cleanup_expired_messages(self):
        """Remove expired messages from queues"""
        
        for agent_id, queue in self.agent_queues.items():
            # This is a simplified cleanup - in practice, you'd need to iterate through queues
            pass
    
    def get_agent_messages(self, agent_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a specific agent"""
        
        agent_messages = [
            msg.to_dict() for msg in self.message_history
            if msg.target_agent == agent_id or msg.source_agent == agent_id
        ]
        
        # Sort by timestamp (newest first)
        agent_messages.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return agent_messages[:limit]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        
        return {
            'delivery_stats': dict(self.delivery_stats),
            'active_agents': len(self.active_agents),
            'total_queues': len(self.agent_queues),
            'queue_stats': {
                agent_id: queue.get_stats() 
                for agent_id, queue in self.agent_queues.items()
            },
            'communication_patterns': dict(self.communication_patterns),
            'bandwidth_usage': dict(self.bandwidth_usage)
        }
    
    def get_network_activity(self) -> Dict[str, Any]:
        """Get current network activity"""
        
        # Recent messages (last 60 seconds)
        recent_cutoff = time.time() - 60
        recent_messages = [
            msg.to_dict() for msg in self.message_history
            if msg.timestamp >= recent_cutoff
        ]
        
        # Active communication flows
        active_flows = defaultdict(int)
        for msg in recent_messages:
            flow_key = f"{msg['source_agent']} -> {msg['target_agent']}"
            active_flows[flow_key] += 1
        
        return {
            'recent_messages': recent_messages,
            'active_flows': dict(active_flows),
            'message_rate': len(recent_messages) / 60,  # messages per second
            'protocol_status': 'running' if self.protocol_running else 'stopped'
        }
    
    def create_memory_pointer_message(self, source_agent: str, target_agent: str,
                                    vector_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a pointer-based message referencing shared memory"""
        
        # Create pointer in shared memory
        pointer_id = self.shared_memory.create_pointer(
            vector_id=vector_id,
            source_agent=source_agent,
            target_agent=target_agent,
            message_type="pointer",
            metadata=metadata
        )
        
        if not pointer_id:
            return None
        
        # Send pointer message
        content = {
            'pointer_id': pointer_id,
            'vector_id': vector_id,
            'metadata': metadata or {}
        }
        
        return self.send_message(
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=MessageType.POINTER,
            content=content,
            memory_pointer=pointer_id
        )
    
    def export_communication_data(self) -> Dict[str, Any]:
        """Export communication data for analysis"""
        
        return {
            'message_history': [msg.to_dict() for msg in self.message_history],
            'communication_patterns': dict(self.communication_patterns),
            'delivery_stats': dict(self.delivery_stats),
            'active_agents': list(self.active_agents),
            'protocol_running': self.protocol_running
        }
    
    def shutdown(self):
        """Gracefully shutdown the communication protocol"""
        
        self.stop_protocol()
        self.executor.shutdown(wait=True)
        
        logger.info("CommunicationProtocol shutdown complete")

# Global communication protocol instance
communication_protocol = CommunicationProtocol()

def get_communication_protocol() -> CommunicationProtocol:
    """Get the global communication protocol instance"""
    return communication_protocol
