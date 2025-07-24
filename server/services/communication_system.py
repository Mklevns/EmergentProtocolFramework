#!/usr/bin/env python3
"""
Communication System Service for Brain-Inspired MARL
Handles agent-to-agent communication patterns and message routing
"""

import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict
import time

try:
    from .communication_types import (
        MessageType,
        CommunicationPattern, 
        Position3D,
        Agent,
        Message,
        CommunicationMetrics
    )
except ImportError:
    # If relative import fails (when run as main), try absolute import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from communication_types import (
        MessageType,
        CommunicationPattern, 
        Position3D,
        Agent,
        Message,
        CommunicationMetrics
    )

class CommunicationSystem:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_queue: List[Message] = []
        self.communication_patterns: Dict[str, Dict[str, Any]] = {}
        self.routing_table: Dict[str, List[str]] = {}
        self.metrics = CommunicationMetrics(0, 0, 0, 0.0, 0.0, [])
        self.message_counter = 0
        
    def load_agents(self, agents_data: List[Dict]):
        """Load agent data from the system"""
        for agent_data in agents_data:
            position = Position3D(
                agent_data['positionX'],
                agent_data['positionY'], 
                agent_data['positionZ']
            )
            
            agent = Agent(
                agent_id=agent_data['agentId'],
                agent_type=agent_data['type'],
                position=position,
                coordinator_id=agent_data.get('coordinatorId'),
                region=agent_data.get('region', 'Unknown'),
                communication_range=2.0,
                neighbors=[],
                is_active=agent_data['isActive'],
                status=agent_data['status'],
                activity_level=random.uniform(0.1, 0.9),
                communication_load=0.0
            )
            
            self.agents[agent.agent_id] = agent
            
        # Build neighbor lists and routing table
        self._build_neighbor_network()
        self._build_routing_table()
        
    def _build_neighbor_network(self):
        """Build neighbor lists for all agents based on communication range"""
        for agent_id, agent in self.agents.items():
            neighbors = []
            for other_id, other_agent in self.agents.items():
                if agent_id != other_id:
                    distance = agent.position.distance_to(other_agent.position)
                    if distance <= agent.communication_range:
                        neighbors.append(other_id)
            agent.neighbors = neighbors
            
    def _build_routing_table(self):
        """Build routing table for efficient message delivery"""
        for agent_id in self.agents:
            self.routing_table[agent_id] = self._find_shortest_paths(agent_id)
            
    def _find_shortest_paths(self, start_agent: str) -> List[str]:
        """Find shortest paths to all other agents using BFS"""
        visited = set()
        queue = [(start_agent, [start_agent])]
        paths = {}
        
        while queue:
            current, path = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            paths[current] = path
            
            for neighbor in self.agents[current].neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
                    
        return paths
    
    def create_message(self, from_agent: str, to_agent: str, message_type: MessageType, 
                      content: Dict[str, Any], priority: float = 0.5) -> Message:
        """Create a new message for transmission"""
        self.message_counter += 1
        
        message = Message(
            message_id=f"msg_{self.message_counter}",
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            message_type=message_type,
            content=content,
            memory_pointer=content.get('memory_pointer'),
            priority=priority,
            timestamp=time.time(),
            ttl=10,  # 10 hops maximum
            hop_count=0
        )
        
        return message
        
    def route_message(self, message: Message) -> List[str]:
        """Determine the best route for a message"""
        if message.to_agent_id == "broadcast":
            return self._broadcast_route(message.from_agent_id)
        
        # Check for direct communication
        if message.to_agent_id in self.agents[message.from_agent_id].neighbors:
            return [message.from_agent_id, message.to_agent_id]
            
        # Use hierarchical routing through coordinators
        return self._hierarchical_route(message.from_agent_id, message.to_agent_id)
        
    def _broadcast_route(self, from_agent: str) -> List[str]:
        """Create broadcast route based on agent type"""
        agent = self.agents[from_agent]
        if agent.agent_type == "coordinator":
            # Coordinators broadcast to their region
            return [aid for aid, a in self.agents.items() 
                   if a.coordinator_id == from_agent or a.agent_id == from_agent]
        else:
            # Regular agents broadcast to neighbors
            return [from_agent] + agent.neighbors
            
    def _hierarchical_route(self, from_agent: str, to_agent: str) -> List[str]:
        """Route message through coordinator hierarchy"""
        from_agent_obj = self.agents[from_agent]
        to_agent_obj = self.agents[to_agent]
        
        # Same region - direct or through coordinator
        if from_agent_obj.coordinator_id == to_agent_obj.coordinator_id:
            if from_agent_obj.coordinator_id:
                return [from_agent, from_agent_obj.coordinator_id, to_agent]
            else:
                return [from_agent, to_agent]
        
        # Different regions - inter-coordinator communication
        from_coord = from_agent_obj.coordinator_id
        to_coord = to_agent_obj.coordinator_id
        
        if from_coord and to_coord:
            return [from_agent, from_coord, to_coord, to_agent]
        
        # Fallback to direct routing
        return [from_agent, to_agent]
        
    def simulate_communication_round(self) -> Dict[str, Any]:
        """Simulate one round of communication between agents"""
        new_messages = []
        communication_events = []
        
        # Generate messages based on agent activity
        for agent_id, agent in self.agents.items():
            if not agent.is_active:
                continue
                
            # Probability of sending a message based on activity level
            if random.random() < agent.activity_level * 0.3:
                message_type = self._choose_message_type(agent)
                target = self._choose_target(agent)
                
                if target:
                    content = self._generate_message_content(agent, message_type)
                    message = self.create_message(agent_id, target, message_type, content)
                    new_messages.append(message)
                    
                    # Log communication event
                    communication_events.append({
                        'from': agent_id,
                        'to': target,
                        'type': message_type.value,
                        'timestamp': message.timestamp
                    })
        
        # Process message routing and delivery
        delivered_messages = []
        for message in new_messages:
            route = self.route_message(message)
            delivery_success = self._simulate_message_delivery(message, route)
            
            if delivery_success:
                delivered_messages.append({
                    'message_id': message.message_id,
                    'from_agent_id': message.from_agent_id,
                    'to_agent_id': message.to_agent_id,
                    'message_type': message.message_type.value,
                    'content': message.content,
                    'route': route,
                    'timestamp': message.timestamp
                })
                
        # Update communication patterns
        self._update_communication_patterns(communication_events)
        
        # Update metrics
        self.metrics.total_messages += len(new_messages)
        self.metrics.successful_deliveries += len(delivered_messages)
        self.metrics.failed_deliveries += len(new_messages) - len(delivered_messages)
        
        return {
            'messages_sent': len(new_messages),
            'messages_delivered': len(delivered_messages),
            'communication_events': communication_events,
            'delivered_messages': delivered_messages,
            'active_patterns': list(self.communication_patterns.keys()),
            'network_efficiency': self._calculate_network_efficiency()
        }
        
    def _choose_message_type(self, agent: Agent) -> MessageType:
        """Choose message type based on agent role and context"""
        if agent.agent_type == "coordinator":
            return random.choice([
                MessageType.COORDINATION,
                MessageType.BROADCAST,
                MessageType.HEARTBEAT
            ])
        else:
            return random.choice([
                MessageType.POINTER,
                MessageType.BREAKTHROUGH,
                MessageType.HEARTBEAT
            ])
            
    def _choose_target(self, agent: Agent) -> Optional[str]:
        """Choose message target based on agent type and context"""
        if agent.agent_type == "coordinator":
            # Coordinators can message other coordinators or their region
            if random.random() < 0.3:  # Inter-coordinator communication
                coordinators = [aid for aid, a in self.agents.items() 
                              if a.agent_type == "coordinator" and aid != agent.agent_id]
                return random.choice(coordinators) if coordinators else None
            else:  # Region broadcast
                return "broadcast"
        else:
            # Regular agents prefer neighbors or coordinator
            if random.random() < 0.7 and agent.neighbors:
                return random.choice(agent.neighbors)
            elif agent.coordinator_id:
                return agent.coordinator_id
                
        return None
        
    def _generate_message_content(self, agent: Agent, message_type: MessageType) -> Dict[str, Any]:
        """Generate appropriate message content"""
        base_content = {
            'sender_position': asdict(agent.position),
            'sender_region': agent.region,
            'urgency': random.uniform(0.1, 1.0),
            'sequence_number': self.message_counter
        }
        
        if message_type == MessageType.POINTER:
            base_content.update({
                'memory_pointer': f"mem_{random.randint(1000, 9999)}",
                'data_type': random.choice(['pattern', 'breakthrough', 'context'])
            })
        elif message_type == MessageType.BREAKTHROUGH:
            base_content.update({
                'breakthrough_type': random.choice(['pattern_recognition', 'coordination_improvement']),
                'confidence': random.uniform(0.7, 1.0),
                'discovery_data': {'pattern_id': f"pattern_{random.randint(100, 999)}"}
            })
        elif message_type == MessageType.COORDINATION:
            base_content.update({
                'coordination_type': random.choice(['task_allocation', 'resource_sharing', 'synchronization']),
                'target_agents': random.sample(agent.neighbors, min(3, len(agent.neighbors)))
            })
            
        return base_content
        
    def _simulate_message_delivery(self, message: Message, route: List[str]) -> bool:
        """Simulate message delivery with potential failures"""
        # Simple delivery simulation - could be enhanced with network conditions
        delivery_probability = 0.95 - (len(route) * 0.05)  # Longer routes have higher failure rate
        return random.random() < delivery_probability
        
    def _update_communication_patterns(self, events: List[Dict[str, Any]]):
        """Update communication patterns based on events"""
        for event in events:
            pattern_key = f"{event['from']}->{event['to']}"
            
            if pattern_key not in self.communication_patterns:
                self.communication_patterns[pattern_key] = {
                    'frequency': 0,
                    'last_communication': event['timestamp'],
                    'message_types': {},
                    'efficiency': 0.8  # Default efficiency
                }
                
            pattern = self.communication_patterns[pattern_key]
            pattern['frequency'] += 1
            pattern['last_communication'] = event['timestamp']
            
            msg_type = event['type']
            pattern['message_types'][msg_type] = pattern['message_types'].get(msg_type, 0) + 1
            
    def _calculate_network_efficiency(self) -> float:
        """Calculate overall network communication efficiency"""
        if self.metrics.total_messages == 0:
            return 0.0
            
        delivery_rate = self.metrics.successful_deliveries / self.metrics.total_messages
        pattern_diversity = len(self.communication_patterns)
        
        # Efficiency based on delivery rate and pattern diversity
        return min(1.0, delivery_rate * 0.8 + (pattern_diversity / 100) * 0.2)
        
    def get_communication_state(self) -> Dict[str, Any]:
        """Get current communication system state"""
        return {
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.is_active]),
            'communication_patterns': [
                {
                    'from_agent_id': pattern.split('->')[0],
                    'to_agent_id': pattern.split('->')[1],
                    'frequency': data['frequency'],
                    'efficiency': data['efficiency'],
                    'last_communication': data['last_communication']
                }
                for pattern, data in self.communication_patterns.items()
            ],
            'network_metrics': {
                'total_messages': self.metrics.total_messages,
                'successful_deliveries': self.metrics.successful_deliveries,
                'failed_deliveries': self.metrics.failed_deliveries,
                'network_efficiency': self._calculate_network_efficiency(),
                'active_patterns': len(self.communication_patterns)
            }
        }

def main():
    """Main function to run communication simulation"""
    import sys
    
    # Read agents data from stdin or use default
    try:
        agents_data = json.loads(sys.stdin.read())
    except:
        # Default empty state for testing
        agents_data = []
    
    # Initialize communication system
    comm_system = CommunicationSystem()
    
    if agents_data:
        comm_system.load_agents(agents_data)
        
        # Run several communication rounds
        rounds_data = []
        for round_num in range(5):  # 5 rounds of communication
            round_result = comm_system.simulate_communication_round()
            round_result['round_number'] = round_num + 1
            rounds_data.append(round_result)
            
        # Get final state
        final_state = comm_system.get_communication_state()
        
        result = {
            'success': True,
            'communication_rounds': rounds_data,
            'final_state': final_state,
            'summary': {
                'total_messages_generated': sum(r['messages_sent'] for r in rounds_data),
                'total_messages_delivered': sum(r['messages_delivered'] for r in rounds_data),
                'final_network_efficiency': final_state['network_metrics']['network_efficiency'],
                'emergent_patterns': len(final_state['communication_patterns'])
            }
        }
    else:
        result = {
            'success': False,
            'error': 'No agents data provided',
            'final_state': comm_system.get_communication_state()
        }
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()