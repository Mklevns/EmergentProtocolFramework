export interface GridPosition {
  x: number;
  y: number;
  z: number;
}

export interface Agent {
  id: string;
  experimentId: string;
  agentType: 'regular' | 'coordinator';
  gridPosition: GridPosition;
  region: number;
  isActive: boolean;
  createdAt: Date;
}

export interface AgentState {
  id: string;
  position: GridPosition;
  type: 'regular' | 'coordinator';
  region: number;
  isActive: boolean;
  neighbors: string[];
  communicationRange: number;
  lastActivity: Date;
  memoryUsage: number;
  performance: number;
}

export interface AgentNetwork {
  agents: Map<string, AgentState>;
  connections: Map<string, string[]>;
  coordinators: string[];
  regions: Map<number, string[]>;
}

export interface AgentAction {
  type: 'communicate' | 'move' | 'breakthrough' | 'idle';
  target?: string;
  payload?: any;
  timestamp: Date;
}
