export interface Agent {
  id: number;
  agentId: string;
  type: 'regular' | 'coordinator';
  positionX: number;
  positionY: number;
  positionZ: number;
  status: string;
  coordinatorId?: string;
  hiddenDim: number;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface Message {
  id: number;
  fromAgentId: string;
  toAgentId: string;
  messageType: string;
  content: any;
  memoryPointer?: string;
  timestamp: Date;
  isProcessed: boolean;
}

export interface MemoryVector {
  id: number;
  vectorId: string;
  content: any;
  vectorType: string;
  coordinates?: string;
  importance: number;
  accessCount: number;
  createdAt: Date;
  lastAccessed: Date;
}

export interface Breakthrough {
  id: number;
  agentId: string;
  breakthroughType: string;
  description?: string;
  vectorId?: string;
  confidence: number;
  wasShared: boolean;
  timestamp: Date;
}

export interface Experiment {
  id: number;
  name: string;
  description?: string;
  config: any;
  status: string;
  startTime?: Date;
  endTime?: Date;
  metrics?: any;
  createdAt: Date;
}

export interface Metric {
  id: number;
  experimentId: number;
  episode: number;
  step: number;
  metricType: string;
  value: number;
  agentId?: string;
  timestamp: Date;
}

export interface CommunicationPattern {
  id: number;
  fromAgentId: string;
  toAgentId: string;
  frequency: number;
  efficiency: number;
  lastCommunication: Date;
  patternType?: string;
}

export interface AgentGridData {
  agents: Agent[];
  communicationPatterns: CommunicationPattern[];
  activeMessages: Message[];
  recentBreakthroughs: Breakthrough[];
}

export interface TrainingStatus {
  experiment: Experiment;
  currentEpisode: number;
  currentStep: number;
  recentMetrics: Metric[];
  isRunning: boolean;
}

export interface MemoryState {
  vectors: MemoryVector[];
  usage: {
    total: number;
    used: number;
    efficiency: number;
  };
  recentAccess: {
    vectorId: string;
    agentId: string;
    timestamp: string;
  }[];
}

export interface NetworkNode {
  id: string;
  type: string;
  position: { x: number; y: number; z: number };
  status: string;
  coordinator_id?: string;
  activity_level?: number;
  communication_load?: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  distance: number;
  type: string;
  cost?: number;
  flow_strength?: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
}
