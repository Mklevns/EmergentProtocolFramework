export interface Message {
  id: string;
  fromAgentId: string;
  toAgentId: string;
  messageType: 'pointer' | 'broadcast' | 'direct';
  payload: any;
  memoryPointer?: string;
  timestamp: Date;
}

export interface CommunicationProtocol {
  id: string;
  name: string;
  description: string;
  efficiency: number;
  usage: number;
  emergenceTime: Date;
}

export interface CommunicationFlow {
  source: string;
  target: string;
  messageType: string;
  frequency: number;
  latency: number;
  reliability: number;
}

export interface NetworkTopology {
  nodes: Array<{
    id: string;
    type: 'regular' | 'coordinator';
    position: { x: number; y: number; z: number };
    active: boolean;
  }>;
  edges: Array<{
    source: string;
    target: string;
    weight: number;
    type: string;
  }>;
}
