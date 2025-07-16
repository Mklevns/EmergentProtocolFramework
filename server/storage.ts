import { 
  agents, 
  messages, 
  memoryVectors, 
  breakthroughs, 
  experiments, 
  metrics, 
  communicationPatterns,
  type Agent, 
  type InsertAgent,
  type Message,
  type InsertMessage,
  type MemoryVector,
  type InsertMemoryVector,
  type Breakthrough,
  type InsertBreakthrough,
  type Experiment,
  type InsertExperiment,
  type Metric,
  type InsertMetric,
  type CommunicationPattern,
  type InsertCommunicationPattern,
  type AgentGridData,
  type TrainingStatus,
  type MemoryState,
} from "@shared/schema";

export interface IStorage {
  // Agent operations
  createAgent(agent: InsertAgent): Promise<Agent>;
  getAgent(id: string): Promise<Agent | undefined>;
  getAllAgents(): Promise<Agent[]>;
  updateAgentStatus(agentId: string, status: string): Promise<void>;
  getAgentsByType(type: string): Promise<Agent[]>;
  
  // Message operations
  createMessage(message: InsertMessage): Promise<Message>;
  getMessagesByAgent(agentId: string): Promise<Message[]>;
  getUnprocessedMessages(): Promise<Message[]>;
  markMessageProcessed(messageId: number): Promise<void>;
  getRecentMessages(limit?: number): Promise<Message[]>;
  
  // Memory operations
  createMemoryVector(vector: InsertMemoryVector): Promise<MemoryVector>;
  getMemoryVector(vectorId: string): Promise<MemoryVector | undefined>;
  getAllMemoryVectors(): Promise<MemoryVector[]>;
  updateMemoryAccess(vectorId: string): Promise<void>;
  getMemoryByType(type: string): Promise<MemoryVector[]>;
  
  // Breakthrough operations
  createBreakthrough(breakthrough: InsertBreakthrough): Promise<Breakthrough>;
  getBreakthroughsByAgent(agentId: string): Promise<Breakthrough[]>;
  getRecentBreakthroughs(limit?: number): Promise<Breakthrough[]>;
  markBreakthroughShared(breakthroughId: number): Promise<void>;
  
  // Experiment operations
  createExperiment(experiment: InsertExperiment): Promise<Experiment>;
  getExperiment(id: number): Promise<Experiment | undefined>;
  getAllExperiments(): Promise<Experiment[]>;
  updateExperimentStatus(id: number, status: string): Promise<void>;
  updateExperimentMetrics(id: number, metrics: any): Promise<void>;
  
  // Metrics operations
  createMetric(metric: InsertMetric): Promise<Metric>;
  getMetricsByExperiment(experimentId: number): Promise<Metric[]>;
  getRecentMetrics(limit?: number): Promise<Metric[]>;
  
  // Communication pattern operations
  createCommunicationPattern(pattern: InsertCommunicationPattern): Promise<CommunicationPattern>;
  updateCommunicationPattern(fromAgentId: string, toAgentId: string, frequency: number, efficiency: number): Promise<void>;
  getCommunicationPatterns(): Promise<CommunicationPattern[]>;
  
  // Complex queries
  getAgentGridData(): Promise<AgentGridData>;
  getTrainingStatus(experimentId: number): Promise<TrainingStatus | undefined>;
  getMemoryState(): Promise<MemoryState>;
}

export class MemStorage implements IStorage {
  private agents: Map<string, Agent> = new Map();
  private messages: Map<number, Message> = new Map();
  private memoryVectors: Map<string, MemoryVector> = new Map();
  private breakthroughs: Map<number, Breakthrough> = new Map();
  private experiments: Map<number, Experiment> = new Map();
  private metricsMap: Map<number, Metric> = new Map();
  private communicationPatterns: Map<string, CommunicationPattern> = new Map();
  
  private currentId = 1;
  
  // Agent operations
  async createAgent(insertAgent: InsertAgent): Promise<Agent> {
    const agent: Agent = {
      ...insertAgent,
      id: this.currentId++,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.agents.set(agent.agentId, agent);
    return agent;
  }
  
  async getAgent(agentId: string): Promise<Agent | undefined> {
    return this.agents.get(agentId);
  }
  
  async getAllAgents(): Promise<Agent[]> {
    return Array.from(this.agents.values());
  }
  
  async updateAgentStatus(agentId: string, status: string): Promise<void> {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.status = status;
      agent.updatedAt = new Date();
    }
  }
  
  async getAgentsByType(type: string): Promise<Agent[]> {
    return Array.from(this.agents.values()).filter(agent => agent.type === type);
  }
  
  // Message operations
  async createMessage(insertMessage: InsertMessage): Promise<Message> {
    const message: Message = {
      ...insertMessage,
      id: this.currentId++,
      timestamp: new Date(),
      isProcessed: false,
    };
    this.messages.set(message.id, message);
    return message;
  }
  
  async getMessagesByAgent(agentId: string): Promise<Message[]> {
    return Array.from(this.messages.values()).filter(
      msg => msg.fromAgentId === agentId || msg.toAgentId === agentId
    );
  }
  
  async getUnprocessedMessages(): Promise<Message[]> {
    return Array.from(this.messages.values()).filter(msg => !msg.isProcessed);
  }
  
  async markMessageProcessed(messageId: number): Promise<void> {
    const message = this.messages.get(messageId);
    if (message) {
      message.isProcessed = true;
    }
  }
  
  async getRecentMessages(limit = 50): Promise<Message[]> {
    return Array.from(this.messages.values())
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }
  
  // Memory operations
  async createMemoryVector(insertVector: InsertMemoryVector): Promise<MemoryVector> {
    const vector: MemoryVector = {
      ...insertVector,
      id: this.currentId++,
      accessCount: 0,
      createdAt: new Date(),
      lastAccessed: new Date(),
    };
    this.memoryVectors.set(vector.vectorId, vector);
    return vector;
  }
  
  async getMemoryVector(vectorId: string): Promise<MemoryVector | undefined> {
    return this.memoryVectors.get(vectorId);
  }
  
  async getAllMemoryVectors(): Promise<MemoryVector[]> {
    return Array.from(this.memoryVectors.values());
  }
  
  async updateMemoryAccess(vectorId: string): Promise<void> {
    const vector = this.memoryVectors.get(vectorId);
    if (vector) {
      vector.accessCount++;
      vector.lastAccessed = new Date();
    }
  }
  
  async getMemoryByType(type: string): Promise<MemoryVector[]> {
    return Array.from(this.memoryVectors.values()).filter(v => v.vectorType === type);
  }
  
  // Breakthrough operations
  async createBreakthrough(insertBreakthrough: InsertBreakthrough): Promise<Breakthrough> {
    const breakthrough: Breakthrough = {
      ...insertBreakthrough,
      id: this.currentId++,
      timestamp: new Date(),
    };
    this.breakthroughs.set(breakthrough.id, breakthrough);
    return breakthrough;
  }
  
  async getBreakthroughsByAgent(agentId: string): Promise<Breakthrough[]> {
    return Array.from(this.breakthroughs.values()).filter(b => b.agentId === agentId);
  }
  
  async getRecentBreakthroughs(limit = 20): Promise<Breakthrough[]> {
    return Array.from(this.breakthroughs.values())
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }
  
  async markBreakthroughShared(breakthroughId: number): Promise<void> {
    const breakthrough = this.breakthroughs.get(breakthroughId);
    if (breakthrough) {
      breakthrough.wasShared = true;
    }
  }
  
  // Experiment operations
  async createExperiment(insertExperiment: InsertExperiment): Promise<Experiment> {
    const experiment: Experiment = {
      ...insertExperiment,
      id: this.currentId++,
      createdAt: new Date(),
    };
    this.experiments.set(experiment.id, experiment);
    return experiment;
  }
  
  async getExperiment(id: number): Promise<Experiment | undefined> {
    return this.experiments.get(id);
  }
  
  async getAllExperiments(): Promise<Experiment[]> {
    return Array.from(this.experiments.values());
  }
  
  async updateExperimentStatus(id: number, status: string): Promise<void> {
    const experiment = this.experiments.get(id);
    if (experiment) {
      experiment.status = status;
      if (status === "running" && !experiment.startTime) {
        experiment.startTime = new Date();
      }
      if (status === "completed" || status === "failed") {
        experiment.endTime = new Date();
      }
    }
  }
  
  async updateExperimentMetrics(id: number, metrics: any): Promise<void> {
    const experiment = this.experiments.get(id);
    if (experiment) {
      experiment.metrics = metrics;
    }
  }
  
  // Metrics operations
  async createMetric(insertMetric: InsertMetric): Promise<Metric> {
    const metric: Metric = {
      ...insertMetric,
      id: this.currentId++,
      timestamp: new Date(),
    };
    this.metricsMap.set(metric.id, metric);
    return metric;
  }
  
  async getMetricsByExperiment(experimentId: number): Promise<Metric[]> {
    return Array.from(this.metricsMap.values()).filter(m => m.experimentId === experimentId);
  }
  
  async getRecentMetrics(limit = 100): Promise<Metric[]> {
    return Array.from(this.metricsMap.values())
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }
  
  // Communication pattern operations
  async createCommunicationPattern(insertPattern: InsertCommunicationPattern): Promise<CommunicationPattern> {
    const pattern: CommunicationPattern = {
      ...insertPattern,
      id: this.currentId++,
      lastCommunication: new Date(),
    };
    const key = `${pattern.fromAgentId}-${pattern.toAgentId}`;
    this.communicationPatterns.set(key, pattern);
    return pattern;
  }
  
  async updateCommunicationPattern(fromAgentId: string, toAgentId: string, frequency: number, efficiency: number): Promise<void> {
    const key = `${fromAgentId}-${toAgentId}`;
    const pattern = this.communicationPatterns.get(key);
    if (pattern) {
      pattern.frequency = frequency;
      pattern.efficiency = efficiency;
      pattern.lastCommunication = new Date();
    }
  }
  
  async getCommunicationPatterns(): Promise<CommunicationPattern[]> {
    return Array.from(this.communicationPatterns.values());
  }
  
  // Complex queries
  async getAgentGridData(): Promise<AgentGridData> {
    const agents = await this.getAllAgents();
    const communicationPatterns = await this.getCommunicationPatterns();
    const activeMessages = await this.getUnprocessedMessages();
    const recentBreakthroughs = await this.getRecentBreakthroughs(10);
    
    return {
      agents,
      communicationPatterns,
      activeMessages,
      recentBreakthroughs,
    };
  }
  
  async getTrainingStatus(experimentId: number): Promise<TrainingStatus | undefined> {
    const experiment = await this.getExperiment(experimentId);
    if (!experiment) return undefined;
    
    const recentMetrics = await this.getMetricsByExperiment(experimentId);
    const episodeMetrics = recentMetrics.filter(m => m.metricType === "episode");
    const stepMetrics = recentMetrics.filter(m => m.metricType === "step");
    
    const currentEpisode = episodeMetrics.length > 0 ? Math.max(...episodeMetrics.map(m => m.episode)) : 0;
    const currentStep = stepMetrics.length > 0 ? Math.max(...stepMetrics.map(m => m.step)) : 0;
    
    return {
      experiment,
      currentEpisode,
      currentStep,
      recentMetrics: recentMetrics.slice(-50),
      isRunning: experiment.status === "running",
    };
  }
  
  async getMemoryState(): Promise<MemoryState> {
    const vectors = await this.getAllMemoryVectors();
    const total = 1000; // simulated memory capacity
    const used = vectors.length;
    const efficiency = used > 0 ? vectors.reduce((sum, v) => sum + v.accessCount, 0) / used : 0;
    
    const recentAccess = vectors
      .sort((a, b) => b.lastAccessed.getTime() - a.lastAccessed.getTime())
      .slice(0, 10)
      .map(v => ({
        vectorId: v.vectorId,
        agentId: v.coordinates || "unknown",
        timestamp: v.lastAccessed.toISOString(),
      }));
    
    return {
      vectors,
      usage: { total, used, efficiency },
      recentAccess,
    };
  }
}

export const storage = new MemStorage();
