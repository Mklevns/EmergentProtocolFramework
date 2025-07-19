import { drizzle } from "drizzle-orm/node-postgres";
import pkg from "pg";
const { Pool } = pkg;
import { eq, desc, and, gt, asc, sql } from "drizzle-orm";
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
import { IStorage } from "./storage";

export class DatabaseStorage implements IStorage {
  private db;
  private pool;

  constructor() {
    if (!process.env.DATABASE_URL) {
      throw new Error("DATABASE_URL environment variable is required");
    }

    this.pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    this.db = drizzle(this.pool);
    console.log("Database storage initialized with direct PostgreSQL connection");
  }

  // Agent operations
  async createAgent(agent: InsertAgent): Promise<Agent> {
    try {
      const [result] = await this.db.insert(agents)
        .values(agent)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create agent:", error);
      throw error;
    }
  }

  async getAgent(agentId: string): Promise<Agent | undefined> {
    try {
      const [result] = await this.db.select()
        .from(agents)
        .where(eq(agents.agentId, agentId));
      return result;
    } catch (error) {
      console.error("Failed to get agent:", error);
      return undefined;
    }
  }

  async getAllAgents(): Promise<Agent[]> {
    try {
      return await this.db.select().from(agents).orderBy(asc(agents.agentId));
    } catch (error) {
      console.error("Failed to get all agents:", error);
      return [];
    }
  }

  async updateAgentStatus(agentId: string, status: string): Promise<void> {
    try {
      await this.db.update(agents)
        .set({ status, updatedAt: new Date() })
        .where(eq(agents.agentId, agentId));
    } catch (error) {
      console.error("Failed to update agent status:", error);
      throw error;
    }
  }

  async getAgentsByType(type: string): Promise<Agent[]> {
    try {
      return await this.db.select()
        .from(agents)
        .where(eq(agents.type, type));
    } catch (error) {
      console.error("Failed to get agents by type:", error);
      return [];
    }
  }

  // Message operations
  async createMessage(message: InsertMessage): Promise<Message> {
    try {
      const [result] = await this.db.insert(messages)
        .values(message)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create message:", error);
      throw error;
    }
  }

  async getMessagesByAgent(agentId: string): Promise<Message[]> {
    try {
      return await this.db.select()
        .from(messages)
        .where(
          and(
            eq(messages.fromAgentId, agentId),
            eq(messages.toAgentId, agentId)
          )
        )
        .orderBy(desc(messages.timestamp));
    } catch (error) {
      console.error("Failed to get messages by agent:", error);
      return [];
    }
  }

  async getUnprocessedMessages(): Promise<Message[]> {
    try {
      return await this.db.select()
        .from(messages)
        .where(eq(messages.isProcessed, false))
        .orderBy(asc(messages.timestamp));
    } catch (error) {
      console.error("Failed to get unprocessed messages:", error);
      return [];
    }
  }

  async markMessageProcessed(messageId: number): Promise<void> {
    try {
      await this.db.update(messages)
        .set({ isProcessed: true })
        .where(eq(messages.id, messageId));
    } catch (error) {
      console.error("Failed to mark message processed:", error);
      throw error;
    }
  }

  async getRecentMessages(limit = 50): Promise<Message[]> {
    try {
      return await this.db.select()
        .from(messages)
        .orderBy(desc(messages.timestamp))
        .limit(limit);
    } catch (error) {
      console.error("Failed to get recent messages:", error);
      return [];
    }
  }

  // Memory operations
  async createMemoryVector(vector: InsertMemoryVector): Promise<MemoryVector> {
    try {
      const [result] = await this.db.insert(memoryVectors)
        .values(vector)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create memory vector:", error);
      throw error;
    }
  }

  async getMemoryVector(vectorId: string): Promise<MemoryVector | undefined> {
    try {
      const [result] = await this.db.select()
        .from(memoryVectors)
        .where(eq(memoryVectors.vectorId, vectorId));
      return result;
    } catch (error) {
      console.error("Failed to get memory vector:", error);
      return undefined;
    }
  }

  async getAllMemoryVectors(): Promise<MemoryVector[]> {
    try {
      return await this.db.select().from(memoryVectors);
    } catch (error) {
      console.error("Failed to get all memory vectors:", error);
      return [];
    }
  }

  async updateMemoryAccess(vectorId: string): Promise<void> {
    try {
      await this.db.update(memoryVectors)
        .set({ 
          accessCount: sql`${memoryVectors.accessCount} + 1`,
          lastAccessed: new Date() 
        })
        .where(eq(memoryVectors.vectorId, vectorId));
    } catch (error) {
      console.error("Failed to update memory access:", error);
      throw error;
    }
  }

  async getMemoryByType(type: string): Promise<MemoryVector[]> {
    try {
      return await this.db.select()
        .from(memoryVectors)
        .where(eq(memoryVectors.vectorType, type));
    } catch (error) {
      console.error("Failed to get memory by type:", error);
      return [];
    }
  }

  // Breakthrough operations
  async createBreakthrough(breakthrough: InsertBreakthrough): Promise<Breakthrough> {
    try {
      const [result] = await this.db.insert(breakthroughs)
        .values(breakthrough)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create breakthrough:", error);
      throw error;
    }
  }

  async getBreakthroughsByAgent(agentId: string): Promise<Breakthrough[]> {
    try {
      return await this.db.select()
        .from(breakthroughs)
        .where(eq(breakthroughs.agentId, agentId))
        .orderBy(desc(breakthroughs.timestamp));
    } catch (error) {
      console.error("Failed to get breakthroughs by agent:", error);
      return [];
    }
  }

  async getRecentBreakthroughs(limit = 20): Promise<Breakthrough[]> {
    try {
      return await this.db.select()
        .from(breakthroughs)
        .orderBy(desc(breakthroughs.timestamp))
        .limit(limit);
    } catch (error) {
      console.error("Failed to get recent breakthroughs:", error);
      return [];
    }
  }

  async markBreakthroughShared(breakthroughId: number): Promise<void> {
    try {
      await this.db.update(breakthroughs)
        .set({ wasShared: true })
        .where(eq(breakthroughs.id, breakthroughId));
    } catch (error) {
      console.error("Failed to mark breakthrough shared:", error);
      throw error;
    }
  }

  // Experiment operations
  async createExperiment(experiment: InsertExperiment): Promise<Experiment> {
    try {
      const [result] = await this.db.insert(experiments)
        .values(experiment)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create experiment:", error);
      throw error;
    }
  }

  async getExperiment(id: number): Promise<Experiment | undefined> {
    try {
      const [result] = await this.db.select()
        .from(experiments)
        .where(eq(experiments.id, id));
      return result;
    } catch (error) {
      console.error("Failed to get experiment:", error);
      return undefined;
    }
  }

  async getAllExperiments(): Promise<Experiment[]> {
    try {
      return await this.db.select()
        .from(experiments)
        .orderBy(desc(experiments.createdAt));
    } catch (error) {
      console.error("Failed to get all experiments:", error);
      return [];
    }
  }

  async updateExperimentStatus(id: number, status: string): Promise<void> {
    try {
      const updateData: any = { status };
      
      if (status === "running") {
        updateData.startTime = new Date();
      } else if (status === "completed" || status === "failed") {
        updateData.endTime = new Date();
      }

      await this.db.update(experiments)
        .set(updateData)
        .where(eq(experiments.id, id));
    } catch (error) {
      console.error("Failed to update experiment status:", error);
      throw error;
    }
  }

  async updateExperimentMetrics(id: number, metricsData: any): Promise<void> {
    try {
      await this.db.update(experiments)
        .set({ metrics: metricsData })
        .where(eq(experiments.id, id));
    } catch (error) {
      console.error("Failed to update experiment metrics:", error);
      throw error;
    }
  }

  // Metrics operations
  async createMetric(metric: InsertMetric): Promise<Metric> {
    try {
      const [result] = await this.db.insert(metrics)
        .values(metric)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create metric:", error);
      throw error;
    }
  }

  async getMetricsByExperiment(experimentId: number): Promise<Metric[]> {
    try {
      return await this.db.select()
        .from(metrics)
        .where(eq(metrics.experimentId, experimentId))
        .orderBy(desc(metrics.timestamp));
    } catch (error) {
      console.error("Failed to get metrics by experiment:", error);
      return [];
    }
  }

  async getRecentMetrics(limit = 100): Promise<Metric[]> {
    try {
      return await this.db.select()
        .from(metrics)
        .orderBy(desc(metrics.timestamp))
        .limit(limit);
    } catch (error) {
      console.error("Failed to get recent metrics:", error);
      return [];
    }
  }

  // Communication pattern operations
  async createCommunicationPattern(pattern: InsertCommunicationPattern): Promise<CommunicationPattern> {
    try {
      const [result] = await this.db.insert(communicationPatterns)
        .values(pattern)
        .returning();
      return result;
    } catch (error) {
      console.error("Failed to create communication pattern:", error);
      throw error;
    }
  }

  async updateCommunicationPattern(fromAgentId: string, toAgentId: string, frequency: number, efficiency: number): Promise<void> {
    try {
      // Try to update existing pattern first
      const existingPattern = await this.db.select()
        .from(communicationPatterns)
        .where(
          and(
            eq(communicationPatterns.fromAgentId, fromAgentId),
            eq(communicationPatterns.toAgentId, toAgentId)
          )
        )
        .limit(1);

      if (existingPattern.length > 0) {
        await this.db.update(communicationPatterns)
          .set({ 
            frequency, 
            efficiency, 
            lastCommunication: new Date() 
          })
          .where(eq(communicationPatterns.id, existingPattern[0].id));
      } else {
        // Create new pattern if doesn't exist
        await this.createCommunicationPattern({
          fromAgentId,
          toAgentId,
          frequency,
          efficiency,
        });
      }
    } catch (error) {
      console.error("Failed to update communication pattern:", error);
      throw error;
    }
  }

  async getCommunicationPatterns(): Promise<CommunicationPattern[]> {
    try {
      return await this.db.select()
        .from(communicationPatterns)
        .orderBy(desc(communicationPatterns.lastCommunication));
    } catch (error) {
      console.error("Failed to get communication patterns:", error);
      return [];
    }
  }

  // Complex queries
  async getAgentGridData(): Promise<AgentGridData> {
    try {
      const agentsList = await this.getAllAgents();
      const communicationPatternsList = await this.getCommunicationPatterns();
      const activeMessages = await this.getUnprocessedMessages();
      const recentBreakthroughsList = await this.getRecentBreakthroughs(10);
      
      return {
        agents: agentsList,
        communicationPatterns: communicationPatternsList,
        activeMessages,
        recentBreakthroughs: recentBreakthroughsList,
      };
    } catch (error) {
      console.error("Failed to get agent grid data:", error);
      return {
        agents: [],
        communicationPatterns: [],
        activeMessages: [],
        recentBreakthroughs: [],
      };
    }
  }

  async getTrainingStatus(experimentId: number): Promise<TrainingStatus | undefined> {
    try {
      const experiment = await this.getExperiment(experimentId);
      if (!experiment) return undefined;
      
      const recentMetricsList = await this.getMetricsByExperiment(experimentId);
      const episodeMetrics = recentMetricsList.filter(m => m.metricType === "episode");
      const stepMetrics = recentMetricsList.filter(m => m.metricType === "step");
      
      const currentEpisode = episodeMetrics.length > 0 ? Math.max(...episodeMetrics.map(m => m.episode)) : 0;
      const currentStep = stepMetrics.length > 0 ? Math.max(...stepMetrics.map(m => m.step)) : 0;
      
      return {
        experiment,
        currentEpisode,
        currentStep,
        recentMetrics: recentMetricsList.slice(-50),
        isRunning: experiment.status === "running",
      };
    } catch (error) {
      console.error("Failed to get training status:", error);
      return undefined;
    }
  }

  async getMemoryState(): Promise<MemoryState> {
    try {
      const vectors = await this.getAllMemoryVectors();
      const total = 1000; // simulated memory capacity
      const used = vectors.length;
      const efficiency = used > 0 ? vectors.reduce((sum, v) => sum + (v.accessCount || 0), 0) / used : 0;
      
      const recentAccess = vectors
        .sort((a, b) => (b.lastAccessed?.getTime() || 0) - (a.lastAccessed?.getTime() || 0))
        .slice(0, 10)
        .map(v => ({
          vectorId: v.vectorId,
          agentId: v.coordinates || "unknown",
          timestamp: (v.lastAccessed || new Date()).toISOString(),
        }));
      
      return {
        vectors,
        usage: { total, used, efficiency },
        recentAccess,
      };
    } catch (error) {
      console.error("Failed to get memory state:", error);
      return {
        vectors: [],
        usage: { total: 1000, used: 0, efficiency: 0 },
        recentAccess: [],
      };
    }
  }

  async clearAll(): Promise<void> {
    try {
      // Clear all tables in the correct order (respecting foreign keys)
      await this.db.delete(metrics);
      await this.db.delete(communicationPatterns);
      await this.db.delete(messages);
      await this.db.delete(breakthroughs);
      await this.db.delete(memoryVectors);
      await this.db.delete(agents);
      await this.db.delete(experiments);
      console.log("All database tables cleared");
    } catch (error) {
      console.error("Failed to clear all data:", error);
      throw error;
    }
  }

  async close(): Promise<void> {
    await this.pool.end();
  }
}