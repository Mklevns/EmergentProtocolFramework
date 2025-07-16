import { pgTable, text, serial, integer, boolean, timestamp, real, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Agent types and statuses
export const agentTypes = ["regular", "coordinator"] as const;
export const agentStatuses = ["idle", "communicating", "processing", "breakthrough"] as const;

// Agents table
export const agents = pgTable("agents", {
  id: serial("id").primaryKey(),
  agentId: text("agent_id").notNull().unique(),
  type: text("type").notNull(), // "regular" or "coordinator"
  positionX: integer("position_x").notNull(),
  positionY: integer("position_y").notNull(),
  positionZ: integer("position_z").notNull(),
  status: text("status").default("idle"),
  coordinatorId: text("coordinator_id"), // which coordinator oversees this agent
  hiddenDim: integer("hidden_dim").default(256),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Communication messages
export const messages = pgTable("messages", {
  id: serial("id").primaryKey(),
  fromAgentId: text("from_agent_id").notNull(),
  toAgentId: text("to_agent_id").notNull(),
  messageType: text("message_type").notNull(), // "pointer", "broadcast", "breakthrough"
  content: jsonb("content").notNull(), // vectorized content
  memoryPointer: text("memory_pointer"), // pointer to shared memory
  timestamp: timestamp("timestamp").defaultNow(),
  isProcessed: boolean("is_processed").default(false),
});

// Shared memory vectors
export const memoryVectors = pgTable("memory_vectors", {
  id: serial("id").primaryKey(),
  vectorId: text("vector_id").notNull().unique(),
  content: jsonb("content").notNull(), // actual vector data
  vectorType: text("vector_type").notNull(), // "breakthrough", "context", "coordination"
  coordinates: text("coordinates"), // grid coordinates where this was generated
  importance: real("importance").default(0.5),
  accessCount: integer("access_count").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  lastAccessed: timestamp("last_accessed").defaultNow(),
});

// Breakthrough events
export const breakthroughs = pgTable("breakthroughs", {
  id: serial("id").primaryKey(),
  agentId: text("agent_id").notNull(),
  breakthroughType: text("breakthrough_type").notNull(),
  description: text("description"),
  vectorId: text("vector_id"), // associated memory vector
  confidence: real("confidence").default(0.0),
  wasShared: boolean("was_shared").default(false),
  timestamp: timestamp("timestamp").defaultNow(),
});

// Training experiments
export const experiments = pgTable("experiments", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  config: jsonb("config").notNull(),
  status: text("status").default("pending"), // "pending", "running", "completed", "failed"
  startTime: timestamp("start_time"),
  endTime: timestamp("end_time"),
  metrics: jsonb("metrics"),
  createdAt: timestamp("created_at").defaultNow(),
});

// Training metrics
export const metrics = pgTable("metrics", {
  id: serial("id").primaryKey(),
  experimentId: integer("experiment_id").notNull(),
  episode: integer("episode").notNull(),
  step: integer("step").notNull(),
  metricType: text("metric_type").notNull(), // "communication_efficiency", "breakthrough_frequency", etc.
  value: real("value").notNull(),
  agentId: text("agent_id"), // specific agent metric or null for global
  timestamp: timestamp("timestamp").defaultNow(),
});

// Communication patterns
export const communicationPatterns = pgTable("communication_patterns", {
  id: serial("id").primaryKey(),
  fromAgentId: text("from_agent_id").notNull(),
  toAgentId: text("to_agent_id").notNull(),
  frequency: integer("frequency").default(1),
  efficiency: real("efficiency").default(0.0),
  lastCommunication: timestamp("last_communication").defaultNow(),
  patternType: text("pattern_type"), // "hierarchical", "direct", "broadcast"
});

// Insert schemas
export const insertAgentSchema = createInsertSchema(agents).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertMessageSchema = createInsertSchema(messages).omit({
  id: true,
  timestamp: true,
  isProcessed: true,
});

export const insertMemoryVectorSchema = createInsertSchema(memoryVectors).omit({
  id: true,
  accessCount: true,
  createdAt: true,
  lastAccessed: true,
});

export const insertBreakthroughSchema = createInsertSchema(breakthroughs).omit({
  id: true,
  timestamp: true,
});

export const insertExperimentSchema = createInsertSchema(experiments).omit({
  id: true,
  createdAt: true,
  startTime: true,
  endTime: true,
});

export const insertMetricSchema = createInsertSchema(metrics).omit({
  id: true,
  timestamp: true,
});

export const insertCommunicationPatternSchema = createInsertSchema(communicationPatterns).omit({
  id: true,
  lastCommunication: true,
});

// Types
export type Agent = typeof agents.$inferSelect;
export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type Message = typeof messages.$inferSelect;
export type InsertMessage = z.infer<typeof insertMessageSchema>;
export type MemoryVector = typeof memoryVectors.$inferSelect;
export type InsertMemoryVector = z.infer<typeof insertMemoryVectorSchema>;
export type Breakthrough = typeof breakthroughs.$inferSelect;
export type InsertBreakthrough = z.infer<typeof insertBreakthroughSchema>;
export type Experiment = typeof experiments.$inferSelect;
export type InsertExperiment = z.infer<typeof insertExperimentSchema>;
export type Metric = typeof metrics.$inferSelect;
export type InsertMetric = z.infer<typeof insertMetricSchema>;
export type CommunicationPattern = typeof communicationPatterns.$inferSelect;
export type InsertCommunicationPattern = z.infer<typeof insertCommunicationPatternSchema>;

// API response types
export type AgentGridData = {
  agents: Agent[];
  communicationPatterns: CommunicationPattern[];
  activeMessages: Message[];
  recentBreakthroughs: Breakthrough[];
};

export type TrainingStatus = {
  experiment: Experiment;
  currentEpisode: number;
  currentStep: number;
  recentMetrics: Metric[];
  isRunning: boolean;
};

export type MemoryState = {
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
};
