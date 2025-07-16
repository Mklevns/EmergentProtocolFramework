import { VectorData } from "@/types/memory";

export interface BreakthroughEvent {
  id: string;
  agentId: string;
  experimentId: string;
  breakthroughType: string;
  vectorData: VectorData;
  context: Record<string, any>;
  significance: number;
  timestamp: Date;
}

export class BreakthroughDetector {
  private thresholds: Map<string, number>;
  private patterns: Map<string, any>;
  private history: BreakthroughEvent[];

  constructor() {
    this.thresholds = new Map([
      ['problem_solving', 0.8],
      ['communication_efficiency', 0.7],
      ['coordination_success', 0.9],
      ['pattern_recognition', 0.75],
      ['adaptation', 0.85]
    ]);
    
    this.patterns = new Map();
    this.history = [];
  }

  public detectBreakthrough(
    agentId: string,
    experimentId: string,
    performanceData: Record<string, number>,
    context: Record<string, any>
  ): BreakthroughEvent | null {
    
    const breakthroughType = this.analyzePerformance(performanceData);
    
    if (breakthroughType) {
      const significance = this.calculateSignificance(performanceData, breakthroughType);
      
      if (significance >= (this.thresholds.get(breakthroughType) || 0.8)) {
        const vectorData = this.vectorizeBreakthrough(performanceData, context);
        
        const breakthrough: BreakthroughEvent = {
          id: this.generateId(),
          agentId,
          experimentId,
          breakthroughType,
          vectorData,
          context,
          significance,
          timestamp: new Date()
        };

        this.history.push(breakthrough);
        this.updatePatterns(breakthrough);
        
        return breakthrough;
      }
    }

    return null;
  }

  public analyzeBreakthroughPatterns(): {
    frequency: number;
    types: Map<string, number>;
    agentContributions: Map<string, number>;
    emergenceRate: number;
  } {
    const frequency = this.history.length;
    const types = new Map<string, number>();
    const agentContributions = new Map<string, number>();

    for (const breakthrough of this.history) {
      types.set(breakthrough.breakthroughType, 
        (types.get(breakthrough.breakthroughType) || 0) + 1);
      
      agentContributions.set(breakthrough.agentId,
        (agentContributions.get(breakthrough.agentId) || 0) + breakthrough.significance);
    }

    // Calculate emergence rate (breakthroughs per hour)
    const timeSpan = this.history.length > 0 
      ? Date.now() - this.history[0].timestamp.getTime()
      : 1;
    const emergenceRate = (frequency / timeSpan) * 3600000; // per hour

    return {
      frequency,
      types,
      agentContributions,
      emergenceRate
    };
  }

  public predictBreakthrough(
    agentId: string,
    currentPerformance: Record<string, number>,
    historicalData: Record<string, number>[]
  ): {
    probability: number;
    expectedType: string;
    timeToBreakthrough: number;
  } {
    const agentHistory = this.history.filter(b => b.agentId === agentId);
    const performanceTrend = this.calculateTrend(historicalData);
    
    let maxProbability = 0;
    let expectedType = 'unknown';

    for (const [type, threshold] of this.thresholds.entries()) {
      const currentValue = currentPerformance[type] || 0;
      const trend = performanceTrend[type] || 0;
      
      const probability = Math.min(
        (currentValue + trend) / threshold,
        1.0
      );

      if (probability > maxProbability) {
        maxProbability = probability;
        expectedType = type;
      }
    }

    // Estimate time to breakthrough based on current trend
    const timeToBreakthrough = this.estimateTimeToBreakthrough(
      currentPerformance,
      performanceTrend,
      expectedType
    );

    return {
      probability: maxProbability,
      expectedType,
      timeToBreakthrough
    };
  }

  private analyzePerformance(performanceData: Record<string, number>): string | null {
    for (const [type, value] of Object.entries(performanceData)) {
      const threshold = this.thresholds.get(type);
      if (threshold && value >= threshold) {
        return type;
      }
    }
    return null;
  }

  private calculateSignificance(
    performanceData: Record<string, number>,
    breakthroughType: string
  ): number {
    const baseValue = performanceData[breakthroughType] || 0;
    const threshold = this.thresholds.get(breakthroughType) || 0.8;
    
    // Normalize significance based on how much the performance exceeds threshold
    const rawSignificance = (baseValue - threshold) / (1 - threshold);
    
    // Apply contextual multipliers
    const contextMultiplier = this.calculateContextMultiplier(performanceData);
    
    return Math.min(rawSignificance * contextMultiplier, 1.0);
  }

  private calculateContextMultiplier(performanceData: Record<string, number>): number {
    // Higher multiplier for simultaneous improvements in multiple areas
    const activeAreas = Object.values(performanceData).filter(v => v > 0.5).length;
    return Math.min(1 + (activeAreas - 1) * 0.2, 2.0);
  }

  private vectorizeBreakthrough(
    performanceData: Record<string, number>,
    context: Record<string, any>
  ): VectorData {
    const embedding = this.createEmbedding(performanceData, context);
    
    return {
      embedding,
      metadata: {
        performance: performanceData,
        context,
        timestamp: new Date(),
        vectorDimension: embedding.length
      },
      timestamp: new Date()
    };
  }

  private createEmbedding(
    performanceData: Record<string, number>,
    context: Record<string, any>
  ): number[] {
    const embedding = new Array(256).fill(0);
    
    // Encode performance data
    const perfKeys = Object.keys(performanceData);
    perfKeys.forEach((key, index) => {
      const value = performanceData[key];
      const baseIndex = (index * 10) % 256;
      
      for (let i = 0; i < 10; i++) {
        embedding[baseIndex + i] = Math.sin(value * (i + 1));
      }
    });

    // Encode context
    const contextStr = JSON.stringify(context);
    for (let i = 0; i < contextStr.length && i < 100; i++) {
      const char = contextStr.charCodeAt(i);
      embedding[156 + i] = (char / 255) * 2 - 1;
    }

    return embedding;
  }

  private updatePatterns(breakthrough: BreakthroughEvent) {
    const typePattern = this.patterns.get(breakthrough.breakthroughType) || {
      count: 0,
      averageSignificance: 0,
      contexts: []
    };

    typePattern.count++;
    typePattern.averageSignificance = 
      (typePattern.averageSignificance * (typePattern.count - 1) + breakthrough.significance) / typePattern.count;
    typePattern.contexts.push(breakthrough.context);

    this.patterns.set(breakthrough.breakthroughType, typePattern);
  }

  private calculateTrend(historicalData: Record<string, number>[]): Record<string, number> {
    if (historicalData.length < 2) return {};

    const trends: Record<string, number> = {};
    const recent = historicalData[historicalData.length - 1];
    const previous = historicalData[historicalData.length - 2];

    for (const key of Object.keys(recent)) {
      trends[key] = recent[key] - (previous[key] || 0);
    }

    return trends;
  }

  private estimateTimeToBreakthrough(
    currentPerformance: Record<string, number>,
    performanceTrend: Record<string, number>,
    expectedType: string
  ): number {
    const current = currentPerformance[expectedType] || 0;
    const trend = performanceTrend[expectedType] || 0;
    const threshold = this.thresholds.get(expectedType) || 0.8;

    if (trend <= 0) return Infinity;

    const stepsNeeded = (threshold - current) / trend;
    return Math.max(stepsNeeded, 0);
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  public getBreakthroughHistory(): BreakthroughEvent[] {
    return this.history;
  }

  public getBreakthroughPatterns(): Map<string, any> {
    return this.patterns;
  }

  public updateThreshold(type: string, threshold: number) {
    this.thresholds.set(type, threshold);
  }

  public getThresholds(): Map<string, number> {
    return this.thresholds;
  }
}
