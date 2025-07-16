export interface TrainingConfig {
  maxEpisodes: number;
  stepsPerEpisode: number;
  learningRate: number;
  batchSize: number;
  replayBufferSize: number;
  explorationRate: number;
  communicationRange: number;
  breakthroughThreshold: number;
  memorySize: number;
  coordinatorRatio: number;
}

export interface TrainingMetrics {
  episode: number;
  step: number;
  reward: number;
  communicationEfficiency: number;
  memoryUtilization: number;
  breakthroughFrequency: number;
  coordinationSuccess: number;
  timestamp: Date;
}

export interface TrainingStatus {
  experimentId: string;
  status: 'created' | 'running' | 'completed' | 'failed';
  currentEpisode: number;
  totalEpisodes: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  metrics: TrainingMetrics;
}
