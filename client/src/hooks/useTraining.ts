import { useState, useCallback, useEffect } from "react";
import { TrainingEngine } from "@/lib/training";
import { TrainingConfig, TrainingStatus, TrainingMetrics } from "@/types/training";

const defaultConfig: TrainingConfig = {
  maxEpisodes: 1000,
  stepsPerEpisode: 200,
  learningRate: 0.001,
  batchSize: 32,
  replayBufferSize: 10000,
  explorationRate: 0.1,
  communicationRange: 2,
  breakthroughThreshold: 0.8,
  memorySize: 1000,
  coordinatorRatio: 0.1
};

export function useTraining(experimentId?: string) {
  const [engine, setEngine] = useState<TrainingEngine | null>(null);
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const initializeTraining = useCallback((newConfig?: Partial<TrainingConfig>) => {
    const finalConfig = newConfig ? { ...config, ...newConfig } : config;
    const newEngine = new TrainingEngine(finalConfig);
    setEngine(newEngine);
    setConfig(finalConfig);
    setStatus(newEngine.getStatus());
    setMetrics([]);
  }, [config]);

  const startTraining = useCallback(async () => {
    if (!engine || !experimentId) return;

    setIsRunning(true);
    try {
      await engine.startTraining(experimentId);
    } catch (error) {
      console.error("Training failed:", error);
    } finally {
      setIsRunning(false);
    }
  }, [engine, experimentId]);

  const stopTraining = useCallback(() => {
    if (!engine) return;
    engine.stopTraining();
    setIsRunning(false);
  }, [engine]);

  const pauseTraining = useCallback(() => {
    if (!engine) return;
    engine.pauseTraining();
    setIsRunning(false);
  }, [engine]);

  const resumeTraining = useCallback(() => {
    if (!engine) return;
    engine.resumeTraining();
    setIsRunning(true);
  }, [engine]);

  const updateConfig = useCallback((newConfig: Partial<TrainingConfig>) => {
    if (!engine) return;
    try {
      engine.updateConfig(newConfig);
      setConfig({ ...config, ...newConfig });
    } catch (error) {
      console.error("Failed to update config:", error);
    }
  }, [engine, config]);

  const exportMetrics = useCallback(() => {
    if (!engine) return "";
    return engine.exportMetrics();
  }, [engine]);

  const importMetrics = useCallback((metricsJson: string) => {
    if (!engine) return;
    try {
      engine.importMetrics(metricsJson);
      setMetrics(engine.getMetrics());
    } catch (error) {
      console.error("Failed to import metrics:", error);
    }
  }, [engine]);

  const getLatestMetrics = useCallback(() => {
    if (!engine) return null;
    return engine.getLatestMetrics();
  }, [engine]);

  const getAverageMetrics = useCallback((windowSize: number = 100) => {
    if (!engine) return null;
    return engine.getAverageMetrics(windowSize);
  }, [engine]);

  // Update status and metrics periodically when training is running
  useEffect(() => {
    if (!engine || !isRunning) return;

    const interval = setInterval(() => {
      const currentStatus = engine.getStatus();
      const currentMetrics = engine.getMetrics();
      
      setStatus(currentStatus);
      setMetrics(currentMetrics);
      
      // Check if training completed
      if (currentStatus.status === 'completed' || currentStatus.status === 'failed') {
        setIsRunning(false);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [engine, isRunning]);

  return {
    engine,
    config,
    status,
    metrics,
    isRunning,
    initializeTraining,
    startTraining,
    stopTraining,
    pauseTraining,
    resumeTraining,
    updateConfig,
    exportMetrics,
    importMetrics,
    getLatestMetrics,
    getAverageMetrics
  };
}
