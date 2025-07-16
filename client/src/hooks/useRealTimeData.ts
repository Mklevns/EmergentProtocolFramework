import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";

interface RealTimeDataOptions {
  experimentId?: string;
  updateInterval?: number;
  enabled?: boolean;
}

interface RealTimeMetrics {
  timestamp: Date;
  agentActivity: Map<string, number>;
  communicationFlow: Array<{
    from: string;
    to: string;
    count: number;
    type: string;
  }>;
  memoryUsage: {
    total: number;
    active: number;
    utilization: number;
  };
  breakthroughs: Array<{
    agentId: string;
    type: string;
    significance: number;
    timestamp: Date;
  }>;
  coordinationSuccess: number;
}

export function useRealTimeData(options: RealTimeDataOptions = {}) {
  const {
    experimentId,
    updateInterval = 1000,
    enabled = true
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [metrics, setMetrics] = useState<RealTimeMetrics | null>(null);

  // Fetch experiment data
  const { data: experiment } = useQuery({
    queryKey: ["/api/experiments", experimentId],
    enabled: enabled && !!experimentId,
    staleTime: 5000,
  });

  // Fetch agents data
  const { data: agents } = useQuery({
    queryKey: ["/api/experiments", experimentId, "agents"],
    enabled: enabled && !!experimentId,
    refetchInterval: updateInterval,
  });

  // Fetch communications data
  const { data: communications } = useQuery({
    queryKey: ["/api/experiments", experimentId, "communications"],
    enabled: enabled && !!experimentId,
    refetchInterval: updateInterval,
  });

  // Fetch memory data
  const { data: memory } = useQuery({
    queryKey: ["/api/experiments", experimentId, "memory"],
    enabled: enabled && !!experimentId,
    refetchInterval: updateInterval,
  });

  // Fetch breakthroughs data
  const { data: breakthroughs } = useQuery({
    queryKey: ["/api/experiments", experimentId, "breakthroughs"],
    enabled: enabled && !!experimentId,
    refetchInterval: updateInterval,
  });

  // Fetch training metrics
  const { data: trainingMetrics } = useQuery({
    queryKey: ["/api/experiments", experimentId, "metrics"],
    enabled: enabled && !!experimentId,
    refetchInterval: updateInterval,
  });

  // Process real-time data
  const processRealTimeData = useCallback(() => {
    if (!agents || !communications || !memory || !breakthroughs) return;

    const agentActivity = new Map<string, number>();
    
    // Calculate agent activity based on recent communications
    const recentCommunications = communications.filter(
      (comm: any) => new Date(comm.timestamp).getTime() > Date.now() - 60000
    );

    for (const agent of agents) {
      const activity = recentCommunications.filter(
        (comm: any) => comm.fromAgentId === agent.id
      ).length;
      agentActivity.set(agent.id, activity);
    }

    // Process communication flows
    const communicationFlow = new Map<string, { count: number; type: string }>();
    
    for (const comm of recentCommunications) {
      const key = `${comm.fromAgentId}-${comm.toAgentId}`;
      const existing = communicationFlow.get(key);
      
      if (existing) {
        existing.count++;
      } else {
        communicationFlow.set(key, { count: 1, type: comm.messageType });
      }
    }

    const flowArray = Array.from(communicationFlow.entries()).map(([key, value]) => {
      const [from, to] = key.split('-');
      return { from, to, count: value.count, type: value.type };
    });

    // Process memory usage
    const memoryUsage = {
      total: memory.length,
      active: memory.filter((m: any) => m.accessCount > 0).length,
      utilization: memory.reduce((sum: number, m: any) => sum + (m.accessCount || 0), 0) / memory.length || 0
    };

    // Process recent breakthroughs
    const recentBreakthroughs = breakthroughs
      .filter((b: any) => new Date(b.timestamp).getTime() > Date.now() - 300000) // Last 5 minutes
      .map((b: any) => ({
        agentId: b.agentId,
        type: b.breakthroughType,
        significance: b.significance,
        timestamp: new Date(b.timestamp)
      }));

    // Calculate coordination success
    const coordinationSuccess = trainingMetrics && trainingMetrics.length > 0
      ? trainingMetrics[trainingMetrics.length - 1].coordinationSuccess
      : 0;

    const newMetrics: RealTimeMetrics = {
      timestamp: new Date(),
      agentActivity,
      communicationFlow: flowArray,
      memoryUsage,
      breakthroughs: recentBreakthroughs,
      coordinationSuccess
    };

    setMetrics(newMetrics);
    setLastUpdate(new Date());
    setIsConnected(true);
  }, [agents, communications, memory, breakthroughs, trainingMetrics]);

  // Process data when it updates
  useEffect(() => {
    if (enabled && experimentId) {
      processRealTimeData();
    }
  }, [enabled, experimentId, processRealTimeData]);

  // Simulate connection status
  useEffect(() => {
    if (enabled && experimentId) {
      setIsConnected(true);
      const timeout = setTimeout(() => setIsConnected(false), updateInterval * 2);
      return () => clearTimeout(timeout);
    } else {
      setIsConnected(false);
    }
  }, [enabled, experimentId, updateInterval, lastUpdate]);

  const getAgentActivity = useCallback((agentId: string): number => {
    return metrics?.agentActivity.get(agentId) || 0;
  }, [metrics]);

  const getCommunicationFlows = useCallback(() => {
    return metrics?.communicationFlow || [];
  }, [metrics]);

  const getMemoryUtilization = useCallback(() => {
    return metrics?.memoryUsage || { total: 0, active: 0, utilization: 0 };
  }, [metrics]);

  const getRecentBreakthroughs = useCallback(() => {
    return metrics?.breakthroughs || [];
  }, [metrics]);

  const getCoordinationSuccess = useCallback(() => {
    return metrics?.coordinationSuccess || 0;
  }, [metrics]);

  return {
    isConnected,
    lastUpdate,
    metrics,
    getAgentActivity,
    getCommunicationFlows,
    getMemoryUtilization,
    getRecentBreakthroughs,
    getCoordinationSuccess,
    experiment,
    agents,
    communications,
    memory,
    breakthroughs,
    trainingMetrics
  };
}
