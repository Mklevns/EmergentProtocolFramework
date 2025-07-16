import { useState, useEffect, useCallback } from "react";
import { Grid3D } from "@/lib/grid";
import { Agent, AgentState, AgentNetwork } from "@/types/agents";

export function useAgentGrid(experimentId?: string) {
  const [grid, setGrid] = useState<Grid3D | null>(null);
  const [agents, setAgents] = useState<Map<string, AgentState>>(new Map());
  const [network, setNetwork] = useState<AgentNetwork | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  const initializeGrid = useCallback(() => {
    const newGrid = new Grid3D();
    const standardLayout = newGrid.generateStandardLayout();
    
    // Create agents in the grid
    standardLayout.forEach(agentData => {
      newGrid.createAgent(
        agentData.id,
        agentData.gridPosition,
        agentData.agentType,
        agentData.region
      );
    });

    setGrid(newGrid);
    setAgents(newGrid.getAllAgents());
    setNetwork(newGrid.getAgentNetwork());
    setIsInitialized(true);
  }, []);

  const updateAgentActivity = useCallback((agentId: string, activity: number) => {
    if (!grid) return;

    grid.updateAgentActivity(agentId, activity);
    setAgents(new Map(grid.getAllAgents()));
    setNetwork(grid.getAgentNetwork());
  }, [grid]);

  const getAgentNeighbors = useCallback((agentId: string): string[] => {
    if (!grid) return [];
    return grid.getNeighbors(agentId);
  }, [grid]);

  const getAgentsByRegion = useCallback((region: number): string[] => {
    if (!grid) return [];
    return grid.getAgentsByRegion(region);
  }, [grid]);

  const getCoordinators = useCallback((): string[] => {
    if (!grid) return [];
    return grid.getCoordinators();
  }, [grid]);

  const getGridDimensions = useCallback(() => {
    if (!grid) return { x: 0, y: 0, z: 0 };
    return grid.getGridDimensions();
  }, [grid]);

  const getAgentCount = useCallback(() => {
    return agents.size;
  }, [agents]);

  const getActiveAgentCount = useCallback(() => {
    return Array.from(agents.values()).filter(agent => agent.isActive).length;
  }, [agents]);

  const getRegionStats = useCallback(() => {
    const stats = new Map<number, { total: number; active: number; coordinators: number }>();
    
    for (const agent of agents.values()) {
      const region = agent.region;
      const current = stats.get(region) || { total: 0, active: 0, coordinators: 0 };
      
      current.total++;
      if (agent.isActive) current.active++;
      if (agent.type === 'coordinator') current.coordinators++;
      
      stats.set(region, current);
    }

    return stats;
  }, [agents]);

  useEffect(() => {
    if (!isInitialized) {
      initializeGrid();
    }
  }, [isInitialized, initializeGrid]);

  return {
    grid,
    agents,
    network,
    isInitialized,
    initializeGrid,
    updateAgentActivity,
    getAgentNeighbors,
    getAgentsByRegion,
    getCoordinators,
    getGridDimensions,
    getAgentCount,
    getActiveAgentCount,
    getRegionStats
  };
}
