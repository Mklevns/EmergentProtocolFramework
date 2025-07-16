import { GridPosition, Agent, AgentState, AgentNetwork } from "@/types/agents";

export class Grid3D {
  private dimensions: { x: number; y: number; z: number };
  private agents: Map<string, AgentState>;
  private coordinators: string[];
  private regions: Map<number, string[]>;

  constructor(dimensions = { x: 4, y: 3, z: 3 }) {
    this.dimensions = dimensions;
    this.agents = new Map();
    this.coordinators = [];
    this.regions = new Map();
    this.initializeRegions();
  }

  private initializeRegions() {
    // Initialize three regions for coordinators
    this.regions.set(0, []); // Region 0: x=0-1
    this.regions.set(1, []); // Region 1: x=2-3
    this.regions.set(2, []); // Region 2: y=0-1, z=0-1
  }

  public createAgent(
    id: string,
    position: GridPosition,
    type: 'regular' | 'coordinator',
    region: number
  ): AgentState {
    const agent: AgentState = {
      id,
      position,
      type,
      region,
      isActive: true,
      neighbors: [],
      communicationRange: type === 'coordinator' ? 2 : 1,
      lastActivity: new Date(),
      memoryUsage: 0,
      performance: 0
    };

    this.agents.set(id, agent);
    
    if (type === 'coordinator') {
      this.coordinators.push(id);
    }

    this.regions.get(region)?.push(id);
    this.updateNeighbors(id);
    
    return agent;
  }

  public updateNeighbors(agentId: string) {
    const agent = this.agents.get(agentId);
    if (!agent) return;

    const neighbors: string[] = [];
    const { x, y, z } = agent.position;
    const range = agent.communicationRange;

    for (const [otherId, other] of this.agents.entries()) {
      if (otherId === agentId) continue;

      const distance = Math.sqrt(
        Math.pow(other.position.x - x, 2) +
        Math.pow(other.position.y - y, 2) +
        Math.pow(other.position.z - z, 2)
      );

      if (distance <= range) {
        neighbors.push(otherId);
      }
    }

    agent.neighbors = neighbors;
  }

  public getNeighbors(agentId: string): string[] {
    const agent = this.agents.get(agentId);
    return agent ? agent.neighbors : [];
  }

  public getAgentsByRegion(region: number): string[] {
    return this.regions.get(region) || [];
  }

  public getCoordinators(): string[] {
    return this.coordinators;
  }

  public getAgentNetwork(): AgentNetwork {
    const connections = new Map<string, string[]>();
    
    for (const [agentId, agent] of this.agents.entries()) {
      connections.set(agentId, agent.neighbors);
    }

    return {
      agents: this.agents,
      connections,
      coordinators: this.coordinators,
      regions: this.regions
    };
  }

  public isValidPosition(position: GridPosition): boolean {
    return (
      position.x >= 0 && position.x < this.dimensions.x &&
      position.y >= 0 && position.y < this.dimensions.y &&
      position.z >= 0 && position.z < this.dimensions.z
    );
  }

  public getPositionKey(position: GridPosition): string {
    return `${position.x},${position.y},${position.z}`;
  }

  public generateStandardLayout(): Agent[] {
    const agents: Agent[] = [];
    let agentId = 0;

    // Place 27 regular agents
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 3; y++) {
        for (let z = 0; z < 3; z++) {
          if (agentId < 27) {
            const region = this.determineRegion({ x, y, z });
            agents.push({
              id: `agent-${agentId}`,
              experimentId: '',
              agentType: 'regular',
              gridPosition: { x, y, z },
              region,
              isActive: true,
              createdAt: new Date()
            });
            agentId++;
          }
        }
      }
    }

    // Place 3 coordinators at strategic positions
    const coordinatorPositions = [
      { x: 1, y: 1, z: 1 }, // Center of region 0
      { x: 2, y: 1, z: 1 }, // Center of region 1
      { x: 3, y: 2, z: 2 }, // Center of region 2
    ];

    coordinatorPositions.forEach((pos, index) => {
      agents.push({
        id: `coordinator-${index}`,
        experimentId: '',
        agentType: 'coordinator',
        gridPosition: pos,
        region: index,
        isActive: true,
        createdAt: new Date()
      });
    });

    return agents;
  }

  private determineRegion(position: GridPosition): number {
    if (position.x <= 1) return 0;
    if (position.x >= 2) return 1;
    return 2;
  }

  public updateAgentActivity(agentId: string, activity: number) {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.lastActivity = new Date();
      agent.performance = activity;
    }
  }

  public getGridDimensions() {
    return this.dimensions;
  }

  public getAllAgents(): Map<string, AgentState> {
    return this.agents;
  }
}
