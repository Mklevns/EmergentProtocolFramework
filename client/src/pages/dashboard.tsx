import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { Activity, Brain, Network, Zap, MessageSquare, Database, TrendingUp } from "lucide-react";

import { AgentGrid3D } from "@/components/AgentGrid3D";
import { CommunicationFlow } from "@/components/CommunicationFlow";
import { MemoryVisualization } from "@/components/MemoryVisualization";
import { MetricsPanel } from "@/components/MetricsPanel";
import { BreakthroughPanel } from "@/components/BreakthroughPanel";
import { SystemControl } from "@/components/SystemControl";
import { CommunicationMetrics } from "@/components/CommunicationMetrics";
import { useWebSocket } from "@/hooks/useWebSocket";
import { AgentGridData, TrainingStatus } from "@/lib/agent-types";

export default function Dashboard() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  
  // WebSocket for real-time updates
  const { data: wsData, isConnected } = useWebSocket('/api/ws');
  
  // Grid data query
  const { data: gridData, isLoading: gridLoading } = useQuery<AgentGridData>({
    queryKey: ['/api/grid'],
    refetchInterval: 5000,
  });
  
  // Training status query
  const { data: trainingStatus } = useQuery<TrainingStatus>({
    queryKey: ['/api/training/status'],
    refetchInterval: 2000,
  });
  
  // Memory state query
  const { data: memoryState } = useQuery({
    queryKey: ['/api/memory'],
    refetchInterval: 3000,
  });
  
  // Breakthrough data query
  const { data: breakthroughData } = useQuery({
    queryKey: ['/api/breakthroughs'],
    refetchInterval: 5000,
  });
  
  // Communication patterns query
  const { data: commPatterns } = useQuery({
    queryKey: ['/api/communication-patterns'],
    refetchInterval: 4000,
  });
  
  // Real-time metrics
  const { data: metrics } = useQuery({
    queryKey: ['/api/metrics'],
    refetchInterval: 1000,
  });
  
  const handleAgentSelect = (agentId: string) => {
    setSelectedAgent(agentId === selectedAgent ? null : agentId);
  };
  
  const getSystemHealthStatus = () => {
    if (!gridData || !trainingStatus) return 'unknown';
    
    const activeAgents = gridData.agents.filter(a => a.isActive).length;
    const totalAgents = gridData.agents.length;
    const healthRatio = activeAgents / totalAgents;
    
    if (healthRatio > 0.9) return 'excellent';
    if (healthRatio > 0.7) return 'good';
    if (healthRatio > 0.5) return 'warning';
    return 'critical';
  };
  
  const getHealthColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'bg-green-500';
      case 'good': return 'bg-blue-500';
      case 'warning': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };
  
  if (gridLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading bio-inspired MARL system...</p>
        </div>
      </div>
    );
  }
  
  const systemHealth = getSystemHealthStatus();
  
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Brain className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">Bio-Inspired MARL Dashboard</h1>
                <p className="text-sm text-muted-foreground">
                  Hierarchical Multi-Agent Coordination System
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            <Badge variant={systemHealth === 'excellent' ? 'default' : 'secondary'}>
              System: {systemHealth}
            </Badge>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="flex-1 p-6">
        {/* System Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Activity className="h-8 w-8 text-blue-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Active Agents</p>
                  <p className="text-2xl font-bold">
                    {gridData?.agents.filter(a => a.isActive).length || 0}
                    <span className="text-sm text-muted-foreground ml-1">/ {gridData?.agents.length || 0}</span>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <MessageSquare className="h-8 w-8 text-green-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Active Messages</p>
                  <p className="text-2xl font-bold">{gridData?.activeMessages.length || 0}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Zap className="h-8 w-8 text-yellow-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Breakthroughs</p>
                  <p className="text-2xl font-bold">{gridData?.recentBreakthroughs.length || 0}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <Database className="h-8 w-8 text-purple-500" />
                <div>
                  <p className="text-sm text-muted-foreground">Memory Usage</p>
                  <p className="text-2xl font-bold">
                    {memoryState?.usage?.used || 0}
                    <span className="text-sm text-muted-foreground ml-1">/ {memoryState?.usage?.total || 0}</span>
                  </p>
                  <Progress 
                    value={memoryState?.usage?.used / memoryState?.usage?.total * 100 || 0} 
                    className="mt-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        
        {/* Main Visualization Tabs */}
        <Tabs defaultValue="grid" className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="grid">3D Grid</TabsTrigger>
            <TabsTrigger value="communication">Communication</TabsTrigger>
            <TabsTrigger value="memory">Memory</TabsTrigger>
            <TabsTrigger value="breakthroughs">Breakthroughs</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          </TabsList>
          
          <TabsContent value="grid" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5" />
                    Agent Grid (4×3×3)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <AgentGrid3D 
                    agents={gridData?.agents || []}
                    communicationPatterns={gridData?.communicationPatterns || []}
                    onAgentSelect={handleAgentSelect}
                    selectedAgent={selectedAgent}
                  />
                </CardContent>
              </Card>
              
              <div className="space-y-4">
                <SystemControl />
              </div>
              
              <Card>
                <CardHeader>
                  <CardTitle>Agent Details</CardTitle>
                </CardHeader>
                <CardContent>
                  {selectedAgent ? (
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-semibold">{selectedAgent}</h4>
                        <p className="text-sm text-muted-foreground">
                          {gridData?.agents.find(a => a.agentId === selectedAgent)?.type || 'Unknown'}
                        </p>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Status:</span>
                          <Badge variant="outline">
                            {gridData?.agents.find(a => a.agentId === selectedAgent)?.status || 'Unknown'}
                          </Badge>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-sm">Position:</span>
                          <span className="text-sm text-muted-foreground">
                            {(() => {
                              const agent = gridData?.agents.find(a => a.agentId === selectedAgent);
                              return agent ? `(${agent.positionX}, ${agent.positionY}, ${agent.positionZ})` : 'Unknown';
                            })()}
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-sm">Coordinator:</span>
                          <span className="text-sm text-muted-foreground">
                            {gridData?.agents.find(a => a.agentId === selectedAgent)?.coordinatorId || 'None'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">Select an agent to view details</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="communication">
            <CommunicationMetrics />
          </TabsContent>
          
          <TabsContent value="memory">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Shared Memory System
                </CardTitle>
              </CardHeader>
              <CardContent>
                <MemoryVisualization 
                  memoryState={memoryState}
                  agents={gridData?.agents || []}
                />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="breakthroughs">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Breakthrough Events
                </CardTitle>
              </CardHeader>
              <CardContent>
                <BreakthroughPanel 
                  breakthroughs={gridData?.recentBreakthroughs || []}
                  agents={gridData?.agents || []}
                />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="metrics">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Training Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <MetricsPanel 
                  metrics={metrics}
                  trainingStatus={trainingStatus}
                  timeRange={timeRange}
                  onTimeRangeChange={setTimeRange}
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
