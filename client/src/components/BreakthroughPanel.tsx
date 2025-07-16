import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Breakthrough, Agent } from '@/lib/agent-types';
import { Zap, Clock, TrendingUp, Activity, Brain, Target } from 'lucide-react';

interface BreakthroughPanelProps {
  breakthroughs: Breakthrough[];
  agents: Agent[];
}

export function BreakthroughPanel({ breakthroughs, agents }: BreakthroughPanelProps) {
  const [selectedBreakthrough, setSelectedBreakthrough] = useState<string | null>(null);

  // Group breakthroughs by type
  const breakthroughsByType = breakthroughs.reduce((acc, bt) => {
    acc[bt.breakthroughType] = (acc[bt.breakthroughType] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Group breakthroughs by agent
  const breakthroughsByAgent = breakthroughs.reduce((acc, bt) => {
    acc[bt.agentId] = (acc[bt.agentId] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Calculate breakthrough frequency over time
  const getBreakthroughFrequency = () => {
    const now = Date.now();
    const lastHour = breakthroughs.filter(bt => 
      now - new Date(bt.timestamp).getTime() < 3600000
    ).length;
    return lastHour;
  };

  // Get top performing agents
  const topAgents = Object.entries(breakthroughsByAgent)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([agentId, count]) => ({
      agentId,
      count,
      agent: agents.find(a => a.agentId === agentId)
    }));

  // Get breakthrough type colors
  const getBreakthroughTypeColor = (type: string) => {
    const colors = {
      'pattern_recognition': '#3b82f6',
      'coordination_improvement': '#10b981',
      'efficiency_gain': '#f59e0b',
      'novel_strategy': '#8b5cf6',
      'communication_protocol': '#ef4444',
      'memory_optimization': '#6b7280',
      'spatial_awareness': '#06b6d4'
    };
    return colors[type as keyof typeof colors] || '#64748b';
  };

  const getBreakthroughTypeIcon = (type: string) => {
    const icons = {
      'pattern_recognition': Brain,
      'coordination_improvement': Activity,
      'efficiency_gain': TrendingUp,
      'novel_strategy': Target,
      'communication_protocol': Zap,
      'memory_optimization': Brain,
      'spatial_awareness': Activity
    };
    return icons[type as keyof typeof icons] || Zap;
  };

  const formatBreakthroughType = (type: string) => {
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Breakthrough Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              <div>
                <p className="text-sm text-muted-foreground">Total Breakthroughs</p>
                <p className="text-2xl font-bold">{breakthroughs.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-blue-500" />
              <div>
                <p className="text-sm text-muted-foreground">Last Hour</p>
                <p className="text-2xl font-bold">{getBreakthroughFrequency()}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm text-muted-foreground">Avg Confidence</p>
                <p className="text-2xl font-bold">
                  {breakthroughs.length > 0 ? 
                    (breakthroughs.reduce((sum, bt) => sum + bt.confidence, 0) / breakthroughs.length).toFixed(1) : 
                    '0.0'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-500" />
              <div>
                <p className="text-sm text-muted-foreground">Shared Rate</p>
                <p className="text-2xl font-bold">
                  {breakthroughs.length > 0 ? 
                    ((breakthroughs.filter(bt => bt.wasShared).length / breakthroughs.length) * 100).toFixed(0) : 
                    '0'
                  }%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="recent" className="space-y-4">
        <TabsList>
          <TabsTrigger value="recent">Recent Breakthroughs</TabsTrigger>
          <TabsTrigger value="types">By Type</TabsTrigger>
          <TabsTrigger value="agents">By Agent</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="recent" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Breakthrough Events</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {breakthroughs.slice().reverse().map((breakthrough) => {
                  const Icon = getBreakthroughTypeIcon(breakthrough.breakthroughType);
                  const agent = agents.find(a => a.agentId === breakthrough.agentId);
                  
                  return (
                    <div
                      key={breakthrough.id}
                      className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                        selectedBreakthrough === breakthrough.id.toString()
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:bg-muted/50'
                      }`}
                      onClick={() => setSelectedBreakthrough(breakthrough.id.toString())}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <div
                            className="p-2 rounded-full"
                            style={{ backgroundColor: `${getBreakthroughTypeColor(breakthrough.breakthroughType)}20` }}
                          >
                            <Icon 
                              className="h-4 w-4" 
                              style={{ color: getBreakthroughTypeColor(breakthrough.breakthroughType) }}
                            />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="outline">{breakthrough.agentId}</Badge>
                              <Badge 
                                variant="secondary"
                                style={{ 
                                  backgroundColor: `${getBreakthroughTypeColor(breakthrough.breakthroughType)}20`,
                                  color: getBreakthroughTypeColor(breakthrough.breakthroughType)
                                }}
                              >
                                {formatBreakthroughType(breakthrough.breakthroughType)}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {breakthrough.description || `${formatBreakthroughType(breakthrough.breakthroughType)} detected`}
                            </p>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className={`text-sm font-medium ${getConfidenceColor(breakthrough.confidence)}`}>
                            {(breakthrough.confidence * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {new Date(breakthrough.timestamp).toLocaleTimeString()}
                          </div>
                          {breakthrough.wasShared && (
                            <Badge variant="outline" className="mt-1 text-xs">
                              Shared
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
                
                {breakthroughs.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    No breakthroughs detected yet
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="types" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Breakthrough Types</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(breakthroughsByType).map(([type, count]) => {
                  const Icon = getBreakthroughTypeIcon(type);
                  const percentage = (count / breakthroughs.length) * 100;
                  
                  return (
                    <div key={type} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-3">
                        <div
                          className="p-2 rounded-full"
                          style={{ backgroundColor: `${getBreakthroughTypeColor(type)}20` }}
                        >
                          <Icon 
                            className="h-4 w-4" 
                            style={{ color: getBreakthroughTypeColor(type) }}
                          />
                        </div>
                        <div>
                          <div className="font-medium">{formatBreakthroughType(type)}</div>
                          <div className="text-sm text-muted-foreground">
                            {count} events ({percentage.toFixed(1)}%)
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2">
                          <div
                            className="h-2 rounded-full transition-all duration-300"
                            style={{ 
                              width: `${percentage}%`,
                              backgroundColor: getBreakthroughTypeColor(type)
                            }}
                          />
                        </div>
                        <span className="text-sm font-medium w-8">{count}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="agents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Top Performing Agents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {topAgents.map(({ agentId, count, agent }, index) => (
                  <div key={agentId} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-primary-foreground font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium">{agentId}</div>
                        <div className="text-sm text-muted-foreground">
                          {agent?.type || 'Unknown'} • Position: {agent ? `(${agent.positionX}, ${agent.positionY}, ${agent.positionZ})` : 'Unknown'}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">{count} breakthroughs</div>
                      <div className="text-sm text-muted-foreground">
                        {((count / breakthroughs.length) * 100).toFixed(1)}% of total
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Breakthrough Timeline</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {breakthroughs.slice(-5).reverse().map((breakthrough) => (
                    <div key={breakthrough.id} className="flex items-center gap-3 p-2 bg-muted rounded">
                      <div className="w-2 h-2 rounded-full bg-primary" />
                      <div className="flex-1">
                        <div className="text-sm font-medium">{breakthrough.agentId}</div>
                        <div className="text-xs text-muted-foreground">
                          {formatBreakthroughType(breakthrough.breakthroughType)}
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(breakthrough.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Sharing Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm">Total Breakthroughs:</span>
                    <span className="font-medium">{breakthroughs.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Shared Breakthroughs:</span>
                    <span className="font-medium">{breakthroughs.filter(bt => bt.wasShared).length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Sharing Rate:</span>
                    <span className="font-medium">
                      {breakthroughs.length > 0 ? 
                        ((breakthroughs.filter(bt => bt.wasShared).length / breakthroughs.length) * 100).toFixed(1) : 
                        '0'
                      }%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Average Confidence:</span>
                    <span className="font-medium">
                      {breakthroughs.length > 0 ? 
                        (breakthroughs.reduce((sum, bt) => sum + bt.confidence, 0) / breakthroughs.length).toFixed(2) : 
                        '0.00'
                      }
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
