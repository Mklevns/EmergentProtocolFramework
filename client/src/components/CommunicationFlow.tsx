import { useState, useEffect } from 'react';
import { Agent, Message, CommunicationPattern } from '@/lib/agent-types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MessageSquare, TrendingUp, Network, Zap } from 'lucide-react';

interface CommunicationFlowProps {
  agents: Agent[];
  messages: Message[];
  patterns: CommunicationPattern[];
}

export function CommunicationFlow({ agents, messages, patterns }: CommunicationFlowProps) {
  const [selectedFlow, setSelectedFlow] = useState<string | null>(null);
  const [timeWindow, setTimeWindow] = useState<'1m' | '5m' | '15m' | '1h'>('5m');

  // Group messages by type
  const messagesByType = messages.reduce((acc, msg) => {
    acc[msg.messageType] = (acc[msg.messageType] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Calculate flow statistics
  const flowStats = patterns.map(pattern => {
    const fromAgent = agents.find(a => a.agentId === pattern.fromAgentId);
    const toAgent = agents.find(a => a.agentId === pattern.toAgentId);
    
    return {
      id: `${pattern.fromAgentId}->${pattern.toAgentId}`,
      fromAgent: fromAgent?.agentId || 'Unknown',
      toAgent: toAgent?.agentId || 'Unknown',
      fromType: fromAgent?.type || 'unknown',
      toType: toAgent?.type || 'unknown',
      frequency: pattern.frequency,
      efficiency: pattern.efficiency,
      lastCommunication: pattern.lastCommunication,
      patternType: pattern.patternType || 'direct'
    };
  }).sort((a, b) => b.frequency - a.frequency);

  // Get top communicators
  const topCommunicators = agents.map(agent => {
    const outgoing = patterns.filter(p => p.fromAgentId === agent.agentId).length;
    const incoming = patterns.filter(p => p.toAgentId === agent.agentId).length;
    
    return {
      agent,
      outgoing,
      incoming,
      total: outgoing + incoming
    };
  }).sort((a, b) => b.total - a.total).slice(0, 5);

  // Communication network visualization
  const renderNetworkGraph = () => {
    const nodes = agents.map(agent => ({
      id: agent.agentId,
      type: agent.type,
      status: agent.status,
      x: agent.positionX * 100 + 50,
      y: agent.positionY * 100 + 50,
      z: agent.positionZ
    }));

    const links = patterns.map(pattern => ({
      source: pattern.fromAgentId,
      target: pattern.toAgentId,
      strength: Math.min(pattern.frequency / 10, 1),
      efficiency: pattern.efficiency
    }));

    return (
      <div className="relative w-full h-96 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg overflow-hidden">
        <svg width="100%" height="100%" viewBox="0 0 400 300">
          {/* Draw connections */}
          {links.map((link, index) => {
            const sourceNode = nodes.find(n => n.id === link.source);
            const targetNode = nodes.find(n => n.id === link.target);
            
            if (!sourceNode || !targetNode) return null;

            const opacity = 0.3 + (link.strength * 0.7);
            const strokeWidth = 1 + (link.strength * 3);

            return (
              <g key={index}>
                <line
                  x1={sourceNode.x}
                  y1={sourceNode.y}
                  x2={targetNode.x}
                  y2={targetNode.y}
                  stroke="#3b82f6"
                  strokeWidth={strokeWidth}
                  opacity={opacity}
                  markerEnd="url(#arrowhead)"
                />
                {/* Flow indicator */}
                <circle
                  cx={sourceNode.x + (targetNode.x - sourceNode.x) * 0.5}
                  cy={sourceNode.y + (targetNode.y - sourceNode.y) * 0.5}
                  r={2}
                  fill="#3b82f6"
                  opacity={opacity}
                >
                  <animateTransform
                    attributeName="transform"
                    type="translate"
                    values={`${sourceNode.x},${sourceNode.y};${targetNode.x},${targetNode.y}`}
                    dur={`${3 - link.strength * 2}s`}
                    repeatCount="indefinite"
                  />
                </circle>
              </g>
            );
          })}

          {/* Draw nodes */}
          {nodes.map((node) => {
            const radius = node.type === 'coordinator' ? 8 : 6;
            const color = node.type === 'coordinator' ? '#f59e0b' : '#64748b';
            const pulseColor = node.status === 'communicating' ? '#3b82f6' : color;

            return (
              <g key={node.id}>
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={radius}
                  fill={color}
                  stroke="#1f2937"
                  strokeWidth={1}
                  className="cursor-pointer hover:opacity-80"
                  onClick={() => setSelectedFlow(node.id)}
                />
                {node.status === 'communicating' && (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={radius + 2}
                    fill="none"
                    stroke={pulseColor}
                    strokeWidth={2}
                    opacity={0.6}
                  >
                    <animate
                      attributeName="r"
                      values={`${radius + 2};${radius + 6};${radius + 2}`}
                      dur="2s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.6;0.2;0.6"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  </circle>
                )}
                <text
                  x={node.x}
                  y={node.y + radius + 12}
                  textAnchor="middle"
                  fontSize="8"
                  fill="#1f2937"
                >
                  {node.id.split('_')[1]}
                </text>
              </g>
            );
          })}

          {/* Arrow marker definition */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="#3b82f6"
              />
            </marker>
          </defs>
        </svg>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Communication Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-blue-500" />
              <div>
                <p className="text-sm text-muted-foreground">Total Messages</p>
                <p className="text-2xl font-bold">{messages.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Network className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm text-muted-foreground">Active Flows</p>
                <p className="text-2xl font-bold">{patterns.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-purple-500" />
              <div>
                <p className="text-sm text-muted-foreground">Avg Efficiency</p>
                <p className="text-2xl font-bold">
                  {patterns.length > 0 ? 
                    (patterns.reduce((sum, p) => sum + p.efficiency, 0) / patterns.length).toFixed(1) : 
                    '0.0'
                  }%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              <div>
                <p className="text-sm text-muted-foreground">Breakthroughs</p>
                <p className="text-2xl font-bold">
                  {messagesByType.breakthrough || 0}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="network" className="space-y-4">
        <TabsList>
          <TabsTrigger value="network">Network Graph</TabsTrigger>
          <TabsTrigger value="flows">Communication Flows</TabsTrigger>
          <TabsTrigger value="patterns">Message Patterns</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
        </TabsList>

        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Real-time Communication Network</CardTitle>
            </CardHeader>
            <CardContent>
              {renderNetworkGraph()}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="flows" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Communication Flows</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {flowStats.map((flow) => (
                  <div 
                    key={flow.id}
                    className="flex items-center justify-between p-3 bg-muted rounded-lg hover:bg-muted/80 cursor-pointer"
                    onClick={() => setSelectedFlow(flow.id)}
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <Badge variant={flow.fromType === 'coordinator' ? 'default' : 'secondary'}>
                          {flow.fromAgent}
                        </Badge>
                        <span className="text-muted-foreground">→</span>
                        <Badge variant={flow.toType === 'coordinator' ? 'default' : 'secondary'}>
                          {flow.toAgent}
                        </Badge>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {flow.patternType}
                      </Badge>
                    </div>
                    
                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-right">
                        <div className="font-medium">{flow.frequency} msgs</div>
                        <div className="text-muted-foreground">{flow.efficiency.toFixed(1)}% efficient</div>
                      </div>
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(flow.frequency / 20 * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Message Type Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(messagesByType).map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{type}</Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{count}</span>
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${(count / Math.max(...Object.values(messagesByType))) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="stats" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Top Communicators</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {topCommunicators.map((comm, index) => (
                  <div key={comm.agent.agentId} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-primary-foreground font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium">{comm.agent.agentId}</div>
                        <div className="text-sm text-muted-foreground">{comm.agent.type}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">{comm.total} connections</div>
                      <div className="text-sm text-muted-foreground">
                        {comm.outgoing} out, {comm.incoming} in
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
