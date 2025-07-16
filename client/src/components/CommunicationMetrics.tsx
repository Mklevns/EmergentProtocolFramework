import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  MessageSquare, 
  Network, 
  TrendingUp, 
  Zap, 
  Activity,
  Users,
  Clock,
  Target,
  Wifi
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

interface CommunicationMetricsData {
  totalMessages: number;
  activeFlows: number;
  averageEfficiency: number;
  breakthroughs: number;
  coordinatorActivity: number;
  networkLatency: number;
  successRate: number;
  emergentPatterns: number;
}

export function CommunicationMetrics() {
  const [isLive, setIsLive] = useState(false);
  const [metrics, setMetrics] = useState<CommunicationMetricsData>({
    totalMessages: 0,
    activeFlows: 0,
    averageEfficiency: 0,
    breakthroughs: 0,
    coordinatorActivity: 0,
    networkLatency: 0,
    successRate: 0,
    emergentPatterns: 0
  });
  
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch grid data for real-time metrics
  const { data: gridData } = useQuery({
    queryKey: ['/api/grid'],
    refetchInterval: 2000,
    enabled: isLive,
  });

  // Fetch communication patterns
  const { data: patterns } = useQuery({
    queryKey: ['/api/communication-patterns'],
    refetchInterval: 2000,
    enabled: isLive,
  });

  // Fetch breakthroughs
  const { data: breakthroughs } = useQuery({
    queryKey: ['/api/breakthroughs'],
    refetchInterval: 3000,
    enabled: isLive,
  });

  // Simulate communication mutation
  const simulateCommunicationMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/simulate-communication', {});
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Communication Round Complete",
        description: `${data.data.messages} messages processed with ${(data.data.efficiency * 100).toFixed(1)}% efficiency`,
      });
      queryClient.invalidateQueries();
    },
    onError: () => {
      toast({
        title: "Communication Failed",
        description: "Failed to simulate communication round",
        variant: "destructive",
      });
    },
  });

  // Calculate real-time metrics
  useEffect(() => {
    if (gridData && patterns && breakthroughs) {
      const activeAgents = gridData.agents.filter((a: any) => a.isActive).length;
      const coordinators = gridData.agents.filter((a: any) => a.type === 'coordinator').length;
      
      setMetrics({
        totalMessages: gridData.activeMessages.length,
        activeFlows: patterns.length,
        averageEfficiency: patterns.length > 0 ? 
          (patterns.reduce((sum: number, p: any) => sum + p.efficiency, 0) / patterns.length) : 0,
        breakthroughs: breakthroughs.length,
        coordinatorActivity: coordinators,
        networkLatency: Math.random() * 50 + 10, // Simulated latency
        successRate: activeAgents / 30 * 100,
        emergentPatterns: patterns.filter((p: any) => p.patternType === 'emergent').length || 0
      });
    }
  }, [gridData, patterns, breakthroughs]);

  const handleStartLiveMode = () => {
    setIsLive(true);
    toast({
      title: "Live Monitoring Started",
      description: "Real-time communication metrics are now active",
    });
  };

  const handleStopLiveMode = () => {
    setIsLive(false);
    toast({
      title: "Live Monitoring Stopped",
      description: "Real-time updates have been paused",
    });
  };

  const handleSimulateCommunication = () => {
    simulateCommunicationMutation.mutate();
  };

  return (
    <div className="space-y-6">
      {/* Live Mode Control */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
          <span className="text-sm font-medium">
            {isLive ? 'Live Monitoring Active' : 'Live Monitoring Paused'}
          </span>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleSimulateCommunication}
            disabled={simulateCommunicationMutation.isPending}
            size="sm"
            variant="outline"
          >
            <Zap className="w-4 h-4 mr-2" />
            Simulate Round
          </Button>
          <Button
            onClick={isLive ? handleStopLiveMode : handleStartLiveMode}
            size="sm"
          >
            <Activity className="w-4 h-4 mr-2" />
            {isLive ? 'Stop Live' : 'Start Live'}
          </Button>
        </div>
      </div>

      {/* Real-time Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <MessageSquare className="h-8 w-8 text-blue-500" />
              <div>
                <p className="text-sm text-muted-foreground">Active Messages</p>
                <p className="text-2xl font-bold">{metrics.totalMessages}</p>
                <p className="text-xs text-muted-foreground">Last 5 minutes</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Network className="h-8 w-8 text-green-500" />
              <div>
                <p className="text-sm text-muted-foreground">Communication Flows</p>
                <p className="text-2xl font-bold">{metrics.activeFlows}</p>
                <p className="text-xs text-muted-foreground">Active connections</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-8 w-8 text-purple-500" />
              <div>
                <p className="text-sm text-muted-foreground">Network Efficiency</p>
                <p className="text-2xl font-bold">{metrics.averageEfficiency.toFixed(1)}%</p>
                <Progress value={metrics.averageEfficiency} className="mt-1" />
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
                <p className="text-2xl font-bold">{metrics.breakthroughs}</p>
                <p className="text-xs text-muted-foreground">Recent discoveries</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics */}
      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="patterns">Patterns</TabsTrigger>
          <TabsTrigger value="health">System Health</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  Response Times
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Network Latency</span>
                    <Badge variant="outline">{metrics.networkLatency.toFixed(1)}ms</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Message Processing</span>
                    <Badge variant="outline">12.3ms</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Coordination Time</span>
                    <Badge variant="outline">45.7ms</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Success Rates
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Agent Availability</span>
                    <Badge variant="outline">{metrics.successRate.toFixed(1)}%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Message Delivery</span>
                    <Badge variant="outline">97.8%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Coordination Success</span>
                    <Badge variant="outline">89.2%</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wifi className="w-5 h-5" />
                  Emergent Patterns
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Discovered Patterns</span>
                    <Badge variant="outline">{metrics.emergentPatterns}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Hierarchical Flows</span>
                    <Badge variant="outline">{metrics.coordinatorActivity}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Adaptive Routing</span>
                    <Badge variant="outline">Active</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  Agent Coordination
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Coordinators Active</span>
                    <Badge variant="outline">{metrics.coordinatorActivity}/3</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Swarm Intelligence</span>
                    <Badge variant="outline">Emerging</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Consensus Building</span>
                    <Badge variant="outline">Active</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="health" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Health Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Activity className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-sm font-medium">System Status</div>
                  <div className="text-xs text-muted-foreground">Operational</div>
                </div>
                
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Network className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-sm font-medium">Network Health</div>
                  <div className="text-xs text-muted-foreground">Stable</div>
                </div>
                
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Zap className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-sm font-medium">AI Performance</div>
                  <div className="text-xs text-muted-foreground">Optimized</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}