import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Brain, Network, Zap, Search, TrendingUp, BarChart3 } from "lucide-react";

interface EnhancedStats {
  communication_metrics: {
    total_messages_processed: number;
    average_embedding_time: number;
    bandwidth_efficiency: number;
    context_coherence_avg: number;
    retrieval_accuracy: number;
  };
  bandwidth_metrics: {
    current_usage: number;
    max_bandwidth: number;
    utilization_percentage: number;
    average_congestion: number;
    peak_congestion: number;
    active_agents: number;
    agent_allocations: Record<string, number>;
  };
  indexing_metrics: {
    retrieval_stats: {
      semantic_queries: number;
      temporal_queries: number;
      spatial_queries: number;
      associative_queries: number;
      cache_hits: number;
      cache_misses: number;
    };
    cache_stats: {
      hit_rate: number;
    };
  };
  context_metrics: {
    active_contexts: number;
    average_context_coherence: number;
    average_urgency: number;
  };
  total_embeddings: number;
}

interface MemoryStats {
  total_vectors: number;
  total_pointers: number;
  usage_percentage: number;
  importance_distribution: Record<string, number>;
  temporal_buckets: number;
  association_connections: number;
  co_access_patterns: number;
  concept_relationships: number;
  advanced_stats: {
    prediction_accuracy: number;
    associative_retrievals: number;
    temporal_retrievals: number;
    spatial_retrievals: number;
  };
}

export function EnhancedCommunicationPanel() {
  const [semanticQuery, setSemanticQuery] = useState("");
  const [associativeVectorId, setAssociativeVectorId] = useState("");
  const [accessSequence, setAccessSequence] = useState("");
  
  const queryClient = useQueryClient();

  // Enhanced communication stats
  const { data: enhancedStats, isLoading: statsLoading } = useQuery({
    queryKey: ['/api/communication/enhanced-stats'],
    refetchInterval: 5000,
  });

  // Advanced memory stats
  const { data: memoryStats, isLoading: memoryLoading } = useQuery({
    queryKey: ['/api/memory/advanced-stats'],
    refetchInterval: 5000,
  });

  // Bandwidth usage
  const { data: bandwidthData, isLoading: bandwidthLoading } = useQuery({
    queryKey: ['/api/communication/bandwidth-usage'],
    refetchInterval: 3000,
  });

  // Semantic memory query mutation
  const semanticQueryMutation = useMutation({
    mutationFn: async (query: string) => {
      const response = await fetch('/api/memory/query-semantic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_text: query, max_results: 10, threshold: 0.7 }),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/memory/advanced-stats'] });
    },
  });

  // Associative memory query mutation
  const associativeQueryMutation = useMutation({
    mutationFn: async (vectorId: string) => {
      const response = await fetch('/api/memory/query-associative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vector_id: vectorId, max_depth: 3, max_results: 15 }),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/memory/advanced-stats'] });
    },
  });

  // Predictive prefetch mutation
  const prefetchMutation = useMutation({
    mutationFn: async (sequence: string[]) => {
      const response = await fetch('/api/memory/predictive-prefetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ access_sequence: sequence, max_predictions: 5 }),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/memory/advanced-stats'] });
    },
  });

  const handleSemanticQuery = () => {
    if (semanticQuery.trim()) {
      semanticQueryMutation.mutate(semanticQuery);
    }
  };

  const handleAssociativeQuery = () => {
    if (associativeVectorId.trim()) {
      associativeQueryMutation.mutate(associativeVectorId);
    }
  };

  const handlePredictivePrefetch = () => {
    if (accessSequence.trim()) {
      const sequence = accessSequence.split(',').map(s => s.trim()).filter(s => s);
      prefetchMutation.mutate(sequence);
    }
  };

  const stats = enhancedStats?.data as EnhancedStats | undefined;
  const memory = memoryStats?.data as MemoryStats | undefined;
  const bandwidth = bandwidthData?.data;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Enhanced Communication & Memory
          </CardTitle>
          <CardDescription>
            Advanced communication protocols with sophisticated memory indexing and retrieval
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs defaultValue="communication" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="communication">Communication</TabsTrigger>
          <TabsTrigger value="memory">Memory</TabsTrigger>
          <TabsTrigger value="bandwidth">Bandwidth</TabsTrigger>
          <TabsTrigger value="queries">Advanced Queries</TabsTrigger>
        </TabsList>

        <TabsContent value="communication" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Message Embeddings</CardTitle>
                <Network className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats?.total_embeddings || 0}</div>
                <p className="text-xs text-muted-foreground">
                  Avg processing: {stats?.communication_metrics.average_embedding_time?.toFixed(3) || 0}ms
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Context Coherence</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(stats?.communication_metrics.context_coherence_avg * 100)?.toFixed(1) || 0}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Active contexts: {stats?.context_metrics.active_contexts || 0}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Bandwidth Efficiency</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(stats?.communication_metrics.bandwidth_efficiency * 100)?.toFixed(1) || 0}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Retrieval accuracy: {(stats?.communication_metrics.retrieval_accuracy * 100)?.toFixed(1) || 0}%
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Advanced Communication Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {stats?.communication_metrics.total_messages_processed || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">Messages Processed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {(stats?.context_metrics.average_urgency * 100)?.toFixed(1) || 0}%
                  </div>
                  <div className="text-sm text-muted-foreground">Avg Urgency</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {stats?.indexing_metrics.retrieval_stats.semantic_queries || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">Semantic Queries</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {(stats?.indexing_metrics.cache_stats.hit_rate * 100)?.toFixed(1) || 0}%
                  </div>
                  <div className="text-sm text-muted-foreground">Cache Hit Rate</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="memory" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{memory?.usage_percentage?.toFixed(1) || 0}%</div>
                <Progress value={memory?.usage_percentage || 0} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-2">
                  {memory?.total_vectors || 0} vectors, {memory?.total_pointers || 0} pointers
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Prediction Accuracy</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(memory?.advanced_stats.prediction_accuracy * 100)?.toFixed(1) || 0}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Associative retrievals: {memory?.advanced_stats.associative_retrievals || 0}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Associations</CardTitle>
                <Network className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{memory?.association_connections || 0}</div>
                <p className="text-xs text-muted-foreground">
                  Co-access patterns: {memory?.co_access_patterns || 0}
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Importance Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {memory?.importance_distribution && Object.entries(memory.importance_distribution).map(([level, count]) => (
                  <div key={level} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant={
                        level === 'critical' ? 'destructive' :
                        level === 'high' ? 'default' :
                        level === 'medium' ? 'secondary' : 'outline'
                      }>
                        {level}
                      </Badge>
                    </div>
                    <span className="font-mono">{count}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="bandwidth" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Adaptive Bandwidth Management</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {bandwidth?.utilization_percentage?.toFixed(1) || 0}%
                  </div>
                  <div className="text-sm text-muted-foreground">Current Utilization</div>
                  <Progress value={bandwidth?.utilization_percentage || 0} className="mt-2" />
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {(bandwidth?.current_usage || 0).toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    / {(bandwidth?.max_bandwidth || 0).toLocaleString()} bytes
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{bandwidth?.active_agents || 0}</div>
                  <div className="text-sm text-muted-foreground">Active Agents</div>
                </div>
              </div>

              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2">Congestion Metrics</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-lg font-semibold">
                      {(bandwidth?.average_congestion * 100)?.toFixed(1) || 0}%
                    </div>
                    <div className="text-xs text-muted-foreground">Average Congestion</div>
                  </div>
                  <div>
                    <div className="text-lg font-semibold">
                      {(bandwidth?.peak_congestion * 100)?.toFixed(1) || 0}%
                    </div>
                    <div className="text-xs text-muted-foreground">Peak Congestion</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="queries" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Semantic Memory Query</CardTitle>
                <CardDescription>Search memory using natural language</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Input
                  placeholder="Enter semantic query..."
                  value={semanticQuery}
                  onChange={(e) => setSemanticQuery(e.target.value)}
                />
                <Button 
                  onClick={handleSemanticQuery} 
                  disabled={semanticQueryMutation.isPending}
                  className="w-full"
                >
                  <Search className="h-4 w-4 mr-2" />
                  {semanticQueryMutation.isPending ? 'Searching...' : 'Search Memory'}
                </Button>
                {semanticQueryMutation.data && (
                  <div className="text-sm">
                    Found {semanticQueryMutation.data.data?.total_found || 0} results
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Associative Query</CardTitle>
                <CardDescription>Find related memories through associations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Input
                  placeholder="Enter vector ID..."
                  value={associativeVectorId}
                  onChange={(e) => setAssociativeVectorId(e.target.value)}
                />
                <Button 
                  onClick={handleAssociativeQuery} 
                  disabled={associativeQueryMutation.isPending}
                  className="w-full"
                >
                  <Network className="h-4 w-4 mr-2" />
                  {associativeQueryMutation.isPending ? 'Searching...' : 'Find Associations'}
                </Button>
                {associativeQueryMutation.data && (
                  <div className="text-sm">
                    Found {associativeQueryMutation.data.data?.total_found || 0} associations
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Predictive Prefetching</CardTitle>
              <CardDescription>Predict next memory accesses based on patterns</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Enter access sequence (comma-separated IDs)..."
                value={accessSequence}
                onChange={(e) => setAccessSequence(e.target.value)}
              />
              <Button 
                onClick={handlePredictivePrefetch} 
                disabled={prefetchMutation.isPending}
                className="w-full"
              >
                <TrendingUp className="h-4 w-4 mr-2" />
                {prefetchMutation.isPending ? 'Predicting...' : 'Predict Next Accesses'}
              </Button>
              {prefetchMutation.data && (
                <div className="text-sm">
                  Predicted {prefetchMutation.data.data?.prediction_count || 0} next accesses
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}