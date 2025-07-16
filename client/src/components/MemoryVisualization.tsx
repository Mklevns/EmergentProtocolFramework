import { useState, useEffect, useRef } from 'react';
import { MemoryState, Agent } from '@/lib/agent-types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Database, Zap, Activity, Clock, Search } from 'lucide-react';
import { Input } from '@/components/ui/input';

interface MemoryVisualizationProps {
  memoryState: MemoryState | undefined;
  agents: Agent[];
}

export function MemoryVisualization({ memoryState, agents }: MemoryVisualizationProps) {
  const [selectedVector, setSelectedVector] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Filter vectors based on search and type
  const filteredVectors = memoryState?.vectors.filter(vector => {
    const matchesSearch = vector.vectorId.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         vector.vectorType.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = filterType === 'all' || vector.vectorType === filterType;
    return matchesSearch && matchesType;
  }) || [];

  // Get unique vector types
  const vectorTypes = [...new Set(memoryState?.vectors.map(v => v.vectorType) || [])];

  // Memory usage visualization
  const renderMemoryHeatmap = () => {
    const canvas = canvasRef.current;
    if (!canvas || !memoryState) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set up grid
    const gridSize = 20;
    const cols = Math.floor(canvas.width / gridSize);
    const rows = Math.floor(canvas.height / gridSize);

    // Create memory map
    const memoryMap = new Array(rows).fill(null).map(() => new Array(cols).fill(0));

    // Map vectors to grid positions
    memoryState.vectors.forEach((vector, index) => {
      const row = Math.floor(index / cols) % rows;
      const col = index % cols;
      if (row < rows && col < cols) {
        memoryMap[row][col] = vector.accessCount;
      }
    });

    // Find max access count for normalization
    const maxAccess = Math.max(...memoryState.vectors.map(v => v.accessCount));

    // Draw heatmap
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x = col * gridSize;
        const y = row * gridSize;
        const accessCount = memoryMap[row][col];
        
        if (accessCount > 0) {
          const intensity = accessCount / maxAccess;
          const alpha = 0.3 + (intensity * 0.7);
          
          // Color based on access frequency
          let color;
          if (intensity > 0.7) {
            color = `rgba(239, 68, 68, ${alpha})`; // Red for high activity
          } else if (intensity > 0.4) {
            color = `rgba(245, 158, 11, ${alpha})`; // Orange for medium activity
          } else {
            color = `rgba(59, 130, 246, ${alpha})`; // Blue for low activity
          }
          
          ctx.fillStyle = color;
          ctx.fillRect(x, y, gridSize - 1, gridSize - 1);
        }
      }
    }

    // Draw grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;
    
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      ctx.moveTo(i * gridSize, 0);
      ctx.lineTo(i * gridSize, canvas.height);
      ctx.stroke();
    }
    
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * gridSize);
      ctx.lineTo(canvas.width, i * gridSize);
      ctx.stroke();
    }
  };

  useEffect(() => {
    renderMemoryHeatmap();
  }, [memoryState]);

  const getVectorTypeColor = (type: string) => {
    const colors = {
      'breakthrough': '#ef4444',
      'context': '#3b82f6',
      'coordination': '#8b5cf6',
      'memory_trace': '#10b981',
      'pattern': '#f59e0b'
    };
    return colors[type as keyof typeof colors] || '#64748b';
  };

  const getVectorTypeBadgeVariant = (type: string) => {
    const variants = {
      'breakthrough': 'destructive',
      'context': 'default',
      'coordination': 'secondary',
      'memory_trace': 'outline',
      'pattern': 'default'
    };
    return variants[type as keyof typeof variants] || 'outline';
  };

  if (!memoryState) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">No memory data available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Memory Usage Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-blue-500" />
              <div className="flex-1">
                <p className="text-sm text-muted-foreground">Memory Usage</p>
                <p className="text-2xl font-bold">
                  {memoryState.usage.used}/{memoryState.usage.total}
                </p>
                <Progress 
                  value={(memoryState.usage.used / memoryState.usage.total) * 100} 
                  className="mt-2"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-sm text-muted-foreground">Efficiency</p>
                <p className="text-2xl font-bold">{memoryState.usage.efficiency.toFixed(1)}%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              <div>
                <p className="text-sm text-muted-foreground">Active Vectors</p>
                <p className="text-2xl font-bold">{memoryState.vectors.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="vectors" className="space-y-4">
        <TabsList>
          <TabsTrigger value="vectors">Memory Vectors</TabsTrigger>
          <TabsTrigger value="heatmap">Memory Heatmap</TabsTrigger>
          <TabsTrigger value="access">Recent Access</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="vectors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Memory Vectors</CardTitle>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search vectors..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-8"
                  />
                </div>
                <select 
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 border rounded-md"
                >
                  <option value="all">All Types</option>
                  {vectorTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {filteredVectors.map((vector) => (
                  <div
                    key={vector.vectorId}
                    className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedVector === vector.vectorId
                        ? 'border-primary bg-primary/5'
                        : 'border-border hover:bg-muted/50'
                    }`}
                    onClick={() => setSelectedVector(vector.vectorId)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: getVectorTypeColor(vector.vectorType) }}
                        />
                        <div>
                          <div className="font-medium">{vector.vectorId}</div>
                          <Badge variant={getVectorTypeBadgeVariant(vector.vectorType) as any}>
                            {vector.vectorType}
                          </Badge>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm font-medium">
                          Accessed: {vector.accessCount}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Importance: {vector.importance.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                
                {filteredVectors.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    No vectors found matching your criteria
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="heatmap" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Memory Access Heatmap</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <canvas
                  ref={canvasRef}
                  width={600}
                  height={400}
                  className="border rounded-lg w-full"
                />
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded"></div>
                    <span>Low Activity</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                    <span>Medium Activity</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500 rounded"></div>
                    <span>High Activity</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="access" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Memory Access</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {memoryState.recentAccess.map((access, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div className="flex items-center gap-3">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="font-medium">{access.vectorId}</div>
                        <div className="text-sm text-muted-foreground">
                          Accessed by {access.agentId}
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {new Date(access.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
                
                {memoryState.recentAccess.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    No recent memory access recorded
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Vector Type Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {vectorTypes.map(type => {
                    const count = memoryState.vectors.filter(v => v.vectorType === type).length;
                    const percentage = (count / memoryState.vectors.length) * 100;
                    
                    return (
                      <div key={type} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getVectorTypeColor(type) }}
                          />
                          <span className="text-sm">{type}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium">{count}</span>
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div
                              className="h-2 rounded-full transition-all duration-300"
                              style={{ 
                                width: `${percentage}%`,
                                backgroundColor: getVectorTypeColor(type)
                              }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground w-8">
                            {percentage.toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm">Total Vectors:</span>
                    <span className="font-medium">{memoryState.vectors.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Average Importance:</span>
                    <span className="font-medium">
                      {(memoryState.vectors.reduce((sum, v) => sum + v.importance, 0) / memoryState.vectors.length).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Total Access Count:</span>
                    <span className="font-medium">
                      {memoryState.vectors.reduce((sum, v) => sum + v.accessCount, 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Most Accessed Vector:</span>
                    <span className="font-medium">
                      {memoryState.vectors.reduce((max, v) => v.accessCount > max.accessCount ? v : max, memoryState.vectors[0])?.vectorId || 'None'}
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
