import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Database, TrendingUp, Activity, Clock } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

interface MemoryUtilizationChartProps {
  memoryData: any[];
  timeWindow: number;
  selectedAgent?: string | null;
}

export function MemoryUtilizationChart({ 
  memoryData, 
  timeWindow, 
  selectedAgent 
}: MemoryUtilizationChartProps) {
  
  const processedData = useMemo(() => {
    const now = Date.now();
    const cutoff = now - (timeWindow * 1000);
    
    // Filter memory data by time window
    const recentMemory = memoryData.filter(mem => {
      const memTime = new Date(mem.createdAt).getTime();
      return memTime >= cutoff;
    });

    // Group by memory key for analysis
    const memoryGroups = new Map<string, {
      key: string;
      accessCount: number;
      lastAccessed: Date;
      vectorSize: number;
      type: string;
    }>();

    recentMemory.forEach(mem => {
      const key = mem.memoryKey;
      const existing = memoryGroups.get(key);
      
      if (existing) {
        existing.accessCount += mem.accessCount || 0;
        existing.lastAccessed = new Date(Math.max(
          existing.lastAccessed.getTime(), 
          new Date(mem.lastAccessed).getTime()
        ));
      } else {
        memoryGroups.set(key, {
          key,
          accessCount: mem.accessCount || 0,
          lastAccessed: new Date(mem.lastAccessed),
          vectorSize: mem.vectorData?.embedding?.length || 0,
          type: mem.vectorData?.metadata?.type || 'unknown'
        });
      }
    });

    // Create chart data
    const chartData = Array.from(memoryGroups.values())
      .sort((a, b) => b.accessCount - a.accessCount)
      .slice(0, 10)
      .map(group => ({
        name: group.key.slice(0, 15) + (group.key.length > 15 ? '...' : ''),
        fullName: group.key,
        accessCount: group.accessCount,
        vectorSize: group.vectorSize,
        type: group.type,
        lastAccessed: group.lastAccessed
      }));

    // Memory type distribution
    const typeDistribution = new Map<string, number>();
    memoryGroups.forEach(group => {
      typeDistribution.set(group.type, (typeDistribution.get(group.type) || 0) + 1);
    });

    const pieData = Array.from(typeDistribution.entries()).map(([type, count]) => ({
      name: type,
      value: count
    }));

    return {
      chartData,
      pieData,
      totalEntries: memoryGroups.size,
      totalAccesses: Array.from(memoryGroups.values()).reduce((sum, g) => sum + g.accessCount, 0),
      averageVectorSize: memoryGroups.size > 0 
        ? Array.from(memoryGroups.values()).reduce((sum, g) => sum + g.vectorSize, 0) / memoryGroups.size
        : 0
    };
  }, [memoryData, timeWindow]);

  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];
  
  const memoryUtilization = processedData.totalEntries > 0 
    ? (processedData.totalAccesses / processedData.totalEntries) * 10 // Scale for visualization
    : 0;

  return (
    <div className="space-y-4">
      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-blue-600" />
              <div>
                <p className="text-sm text-muted-foreground">Total Entries</p>
                <p className="text-lg font-bold">{processedData.totalEntries}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-green-600" />
              <div>
                <p className="text-sm text-muted-foreground">Total Accesses</p>
                <p className="text-lg font-bold">{processedData.totalAccesses}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-600" />
              <div>
                <p className="text-sm text-muted-foreground">Avg Vector Size</p>
                <p className="text-lg font-bold">{Math.round(processedData.averageVectorSize)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-orange-600" />
              <div>
                <p className="text-sm text-muted-foreground">Utilization</p>
                <p className="text-lg font-bold">{Math.round(memoryUtilization)}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Memory Access Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Memory Access Patterns</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={processedData.chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => [value, name]}
                  labelFormatter={(label) => {
                    const item = processedData.chartData.find(d => d.name === label);
                    return item ? item.fullName : label;
                  }}
                />
                <Bar dataKey="accessCount" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Memory Type Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Memory Type Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={processedData.pieData}
                    cx="50%"
                    cy="50%"
                    outerRadius={60}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {processedData.pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Memory Usage Details */}
        <Card>
          <CardHeader>
            <CardTitle>Usage Details</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Memory Utilization</span>
                  <span className="text-sm font-medium">{Math.round(memoryUtilization)}%</span>
                </div>
                <Progress value={Math.min(memoryUtilization, 100)} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Active Entries</span>
                  <span className="text-sm font-medium">
                    {processedData.chartData.filter(d => d.accessCount > 0).length}
                  </span>
                </div>
                <Progress 
                  value={processedData.totalEntries > 0 
                    ? (processedData.chartData.filter(d => d.accessCount > 0).length / processedData.totalEntries) * 100 
                    : 0
                  } 
                  className="h-2" 
                />
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Vector Efficiency</span>
                  <span className="text-sm font-medium">
                    {Math.round((processedData.averageVectorSize / 256) * 100)}%
                  </span>
                </div>
                <Progress 
                  value={Math.min((processedData.averageVectorSize / 256) * 100, 100)} 
                  className="h-2" 
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Memory Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {processedData.chartData.slice(0, 5).map((item, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                <div className="flex items-center gap-3">
                  <Badge variant="outline">{item.type}</Badge>
                  <span className="text-sm font-medium">{item.fullName}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {item.accessCount} accesses
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {item.lastAccessed.toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
            {processedData.chartData.length === 0 && (
              <p className="text-muted-foreground text-center py-4">No memory activity in selected time window</p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
