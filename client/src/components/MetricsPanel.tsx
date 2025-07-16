import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { TrainingStatus, Metric } from '@/lib/agent-types';
import { TrendingUp, TrendingDown, Activity, Zap, MessageSquare, Database } from 'lucide-react';

interface MetricsPanelProps {
  metrics: Metric[] | undefined;
  trainingStatus: TrainingStatus | undefined;
  timeRange: '1h' | '6h' | '24h' | '7d';
  onTimeRangeChange: (range: '1h' | '6h' | '24h' | '7d') => void;
}

export function MetricsPanel({ 
  metrics, 
  trainingStatus, 
  timeRange, 
  onTimeRangeChange 
}: MetricsPanelProps) {
  const [selectedMetric, setSelectedMetric] = useState<string>('communication_efficiency');

  // Process metrics data
  const processedMetrics = metrics?.reduce((acc, metric) => {
    if (!acc[metric.metricType]) {
      acc[metric.metricType] = [];
    }
    acc[metric.metricType].push(metric);
    return acc;
  }, {} as Record<string, Metric[]>) || {};

  // Calculate metric trends
  const getMetricTrend = (metricType: string) => {
    const metricData = processedMetrics[metricType];
    if (!metricData || metricData.length < 2) return 0;

    const recent = metricData.slice(-5);
    const older = metricData.slice(-10, -5);

    if (older.length === 0) return 0;

    const recentAvg = recent.reduce((sum, m) => sum + m.value, 0) / recent.length;
    const olderAvg = older.reduce((sum, m) => sum + m.value, 0) / older.length;

    return ((recentAvg - olderAvg) / olderAvg) * 100;
  };

  // Get current metric value
  const getCurrentMetricValue = (metricType: string) => {
    const metricData = processedMetrics[metricType];
    if (!metricData || metricData.length === 0) return 0;
    return metricData[metricData.length - 1].value;
  };

  // Render line chart
  const renderLineChart = (metricType: string) => {
    const metricData = processedMetrics[metricType];
    if (!metricData || metricData.length === 0) {
      return (
        <div className="w-full h-48 flex items-center justify-center text-muted-foreground">
          No data available
        </div>
      );
    }

    const maxValue = Math.max(...metricData.map(m => m.value));
    const minValue = Math.min(...metricData.map(m => m.value));
    const range = maxValue - minValue || 1;

    return (
      <div className="w-full h-48 relative">
        <svg width="100%" height="100%" viewBox="0 0 600 200">
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="60" height="40" patternUnits="userSpaceOnUse">
              <path d="M 60 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" opacity="0.3"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />

          {/* Data line */}
          <path
            d={metricData.map((metric, index) => {
              const x = (index / (metricData.length - 1)) * 580 + 10;
              const y = 180 - ((metric.value - minValue) / range) * 160;
              return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
            }).join(' ')}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
          />

          {/* Data points */}
          {metricData.map((metric, index) => {
            const x = (index / (metricData.length - 1)) * 580 + 10;
            const y = 180 - ((metric.value - minValue) / range) * 160;
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="3"
                fill="#3b82f6"
                className="hover:r-4 cursor-pointer"
              />
            );
          })}

          {/* Y-axis labels */}
          <text x="5" y="15" fontSize="10" fill="#6b7280">{maxValue.toFixed(1)}</text>
          <text x="5" y="105" fontSize="10" fill="#6b7280">{((maxValue + minValue) / 2).toFixed(1)}</text>
          <text x="5" y="195" fontSize="10" fill="#6b7280">{minValue.toFixed(1)}</text>
        </svg>
      </div>
    );
  };

  // Key metrics overview
  const keyMetrics = [
    {
      key: 'communication_efficiency',
      label: 'Communication Efficiency',
      icon: MessageSquare,
      color: 'text-blue-500',
      unit: '%'
    },
    {
      key: 'breakthrough_frequency',
      label: 'Breakthrough Frequency',
      icon: Zap,
      color: 'text-yellow-500',
      unit: '/min'
    },
    {
      key: 'memory_utilization',
      label: 'Memory Utilization',
      icon: Database,
      color: 'text-purple-500',
      unit: '%'
    },
    {
      key: 'coordination_success',
      label: 'Coordination Success',
      icon: Activity,
      color: 'text-green-500',
      unit: '%'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Time Range Selector */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {(['1h', '6h', '24h', '7d'] as const).map((range) => (
            <Button
              key={range}
              variant={timeRange === range ? 'default' : 'outline'}
              size="sm"
              onClick={() => onTimeRangeChange(range)}
            >
              {range}
            </Button>
          ))}
        </div>
        
        {trainingStatus && (
          <Badge variant={trainingStatus.isRunning ? 'default' : 'secondary'}>
            {trainingStatus.isRunning ? 'Training Active' : 'Training Stopped'}
          </Badge>
        )}
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {keyMetrics.map((metric) => {
          const currentValue = getCurrentMetricValue(metric.key);
          const trend = getMetricTrend(metric.key);
          const Icon = metric.icon;

          return (
            <Card key={metric.key}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className={`h-5 w-5 ${metric.color}`} />
                    <div>
                      <p className="text-sm text-muted-foreground">{metric.label}</p>
                      <p className="text-2xl font-bold">
                        {currentValue.toFixed(1)}{metric.unit}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    {trend > 0 ? (
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    ) : trend < 0 ? (
                      <TrendingDown className="h-4 w-4 text-red-500" />
                    ) : (
                      <div className="h-4 w-4" />
                    )}
                    <span className={`text-sm ${
                      trend > 0 ? 'text-green-600' : trend < 0 ? 'text-red-600' : 'text-gray-600'
                    }`}>
                      {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Training Progress */}
      {trainingStatus && (
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Episode Progress</span>
                <span className="text-sm font-medium">
                  {trainingStatus.currentEpisode} / {trainingStatus.experiment.config?.total_episodes || 1000}
                </span>
              </div>
              <Progress 
                value={(trainingStatus.currentEpisode / (trainingStatus.experiment.config?.total_episodes || 1000)) * 100} 
              />
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Current Step:</span>
                  <span className="ml-2 font-medium">{trainingStatus.currentStep}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Experiment:</span>
                  <span className="ml-2 font-medium">{trainingStatus.experiment.name}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Detailed Metrics */}
      <Tabs value={selectedMetric} onValueChange={setSelectedMetric}>
        <TabsList className="grid w-full grid-cols-4">
          {keyMetrics.map((metric) => (
            <TabsTrigger key={metric.key} value={metric.key}>
              {metric.label.split(' ')[0]}
            </TabsTrigger>
          ))}
        </TabsList>

        {keyMetrics.map((metric) => (
          <TabsContent key={metric.key} value={metric.key}>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <metric.icon className={`h-5 w-5 ${metric.color}`} />
                  {metric.label}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {renderLineChart(metric.key)}
                
                <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Current:</span>
                    <span className="ml-2 font-medium">
                      {getCurrentMetricValue(metric.key).toFixed(2)}{metric.unit}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Trend:</span>
                    <span className={`ml-2 font-medium ${
                      getMetricTrend(metric.key) > 0 ? 'text-green-600' : 
                      getMetricTrend(metric.key) < 0 ? 'text-red-600' : 'text-gray-600'
                    }`}>
                      {getMetricTrend(metric.key) > 0 ? '+' : ''}
                      {getMetricTrend(metric.key).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Data Points:</span>
                    <span className="ml-2 font-medium">
                      {processedMetrics[metric.key]?.length || 0}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      {/* Recent Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {trainingStatus?.recentMetrics.slice(-10).reverse().map((metric, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{metric.metricType}</Badge>
                  <span className="text-sm">Episode {metric.episode}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-medium">{metric.value.toFixed(3)}</span>
                  {metric.agentId && (
                    <Badge variant="secondary" className="text-xs">
                      {metric.agentId}
                    </Badge>
                  )}
                </div>
              </div>
            ))}
            
            {(!trainingStatus?.recentMetrics || trainingStatus.recentMetrics.length === 0) && (
              <div className="text-center py-8 text-muted-foreground">
                No recent metrics available
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
