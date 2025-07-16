import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useQuery } from '@tanstack/react-query';
import { 
  Eye, 
  Network, 
  Database, 
  Activity, 
  Download, 
  Play, 
  Pause,
  Maximize2,
  Settings
} from 'lucide-react';

import { AgentGrid3D } from '@/components/AgentGrid3D';
import { CommunicationFlow } from '@/components/CommunicationFlow';
import { MemoryVisualization } from '@/components/MemoryVisualization';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AgentGridData } from '@/lib/agent-types';

export default function Visualization() {
  const [selectedView, setSelectedView] = useState<string>('3d-grid');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'realtime' | 'playback'>('realtime');
  const [isFullscreen, setIsFullscreen] = useState(false);

  // WebSocket for real-time updates
  const { data: wsData, isConnected } = useWebSocket('/ws');

  // Grid data query
  const { data: gridData, isLoading } = useQuery<AgentGridData>({
    queryKey: ['/api/grid'],
    refetchInterval: viewMode === 'realtime' ? 2000 : false,
  });

  // Memory state query
  const { data: memoryState } = useQuery({
    queryKey: ['/api/memory'],
    refetchInterval: viewMode === 'realtime' ? 3000 : false,
  });

  // Communication patterns query
  const { data: commPatterns } = useQuery({
    queryKey: ['/api/communication-patterns'],
    refetchInterval: viewMode === 'realtime' ? 2000 : false,
  });

  const handlePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSpeedChange = (speed: string) => {
    setPlaybackSpeed(parseFloat(speed));
  };

  const handleExport = () => {
    // Export current visualization data
    const exportData = {
      timestamp: new Date().toISOString(),
      gridData,
      memoryState,
      commPatterns,
      selectedView,
      selectedAgent
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visualization_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleAgentSelect = (agentId: string) => {
    setSelectedAgent(agentId === selectedAgent ? null : agentId);
  };

  const toggleFullscreen = () => {
    if (!isFullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
    setIsFullscreen(!isFullscreen);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading visualization...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col ${isFullscreen ? 'fixed inset-0 z-50 bg-background' : 'min-h-screen'}`}>
      {/* Header */}
      <header className="bg-card border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Eye className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">3D Visualization</h1>
                <p className="text-sm text-muted-foreground">
                  Real-time MARL System Visualization
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* View Mode Toggle */}
            <div className="flex items-center gap-2">
              <Button
                variant={viewMode === 'realtime' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('realtime')}
              >
                Real-time
              </Button>
              <Button
                variant={viewMode === 'playback' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('playback')}
              >
                Playback
              </Button>
            </div>

            {/* Playback Controls */}
            {viewMode === 'playback' && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePlayback}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                
                <Select value={playbackSpeed.toString()} onValueChange={handleSpeedChange}>
                  <SelectTrigger className="w-20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0.5">0.5x</SelectItem>
                    <SelectItem value="1">1x</SelectItem>
                    <SelectItem value="2">2x</SelectItem>
                    <SelectItem value="4">4x</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleExport}>
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              
              <Button variant="outline" size="sm" onClick={toggleFullscreen}>
                <Maximize2 className="h-4 w-4 mr-2" />
                {isFullscreen ? 'Exit' : 'Fullscreen'}
              </Button>
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-muted-foreground">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Sidebar */}
        <div className="w-80 bg-card border-r border-border p-4 space-y-4">
          {/* View Selector */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Visualization Mode</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button
                variant={selectedView === '3d-grid' ? 'default' : 'outline'}
                className="w-full justify-start"
                onClick={() => setSelectedView('3d-grid')}
              >
                <Network className="h-4 w-4 mr-2" />
                3D Agent Grid
              </Button>
              
              <Button
                variant={selectedView === 'communication' ? 'default' : 'outline'}
                className="w-full justify-start"
                onClick={() => setSelectedView('communication')}
              >
                <Activity className="h-4 w-4 mr-2" />
                Communication Flow
              </Button>
              
              <Button
                variant={selectedView === 'memory' ? 'default' : 'outline'}
                className="w-full justify-start"
                onClick={() => setSelectedView('memory')}
              >
                <Database className="h-4 w-4 mr-2" />
                Memory Visualization
              </Button>
            </CardContent>
          </Card>

          {/* System Stats */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">System Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Active Agents:</span>
                <Badge variant="secondary">
                  {gridData?.agents.filter(a => a.isActive).length || 0}
                </Badge>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Active Messages:</span>
                <Badge variant="secondary">
                  {gridData?.activeMessages.length || 0}
                </Badge>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Breakthroughs:</span>
                <Badge variant="secondary">
                  {gridData?.recentBreakthroughs.length || 0}
                </Badge>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Memory Usage:</span>
                <Badge variant="secondary">
                  {memoryState?.usage ? 
                    `${((memoryState.usage.used / memoryState.usage.total) * 100).toFixed(1)}%` : 
                    'N/A'
                  }
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Selected Agent Details */}
          {selectedAgent && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Agent Details</CardTitle>
              </CardHeader>
              <CardContent>
                {(() => {
                  const agent = gridData?.agents.find(a => a.agentId === selectedAgent);
                  if (!agent) return <p className="text-sm text-muted-foreground">Agent not found</p>;
                  
                  return (
                    <div className="space-y-2">
                      <div>
                        <h4 className="font-semibold">{agent.agentId}</h4>
                        <Badge variant="outline">{agent.type}</Badge>
                      </div>
                      
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Status:</span>
                          <Badge variant="secondary">{agent.status}</Badge>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Position:</span>
                          <span>({agent.positionX}, {agent.positionY}, {agent.positionZ})</span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Coordinator:</span>
                          <span>{agent.coordinatorId || 'None'}</span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Hidden Dim:</span>
                          <span>{agent.hiddenDim}</span>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </CardContent>
            </Card>
          )}

          {/* Visualization Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Auto-rotate:</span>
                <input type="checkbox" className="rounded" />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Show grid:</span>
                <input type="checkbox" className="rounded" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Animation:</span>
                <input type="checkbox" className="rounded" defaultChecked />
              </div>
              
              <div className="space-y-1">
                <span className="text-sm">Zoom level:</span>
                <input 
                  type="range" 
                  min="0.5" 
                  max="3" 
                  step="0.1" 
                  defaultValue="1"
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Visualization Area */}
        <div className="flex-1 p-4">
          {selectedView === '3d-grid' && (
            <Card className="h-full">
              <CardHeader>
                <CardTitle>3D Agent Grid Visualization</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <AgentGrid3D
                  agents={gridData?.agents || []}
                  communicationPatterns={gridData?.communicationPatterns || []}
                  onAgentSelect={handleAgentSelect}
                  selectedAgent={selectedAgent}
                />
              </CardContent>
            </Card>
          )}

          {selectedView === 'communication' && (
            <Card className="h-full">
              <CardHeader>
                <CardTitle>Communication Flow Visualization</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <CommunicationFlow
                  agents={gridData?.agents || []}
                  messages={gridData?.activeMessages || []}
                  patterns={commPatterns || []}
                />
              </CardContent>
            </Card>
          )}

          {selectedView === 'memory' && (
            <Card className="h-full">
              <CardHeader>
                <CardTitle>Memory System Visualization</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <MemoryVisualization
                  memoryState={memoryState}
                  agents={gridData?.agents || []}
                />
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
