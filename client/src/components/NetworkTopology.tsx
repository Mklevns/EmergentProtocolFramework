import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Network, Maximize2, Minimize2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

interface NetworkTopologyProps {
  agents: any[];
  communications: any[];
  selectedAgent?: string | null;
  onAgentClick?: (agent: any) => void;
  className?: string;
}

export function NetworkTopology({ 
  agents, 
  communications, 
  selectedAgent, 
  onAgentClick,
  className 
}: NetworkTopologyProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [layout, setLayout] = useState<"force" | "grid" | "hierarchical">("force");
  const [showConnections, setShowConnections] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  // Calculate agent positions based on layout
  const agentPositions = new Map<string, { x: number; y: number }>();
  
  useEffect(() => {
    const centerX = 300;
    const centerY = 200;
    
    agents.forEach((agent, index) => {
      let x, y;
      
      switch (layout) {
        case "grid":
          // 3D grid flattened to 2D
          const gridX = agent.position.x;
          const gridY = agent.position.y;
          const gridZ = agent.position.z;
          x = centerX + (gridX - 2) * 80 + (gridZ - 1) * 20;
          y = centerY + (gridY - 1) * 80 + (gridZ - 1) * 15;
          break;
          
        case "hierarchical":
          // Coordinators at top, regular agents below
          if (agent.type === 'coordinator') {
            x = centerX + (agent.region - 1) * 150;
            y = centerY - 100;
          } else {
            const angle = (index / agents.length) * 2 * Math.PI;
            const radius = 120;
            x = centerX + radius * Math.cos(angle);
            y = centerY + 50 + radius * Math.sin(angle);
          }
          break;
          
        case "force":
        default:
          // Force-directed layout simulation
          const angle = (index / agents.length) * 2 * Math.PI;
          const radius = agent.type === 'coordinator' ? 80 : 150;
          x = centerX + radius * Math.cos(angle);
          y = centerY + radius * Math.sin(angle);
          break;
      }
      
      agentPositions.set(agent.id, { x, y });
    });
  }, [agents, layout]);

  // Group communications by connection
  const connections = new Map<string, {
    from: string;
    to: string;
    count: number;
    types: Set<string>;
    strength: number;
  }>();

  communications.forEach(comm => {
    const key = `${comm.fromAgentId}-${comm.toAgentId}`;
    const existing = connections.get(key);
    
    if (existing) {
      existing.count++;
      existing.types.add(comm.messageType);
      existing.strength = Math.min(existing.count / 10, 1);
    } else {
      connections.set(key, {
        from: comm.fromAgentId,
        to: comm.toAgentId,
        count: 1,
        types: new Set([comm.messageType]),
        strength: 0.1
      });
    }
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply zoom and pan
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // Draw connections
    if (showConnections) {
      connections.forEach(connection => {
        const fromPos = agentPositions.get(connection.from);
        const toPos = agentPositions.get(connection.to);
        
        if (!fromPos || !toPos) return;

        // Connection line
        ctx.beginPath();
        ctx.moveTo(fromPos.x, fromPos.y);
        ctx.lineTo(toPos.x, toPos.y);
        
        // Color based on connection strength and type
        const hasPointer = connection.types.has('pointer');
        const hasBroadcast = connection.types.has('broadcast');
        
        if (hasPointer) {
          ctx.strokeStyle = "#10b981";
        } else if (hasBroadcast) {
          ctx.strokeStyle = "#f59e0b";
        } else {
          ctx.strokeStyle = "#3b82f6";
        }
        
        ctx.lineWidth = Math.max(1, connection.strength * 4);
        ctx.globalAlpha = Math.max(0.3, connection.strength);
        ctx.stroke();
        
        // Connection weight indicator
        if (connection.count > 1) {
          const midX = (fromPos.x + toPos.x) / 2;
          const midY = (fromPos.y + toPos.y) / 2;
          
          ctx.globalAlpha = 1;
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(midX - 8, midY - 8, 16, 16);
          ctx.strokeStyle = "#000000";
          ctx.lineWidth = 1;
          ctx.strokeRect(midX - 8, midY - 8, 16, 16);
          
          ctx.fillStyle = "#000000";
          ctx.font = "10px Arial";
          ctx.textAlign = "center";
          ctx.fillText(connection.count.toString(), midX, midY + 3);
        }
      });
    }

    ctx.globalAlpha = 1;

    // Draw agents
    agents.forEach(agent => {
      const pos = agentPositions.get(agent.id);
      if (!pos) return;

      const isSelected = selectedAgent === agent.id;
      const isCoordinator = agent.type === 'coordinator';
      const radius = isCoordinator ? 15 : 10;
      
      // Agent circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      
      // Color based on type and activity
      let fillColor;
      if (isCoordinator) {
        fillColor = agent.isActive ? "#ef4444" : "#fca5a5";
      } else {
        fillColor = agent.isActive ? "#3b82f6" : "#93c5fd";
      }
      
      if (isSelected) {
        ctx.fillStyle = "#fbbf24";
        ctx.fill();
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 3;
        ctx.stroke();
      } else {
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Region indicator
      if (agent.region !== undefined) {
        const regionColors = ["#fbbf24", "#34d399", "#a78bfa"];
        ctx.fillStyle = regionColors[agent.region] || "#9ca3af";
        ctx.beginPath();
        ctx.arc(pos.x + radius - 5, pos.y - radius + 5, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Agent label
      ctx.fillStyle = "#000000";
      ctx.font = "10px Arial";
      ctx.textAlign = "center";
      ctx.fillText(agent.id.slice(-3), pos.x, pos.y + radius + 15);
    });

    ctx.restore();
  }, [agents, connections, selectedAgent, showConnections, zoom, pan, agentPositions]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    const deltaX = e.clientX - lastMousePos.x;
    const deltaY = e.clientY - lastMousePos.y;

    setPan(prev => ({
      x: prev.x + deltaX,
      y: prev.y + deltaY
    }));

    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (isDragging || !onAgentClick) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left - pan.x) / zoom;
    const clickY = (e.clientY - rect.top - pan.y) / zoom;

    // Find clicked agent
    for (const agent of agents) {
      const pos = agentPositions.get(agent.id);
      if (!pos) continue;

      const radius = agent.type === 'coordinator' ? 15 : 10;
      const distance = Math.sqrt(
        Math.pow(clickX - pos.x, 2) + Math.pow(clickY - pos.y, 2)
      );

      if (distance <= radius) {
        onAgentClick(agent);
        break;
      }
    }
  };

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.2, 0.5));
  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const coordinatorCount = agents.filter(a => a.type === 'coordinator').length;
  const activeCount = agents.filter(a => a.isActive).length;
  const connectionCount = connections.size;

  return (
    <Card className={`${className} ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Network Topology
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{agents.length} nodes</Badge>
            <Badge variant="outline">{connectionCount} edges</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Controls */}
        <div className="flex items-center gap-4 mb-4 flex-wrap">
          <Select value={layout} onValueChange={(value: "force" | "grid" | "hierarchical") => setLayout(value)}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="force">Force</SelectItem>
              <SelectItem value="grid">Grid</SelectItem>
              <SelectItem value="hierarchical">Hierarchical</SelectItem>
            </SelectContent>
          </Select>
          
          <Button
            size="sm"
            variant="outline"
            onClick={() => setShowConnections(!showConnections)}
          >
            {showConnections ? "Hide" : "Show"} Connections
          </Button>
          
          <div className="flex items-center gap-1">
            <Button size="sm" variant="outline" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={handleResetView}>
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="outline" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
          </div>
          
          <Button size="sm" variant="outline" onClick={() => setIsFullscreen(!isFullscreen)}>
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </Button>
        </div>

        {/* Canvas */}
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={600}
            height={400}
            className="border rounded-lg cursor-move w-full"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleCanvasClick}
          />
          
          {/* Legend */}
          <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm p-3 rounded-lg border">
            <div className="text-sm font-semibold mb-2">Legend</div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                <span className="text-sm">Regular Agent</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-500"></div>
                <span className="text-sm">Coordinator</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
                <span className="text-sm">Selected</span>
              </div>
            </div>
          </div>

          {/* Connection Legend */}
          {showConnections && (
            <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm p-3 rounded-lg border">
              <div className="text-sm font-semibold mb-2">Connections</div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-blue-500"></div>
                  <span className="text-sm">Direct</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-green-500"></div>
                  <span className="text-sm">Pointer</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-yellow-500"></div>
                  <span className="text-sm">Broadcast</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-3 gap-4 mt-4">
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-lg font-bold text-green-600">{activeCount}</div>
            <div className="text-xs text-muted-foreground">Active Agents</div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-lg font-bold text-red-600">{coordinatorCount}</div>
            <div className="text-xs text-muted-foreground">Coordinators</div>
          </div>
          <div className="text-center p-2 bg-muted rounded">
            <div className="text-lg font-bold text-blue-600">{connectionCount}</div>
            <div className="text-xs text-muted-foreground">Connections</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
