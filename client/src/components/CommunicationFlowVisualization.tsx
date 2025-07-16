import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MessageSquare, Filter, Eye, EyeOff } from "lucide-react";

interface CommunicationFlowVisualizationProps {
  communications: any[];
  agents: any[];
  timeWindow: number;
  selectedAgent?: string | null;
}

export function CommunicationFlowVisualization({
  communications,
  agents,
  timeWindow,
  selectedAgent
}: CommunicationFlowVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [filterType, setFilterType] = useState<string>("all");
  const [showLabels, setShowLabels] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(true);

  // Filter communications based on time window and type
  const filteredCommunications = communications.filter(comm => {
    const commTime = new Date(comm.timestamp).getTime();
    const cutoff = Date.now() - (timeWindow * 1000);
    
    if (commTime < cutoff) return false;
    if (filterType !== "all" && comm.messageType !== filterType) return false;
    if (selectedAgent && comm.fromAgentId !== selectedAgent && comm.toAgentId !== selectedAgent) return false;
    
    return true;
  });

  // Create agent position mapping
  const agentPositions = new Map<string, { x: number; y: number }>();
  agents.forEach((agent, index) => {
    const angle = (index / agents.length) * 2 * Math.PI;
    const radius = 150;
    const centerX = 300;
    const centerY = 200;
    
    agentPositions.set(agent.id, {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle)
    });
  });

  // Group communications by flow
  const communicationFlows = new Map<string, {
    from: string;
    to: string;
    count: number;
    type: string;
    lastTime: number;
  }>();

  filteredCommunications.forEach(comm => {
    const key = `${comm.fromAgentId}-${comm.toAgentId}`;
    const existing = communicationFlows.get(key);
    const commTime = new Date(comm.timestamp).getTime();
    
    if (existing) {
      existing.count++;
      existing.lastTime = Math.max(existing.lastTime, commTime);
    } else {
      communicationFlows.set(key, {
        from: comm.fromAgentId,
        to: comm.toAgentId,
        count: 1,
        type: comm.messageType,
        lastTime: commTime
      });
    }
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Animation loop
    let animationId: number;
    let lastTime = Date.now();

    const animate = () => {
      const currentTime = Date.now();
      const deltaTime = currentTime - lastTime;
      lastTime = currentTime;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw agents
      agents.forEach(agent => {
        const pos = agentPositions.get(agent.id);
        if (!pos) return;

        // Agent circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 15, 0, 2 * Math.PI);
        
        // Color based on type and selection
        if (agent.type === 'coordinator') {
          ctx.fillStyle = selectedAgent === agent.id ? "#ef4444" : "#fca5a5";
        } else {
          ctx.fillStyle = selectedAgent === agent.id ? "#3b82f6" : "#93c5fd";
        }
        
        ctx.fill();
        
        // Border
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Agent label
        if (showLabels) {
          ctx.fillStyle = "#000000";
          ctx.font = "10px Arial";
          ctx.textAlign = "center";
          ctx.fillText(agent.id.slice(-3), pos.x, pos.y + 25);
        }
      });

      // Draw communication flows
      communicationFlows.forEach(flow => {
        const fromPos = agentPositions.get(flow.from);
        const toPos = agentPositions.get(flow.to);
        
        if (!fromPos || !toPos) return;

        // Calculate flow intensity based on recency and count
        const age = (currentTime - flow.lastTime) / 1000; // seconds
        const intensity = Math.max(0, 1 - (age / timeWindow)) * Math.min(flow.count / 10, 1);
        
        if (intensity < 0.1) return;

        // Flow line
        ctx.beginPath();
        ctx.moveTo(fromPos.x, fromPos.y);
        ctx.lineTo(toPos.x, toPos.y);
        
        // Color based on message type
        const typeColors = {
          direct: "#3b82f6",
          pointer: "#10b981",
          broadcast: "#f59e0b"
        };
        
        ctx.strokeStyle = typeColors[flow.type as keyof typeof typeColors] || "#6b7280";
        ctx.lineWidth = Math.max(1, intensity * 4);
        ctx.globalAlpha = intensity;
        ctx.stroke();
        
        // Arrow head
        const angle = Math.atan2(toPos.y - fromPos.y, toPos.x - fromPos.x);
        const arrowLength = 10;
        const arrowAngle = 0.5;
        
        ctx.beginPath();
        ctx.moveTo(toPos.x, toPos.y);
        ctx.lineTo(
          toPos.x - arrowLength * Math.cos(angle - arrowAngle),
          toPos.y - arrowLength * Math.sin(angle - arrowAngle)
        );
        ctx.lineTo(
          toPos.x - arrowLength * Math.cos(angle + arrowAngle),
          toPos.y - arrowLength * Math.sin(angle + arrowAngle)
        );
        ctx.closePath();
        ctx.fill();
        
        // Flow count
        if (showLabels && flow.count > 1) {
          const midX = (fromPos.x + toPos.x) / 2;
          const midY = (fromPos.y + toPos.y) / 2;
          
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(midX - 8, midY - 8, 16, 16);
          ctx.fillStyle = "#000000";
          ctx.font = "10px Arial";
          ctx.textAlign = "center";
          ctx.fillText(flow.count.toString(), midX, midY + 3);
        }
        
        ctx.globalAlpha = 1;
      });

      if (isPlaying) {
        animationId = requestAnimationFrame(animate);
      }
    };

    if (isPlaying) {
      animate();
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [agents, communicationFlows, selectedAgent, showLabels, isPlaying, timeWindow]);

  const messageTypes = ["all", "direct", "pointer", "broadcast"];
  const totalFlows = communicationFlows.size;
  const avgFlowIntensity = totalFlows > 0 
    ? Array.from(communicationFlows.values()).reduce((sum, flow) => sum + flow.count, 0) / totalFlows
    : 0;

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4" />
          <span className="text-sm font-medium">Filter:</span>
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {messageTypes.map(type => (
                <SelectItem key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <Button
          size="sm"
          variant="outline"
          onClick={() => setShowLabels(!showLabels)}
        >
          {showLabels ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          Labels
        </Button>
        
        <Button
          size="sm"
          variant="outline"
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? "Pause" : "Play"}
        </Button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-2 bg-muted rounded">
          <div className="text-lg font-bold text-blue-600">{totalFlows}</div>
          <div className="text-xs text-muted-foreground">Active Flows</div>
        </div>
        <div className="text-center p-2 bg-muted rounded">
          <div className="text-lg font-bold text-green-600">{filteredCommunications.length}</div>
          <div className="text-xs text-muted-foreground">Messages</div>
        </div>
        <div className="text-center p-2 bg-muted rounded">
          <div className="text-lg font-bold text-purple-600">{avgFlowIntensity.toFixed(1)}</div>
          <div className="text-xs text-muted-foreground">Avg Intensity</div>
        </div>
      </div>

      {/* Visualization Canvas */}
      <div className="relative border rounded-lg">
        <canvas
          ref={canvasRef}
          width={600}
          height={400}
          className="w-full h-full"
        />
        
        {/* Legend */}
        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm p-3 rounded-lg border">
          <div className="text-sm font-semibold mb-2">Message Types</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-blue-500"></div>
              <span className="text-xs">Direct</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-green-500"></div>
              <span className="text-xs">Pointer</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-yellow-500"></div>
              <span className="text-xs">Broadcast</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
