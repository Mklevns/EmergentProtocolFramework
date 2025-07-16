import { useRef, useEffect, useState } from 'react';
import { Agent, CommunicationPattern } from '@/lib/agent-types';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Maximize2, Minimize2, RotateCw, Zap } from 'lucide-react';

interface AgentGrid3DProps {
  agents: Agent[];
  communicationPatterns: CommunicationPattern[];
  onAgentSelect: (agentId: string) => void;
  selectedAgent?: string | null;
}

export function AgentGrid3D({ 
  agents, 
  communicationPatterns, 
  onAgentSelect, 
  selectedAgent 
}: AgentGrid3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Set up 3D projection
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const scale = 40 * zoom;

      // Draw grid lines
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3;

      // Draw 3D grid structure
      for (let x = 0; x < 4; x++) {
        for (let y = 0; y < 3; y++) {
          for (let z = 0; z < 3; z++) {
            const projected = project3D(x, y, z, centerX, centerY, scale);
            
            // Draw grid connections
            if (x < 3) {
              const nextX = project3D(x + 1, y, z, centerX, centerY, scale);
              drawLine(ctx, projected, nextX);
            }
            if (y < 2) {
              const nextY = project3D(x, y + 1, z, centerX, centerY, scale);
              drawLine(ctx, projected, nextY);
            }
            if (z < 2) {
              const nextZ = project3D(x, y, z + 1, centerX, centerY, scale);
              drawLine(ctx, projected, nextZ);
            }
          }
        }
      }

      ctx.globalAlpha = 1;

      // Draw communication patterns
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.6;

      communicationPatterns.forEach(pattern => {
        const fromAgent = agents.find(a => a.agentId === pattern.fromAgentId);
        const toAgent = agents.find(a => a.agentId === pattern.toAgentId);

        if (fromAgent && toAgent) {
          const fromPos = project3D(fromAgent.positionX, fromAgent.positionY, fromAgent.positionZ, centerX, centerY, scale);
          const toPos = project3D(toAgent.positionX, toAgent.positionY, toAgent.positionZ, centerX, centerY, scale);

          // Draw communication line with intensity based on frequency
          const intensity = Math.min(pattern.frequency / 10, 1);
          ctx.globalAlpha = intensity * 0.8;
          drawLine(ctx, fromPos, toPos);

          // Draw arrow
          drawArrow(ctx, fromPos, toPos);
        }
      });

      ctx.globalAlpha = 1;

      // Draw agents
      agents.forEach(agent => {
        const pos = project3D(agent.positionX, agent.positionY, agent.positionZ, centerX, centerY, scale);
        drawAgent(ctx, pos, agent, selectedAgent === agent.agentId);
      });

      // Draw legend
      drawLegend(ctx, canvas.width - 200, 20);
    };

    const project3D = (x: number, y: number, z: number, centerX: number, centerY: number, scale: number) => {
      // Apply rotation
      const rotX = rotation.x * Math.PI / 180;
      const rotY = rotation.y * Math.PI / 180;

      // Rotate around Y axis
      const cos_y = Math.cos(rotY);
      const sin_y = Math.sin(rotY);
      const x1 = x * cos_y - z * sin_y;
      const z1 = x * sin_y + z * cos_y;

      // Rotate around X axis
      const cos_x = Math.cos(rotX);
      const sin_x = Math.sin(rotX);
      const y1 = y * cos_x - z1 * sin_x;
      const z2 = y * sin_x + z1 * cos_x;

      // Project to 2D
      const perspective = 1 / (1 + z2 * 0.1);
      return {
        x: centerX + x1 * scale * perspective,
        y: centerY + y1 * scale * perspective,
        z: z2
      };
    };

    const drawLine = (ctx: CanvasRenderingContext2D, from: any, to: any) => {
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    };

    const drawArrow = (ctx: CanvasRenderingContext2D, from: any, to: any) => {
      const angle = Math.atan2(to.y - from.y, to.x - from.x);
      const arrowLength = 10;
      const arrowAngle = 0.5;

      ctx.beginPath();
      ctx.moveTo(to.x, to.y);
      ctx.lineTo(
        to.x - arrowLength * Math.cos(angle - arrowAngle),
        to.y - arrowLength * Math.sin(angle - arrowAngle)
      );
      ctx.moveTo(to.x, to.y);
      ctx.lineTo(
        to.x - arrowLength * Math.cos(angle + arrowAngle),
        to.y - arrowLength * Math.sin(angle + arrowAngle)
      );
      ctx.stroke();
    };

    const drawAgent = (ctx: CanvasRenderingContext2D, pos: any, agent: Agent, isSelected: boolean) => {
      const radius = agent.type === 'coordinator' ? 8 : 6;
      
      // Agent color based on type and status
      let color = '#64748b'; // Default gray
      if (agent.type === 'coordinator') {
        color = '#f59e0b'; // Amber for coordinators
      } else if (agent.status === 'breakthrough') {
        color = '#ef4444'; // Red for breakthroughs
      } else if (agent.status === 'communicating') {
        color = '#3b82f6'; // Blue for communication
      } else if (agent.status === 'processing') {
        color = '#8b5cf6'; // Purple for processing
      }

      // Draw agent circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // Draw selection ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 3, 0, 2 * Math.PI);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw agent border
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Draw agent ID
      ctx.fillStyle = '#000';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(agent.agentId.split('_')[1], pos.x, pos.y + radius + 12);
    };

    const drawLegend = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(x - 10, y - 10, 180, 120);
      ctx.strokeStyle = '#e2e8f0';
      ctx.strokeRect(x - 10, y - 10, 180, 120);

      ctx.fillStyle = '#1f2937';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Legend:', x, y + 10);

      const items = [
        { color: '#f59e0b', label: 'Coordinator', size: 8 },
        { color: '#64748b', label: 'Regular Agent', size: 6 },
        { color: '#ef4444', label: 'Breakthrough', size: 6 },
        { color: '#3b82f6', label: 'Communicating', size: 6 },
        { color: '#8b5cf6', label: 'Processing', size: 6 }
      ];

      items.forEach((item, index) => {
        const itemY = y + 30 + index * 16;
        ctx.beginPath();
        ctx.arc(x + 10, itemY, item.size, 0, 2 * Math.PI);
        ctx.fillStyle = item.color;
        ctx.fill();
        ctx.strokeStyle = '#1f2937';
        ctx.stroke();

        ctx.fillStyle = '#1f2937';
        ctx.fillText(item.label, x + 25, itemY + 4);
      });
    };

    // Set canvas size
    const resizeCanvas = () => {
      const container = canvas.parentElement;
      if (container) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        render();
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      setIsDragging(true);
      setLastMousePos({ x: e.clientX, y: e.clientY });
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const deltaX = e.clientX - lastMousePos.x;
      const deltaY = e.clientY - lastMousePos.y;

      setRotation(prev => ({
        x: prev.x + deltaY * 0.5,
        y: prev.y + deltaX * 0.5
      }));

      setLastMousePos({ x: e.clientX, y: e.clientY });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      setZoom(prev => Math.max(0.5, Math.min(3, prev + e.deltaY * -0.001)));
    };

    const handleClick = (e: MouseEvent) => {
      if (isDragging) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Find clicked agent
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const scale = 40 * zoom;

      for (const agent of agents) {
        const pos = project3D(agent.positionX, agent.positionY, agent.positionZ, centerX, centerY, scale);
        const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
        const radius = agent.type === 'coordinator' ? 8 : 6;

        if (distance <= radius) {
          onAgentSelect(agent.agentId);
          break;
        }
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('wheel', handleWheel);
    canvas.addEventListener('click', handleClick);

    render();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('wheel', handleWheel);
      canvas.removeEventListener('click', handleClick);
    };
  }, [agents, communicationPatterns, selectedAgent, rotation, zoom, isDragging]);

  const handleAutoRotate = () => {
    setRotation(prev => ({
      x: prev.x + 5,
      y: prev.y + 5
    }));
  };

  const handleResetView = () => {
    setRotation({ x: 0, y: 0 });
    setZoom(1);
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-background' : 'h-96'}`}>
      <div className="absolute top-2 right-2 z-10 flex gap-2">
        <Button variant="outline" size="sm" onClick={handleAutoRotate}>
          <RotateCw className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="sm" onClick={handleResetView}>
          Reset
        </Button>
        <Button variant="outline" size="sm" onClick={toggleFullscreen}>
          {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
        </Button>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full h-full border rounded cursor-move"
        style={{ background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)' }}
      />

      <div className="absolute bottom-2 left-2 text-xs text-muted-foreground">
        <p>Drag to rotate • Scroll to zoom • Click agents to select</p>
        <p>Agents: {agents.length} • Communications: {communicationPatterns.length}</p>
      </div>
    </div>
  );
}
