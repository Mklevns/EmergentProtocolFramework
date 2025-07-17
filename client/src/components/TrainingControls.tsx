import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { TrainingStatus } from '@/lib/agent-types';
import { Play, Pause, Square, RotateCcw, Settings, Brain, Activity, Zap } from 'lucide-react';

interface TrainingControlsProps {
  experimentId: number | null;
  trainingStatus: TrainingStatus | undefined;
  onStart: () => void;
  onStop: () => void;
  onQuickStart: () => void;
  isStarting: boolean;
  isStopping: boolean;
  realtimeMetrics?: any;
}

export function TrainingControls({
  experimentId,
  trainingStatus,
  onStart,
  onStop,
  onQuickStart,
  isStarting,
  isStopping,
  realtimeMetrics
}: TrainingControlsProps) {
  const isRunning = trainingStatus?.isRunning || false;
  const canStart = !isRunning && !isStarting;
  const canStop = isRunning && !isStopping;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500';
      case 'completed': return 'bg-blue-500';
      case 'failed': return 'bg-red-500';
      case 'stopped': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return 'Running';
      case 'completed': return 'Completed';
      case 'failed': return 'Failed';
      case 'stopped': return 'Stopped';
      default: return 'Pending';
    }
  };

  return (
    <div className="space-y-4">
      {/* Control Buttons */}
      <div className="flex gap-2">
        <Button
          onClick={onStart}
          disabled={!canStart}
          className="flex-1"
          variant={canStart ? 'default' : 'secondary'}
        >
          <Play className="h-4 w-4 mr-2" />
          {isStarting ? 'Starting...' : 'Start Training'}
        </Button>
        
        <Button
          onClick={onQuickStart}
          disabled={!canStart}
          variant="outline"
          className="flex-1"
        >
          <Zap className="h-4 w-4 mr-2" />
          Quick Start
        </Button>
        
        <Button
          onClick={onStop}
          disabled={!canStop}
          variant="destructive"
          className="flex-1"
        >
          <Square className="h-4 w-4 mr-2" />
          {isStopping ? 'Stopping...' : 'Stop Training'}
        </Button>
      </div>

      {/* Status Information */}
      {trainingStatus && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Status:</span>
            <Badge variant={isRunning ? 'default' : 'secondary'}>
              <div className={`w-2 h-2 rounded-full mr-2 ${isRunning ? 'bg-green-500' : 'bg-gray-500'}`} />
              {isRunning ? 'Running' : 'Idle'}
            </Badge>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Episode:</span>
              <span className="font-medium">{trainingStatus.currentEpisode || 0}</span>
            </div>
            
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Step:</span>
              <span className="font-medium">{trainingStatus.currentStep || 0}</span>
            </div>
            
            {trainingStatus.experiment?.config && (
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Progress:</span>
                  <span className="font-medium">
                    {((trainingStatus.currentEpisode / (trainingStatus.experiment.config.total_episodes || 1000)) * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress 
                  value={(trainingStatus.currentEpisode / (trainingStatus.experiment.config.total_episodes || 1000)) * 100} 
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Training Configuration */}
      {trainingStatus?.experiment?.config && (
        <div className="space-y-2 p-3 bg-muted rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Configuration</span>
          </div>
          
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Episodes:</span>
              <span className="ml-1 font-medium">{trainingStatus.experiment.config.total_episodes || 'N/A'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Learning Rate:</span>
              <span className="ml-1 font-medium">{trainingStatus.experiment.config.learning_rate || 'N/A'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Batch Size:</span>
              <span className="ml-1 font-medium">{trainingStatus.experiment.config.batch_size || 'N/A'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Hidden Dim:</span>
              <span className="ml-1 font-medium">{trainingStatus.experiment.config.hidden_dim || 'N/A'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Training Actions */}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" disabled={isRunning} className="flex-1">
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset
        </Button>
        
        <Button variant="outline" size="sm" disabled={isRunning} className="flex-1">
          <Settings className="h-4 w-4 mr-2" />
          Configure
        </Button>
      </div>

      {/* Recent Metrics Summary */}
      {realtimeMetrics && (
        <div className="p-3 bg-muted rounded-lg">
          <div className="text-sm font-medium mb-2">Real-Time Metrics</div>
          <div className="space-y-1">
            {realtimeMetrics.pheromone_strength && (
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Pheromone Strength:</span>
                <span className="font-medium">{realtimeMetrics.pheromone_strength.toFixed(3)}</span>
              </div>
            )}
            {realtimeMetrics.neural_plasticity && (
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Neural Plasticity:</span>
                <span className="font-medium">{realtimeMetrics.neural_plasticity.toFixed(3)}</span>
              </div>
            )}
            {realtimeMetrics.swarm_coordination && (
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Swarm Coordination:</span>
                <span className="font-medium">{realtimeMetrics.swarm_coordination.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
