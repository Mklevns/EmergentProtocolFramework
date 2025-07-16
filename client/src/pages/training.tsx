import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Play, Pause, Square, Settings, FileText, Download, Upload } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

import { TrainingControls } from "@/components/TrainingControls";
import { MetricsPanel } from "@/components/MetricsPanel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { TrainingStatus, Experiment } from "@/lib/agent-types";
import { apiRequest } from "@/lib/queryClient";

export default function Training() {
  const [selectedExperiment, setSelectedExperiment] = useState<number | null>(null);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);
  const [newExperimentConfig, setNewExperimentConfig] = useState({
    name: "",
    description: "",
    total_episodes: 1000,
    learning_rate: 0.001,
    batch_size: 32,
    hidden_dim: 256,
    breakthrough_threshold: 0.7,
  });
  
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // WebSocket for real-time training updates
  const { data: wsData, isConnected } = useWebSocket('/api/ws');
  
  // Experiments query
  const { data: experiments, isLoading: experimentsLoading } = useQuery<Experiment[]>({
    queryKey: ['/api/experiments'],
    refetchInterval: 5000,
  });
  
  // Training status query
  const { data: trainingStatus } = useQuery<TrainingStatus>({
    queryKey: ['/api/experiments', selectedExperiment, 'status'],
    enabled: selectedExperiment !== null,
    refetchInterval: 2000,
  });
  
  // Metrics query
  const { data: metrics } = useQuery({
    queryKey: ['/api/metrics/experiment', selectedExperiment],
    enabled: selectedExperiment !== null,
    refetchInterval: 1000,
  });
  
  // Create experiment mutation
  const createExperimentMutation = useMutation({
    mutationFn: async (config: any) => {
      const response = await apiRequest('POST', '/api/experiments', config);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      toast({
        title: "Experiment Created",
        description: "New training experiment has been created successfully.",
      });
      setIsConfigModalOpen(false);
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to create experiment: ${error.message}`,
        variant: "destructive",
      });
    },
  });
  
  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: async (experimentId: number) => {
      const response = await apiRequest('POST', '/api/training/start', {
        experimentId,
        config: newExperimentConfig,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      toast({
        title: "Training Started",
        description: "Training has been started successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to start training: ${error.message}`,
        variant: "destructive",
      });
    },
  });
  
  // Stop training mutation
  const stopTrainingMutation = useMutation({
    mutationFn: async (experimentId: number) => {
      const response = await apiRequest('POST', '/api/training/stop', {
        experimentId,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      toast({
        title: "Training Stopped",
        description: "Training has been stopped successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to stop training: ${error.message}`,
        variant: "destructive",
      });
    },
  });
  
  // Initialize demo mutation
  const initializeDemoMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/initialize-demo', {});
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/grid'] });
      toast({
        title: "Demo Initialized",
        description: "Agent grid has been initialized with demo data.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to initialize demo: ${error.message}`,
        variant: "destructive",
      });
    },
  });
  
  const handleCreateExperiment = () => {
    if (!newExperimentConfig.name) {
      toast({
        title: "Error",
        description: "Please enter an experiment name.",
        variant: "destructive",
      });
      return;
    }
    
    createExperimentMutation.mutate({
      name: newExperimentConfig.name,
      description: newExperimentConfig.description,
      config: newExperimentConfig,
    });
  };
  
  const handleStartTraining = () => {
    if (selectedExperiment) {
      startTrainingMutation.mutate(selectedExperiment);
    }
  };
  
  const handleStopTraining = () => {
    if (selectedExperiment) {
      stopTrainingMutation.mutate(selectedExperiment);
    }
  };
  
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
    <div className="flex flex-col min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Training Management</h1>
            <p className="text-sm text-muted-foreground">
              Manage experiments and monitor training progress
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              onClick={() => initializeDemoMutation.mutate()}
              disabled={initializeDemoMutation.isPending}
            >
              <Settings className="h-4 w-4 mr-2" />
              Initialize Demo
            </Button>
            
            <Button 
              onClick={() => setIsConfigModalOpen(true)}
              disabled={createExperimentMutation.isPending}
            >
              <FileText className="h-4 w-4 mr-2" />
              New Experiment
            </Button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="flex-1 p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Experiments List */}
          <Card>
            <CardHeader>
              <CardTitle>Experiments</CardTitle>
            </CardHeader>
            <CardContent>
              {experimentsLoading ? (
                <div className="space-y-2">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="h-16 bg-muted rounded animate-pulse" />
                  ))}
                </div>
              ) : (
                <div className="space-y-2">
                  {experiments?.map((experiment) => (
                    <div
                      key={experiment.id}
                      className={`p-3 rounded cursor-pointer transition-colors ${
                        selectedExperiment === experiment.id
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted hover:bg-muted/80'
                      }`}
                      onClick={() => setSelectedExperiment(experiment.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium">{experiment.name}</h4>
                          <p className="text-sm opacity-70">{experiment.description}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${getStatusColor(experiment.status)}`} />
                          <span className="text-xs">{getStatusText(experiment.status)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {experiments?.length === 0 && (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      No experiments yet. Create your first experiment to get started.
                    </p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Training Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Training Controls</CardTitle>
            </CardHeader>
            <CardContent>
              {selectedExperiment ? (
                <TrainingControls
                  experimentId={selectedExperiment}
                  trainingStatus={trainingStatus}
                  onStart={handleStartTraining}
                  onStop={handleStopTraining}
                  isStarting={startTrainingMutation.isPending}
                  isStopping={stopTrainingMutation.isPending}
                />
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  Select an experiment to control training
                </p>
              )}
            </CardContent>
          </Card>
          
          {/* Training Status */}
          <Card>
            <CardHeader>
              <CardTitle>Training Status</CardTitle>
            </CardHeader>
            <CardContent>
              {trainingStatus ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Status:</span>
                    <Badge variant={trainingStatus.isRunning ? "default" : "secondary"}>
                      {trainingStatus.isRunning ? "Running" : "Stopped"}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Episode:</span>
                      <span>{trainingStatus.currentEpisode}</span>
                    </div>
                    
                    <div className="flex justify-between text-sm">
                      <span>Step:</span>
                      <span>{trainingStatus.currentStep}</span>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>Progress:</span>
                        <span>{((trainingStatus.currentEpisode / 1000) * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={(trainingStatus.currentEpisode / 1000) * 100} />
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No training status available
                </p>
              )}
            </CardContent>
          </Card>
        </div>
        
        {/* Metrics Panel */}
        {selectedExperiment && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Training Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <MetricsPanel 
                metrics={metrics}
                trainingStatus={trainingStatus}
                timeRange="1h"
                onTimeRangeChange={() => {}}
              />
            </CardContent>
          </Card>
        )}
      </div>
      
      {/* Create Experiment Modal */}
      {isConfigModalOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <CardHeader>
              <CardTitle>Create New Experiment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="name">Experiment Name</Label>
                <Input
                  id="name"
                  value={newExperimentConfig.name}
                  onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Enter experiment name"
                />
              </div>
              
              <div>
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  value={newExperimentConfig.description}
                  onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Enter experiment description"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="total_episodes">Total Episodes</Label>
                  <Input
                    id="total_episodes"
                    type="number"
                    value={newExperimentConfig.total_episodes}
                    onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, total_episodes: parseInt(e.target.value) }))}
                  />
                </div>
                
                <div>
                  <Label htmlFor="learning_rate">Learning Rate</Label>
                  <Input
                    id="learning_rate"
                    type="number"
                    step="0.0001"
                    value={newExperimentConfig.learning_rate}
                    onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="batch_size">Batch Size</Label>
                  <Input
                    id="batch_size"
                    type="number"
                    value={newExperimentConfig.batch_size}
                    onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                  />
                </div>
                
                <div>
                  <Label htmlFor="hidden_dim">Hidden Dimension</Label>
                  <Input
                    id="hidden_dim"
                    type="number"
                    value={newExperimentConfig.hidden_dim}
                    onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, hidden_dim: parseInt(e.target.value) }))}
                  />
                </div>
              </div>
              
              <div>
                <Label htmlFor="breakthrough_threshold">Breakthrough Threshold</Label>
                <Input
                  id="breakthrough_threshold"
                  type="number"
                  step="0.1"
                  min="0"
                  max="1"
                  value={newExperimentConfig.breakthrough_threshold}
                  onChange={(e) => setNewExperimentConfig(prev => ({ ...prev, breakthrough_threshold: parseFloat(e.target.value) }))}
                />
              </div>
              
              <div className="flex justify-end gap-2">
                <Button 
                  variant="outline" 
                  onClick={() => setIsConfigModalOpen(false)}
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handleCreateExperiment}
                  disabled={createExperimentMutation.isPending}
                >
                  {createExperimentMutation.isPending ? "Creating..." : "Create Experiment"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
