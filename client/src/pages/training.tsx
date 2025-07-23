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
import { Play, Pause, Square, Settings, FileText, Download, Upload, AlertTriangle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import { TrainingControls } from "@/components/TrainingControls";
import { MetricsPanel } from "@/components/MetricsPanel";
import { useWebSocket } from "@/hooks/useWebSocket";
import { TrainingStatus, Experiment, Metric } from "@/lib/agent-types";
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
  
  const [rayConfig, setRayConfig] = useState({
    name: "",
    description: "",
    total_episodes: 100,
    learning_rate: 0.0003,
    batch_size: 128,
    train_batch_size: 4000,
    hidden_dim: 256,
    num_rollout_workers: 4,
    num_attention_heads: 8,
    pheromone_decay: 0.95,
    neural_plasticity_rate: 0.1,
    communication_range: 2.0,
    breakthrough_threshold: 0.7,
  });
  
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // WebSocket for real-time training updates
  const { data: wsData, isConnected } = useWebSocket('/api/ws');
  
  // Load Ray config template on component mount
  useEffect(() => {
    rayConfigTemplateMutation.mutate();
  }, []);

  // Handle WebSocket training updates
  useEffect(() => {
    if (wsData?.type === 'training_started' || wsData?.type === 'ray_training_started') {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
    }
    if (wsData?.type === 'training_metrics' || wsData?.type === 'ray_training_metrics') {
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
    }
    if (wsData?.type === 'training_completed' || wsData?.type === 'training_failed' || 
        wsData?.type === 'ray_training_completed' || wsData?.type === 'ray_training_failed') {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
      
      // Show completion toast
      if (wsData?.type === 'training_completed' || wsData?.type === 'ray_training_completed') {
        toast({
          title: "Training Completed",
          description: wsData?.type === 'ray_training_completed' ? 
            "Ray RLlib training completed successfully!" : 
            "Training completed successfully!",
        });
      } else if (wsData?.type === 'training_failed' || wsData?.type === 'ray_training_failed') {
        toast({
          title: "Training Failed",
          description: "Training session encountered an error.",
          variant: "destructive",
        });
      }
    }
  }, [wsData, queryClient, toast]);
  
  // Experiments query
  const { data: experiments, isLoading: experimentsLoading } = useQuery<Experiment[]>({
    queryKey: ['/api/experiments'],
    refetchInterval: 5000,
  });
  
  // Training status query
  const { data: trainingStatus } = useQuery<TrainingStatus>({
    queryKey: ['/api/training/status'],
    refetchInterval: 2000,
  });
  
  // Metrics query
  const { data: metrics } = useQuery<Metric[]>({
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
    mutationFn: async (config: any) => {
      const response = await apiRequest('POST', '/api/training/start', {
        experimentId: selectedExperiment,
        config,
      });
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
      setSelectedExperiment(data.experimentId);
      toast({
        title: "Training Started",
        description: "Bio-inspired MARL training has been started successfully.",
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

  // Ray training mutations
  const rayConfigTemplateMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('GET', '/api/training/ray/config-template');
      return response.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        setRayConfig(data.config_template);
        toast({
          title: "Ray Config Loaded",
          description: "Ray RLlib configuration template loaded successfully.",
        });
      }
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to load Ray config: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  const startRayTrainingMutation = useMutation({
    mutationFn: async (config: any) => {
      const response = await apiRequest('POST', '/api/training/ray/start', { config });
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
      setSelectedExperiment(data.experimentId);
      toast({
        title: "Ray Training Started",
        description: "Ray RLlib training has been started successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to start Ray training: ${error.message}`,
        variant: "destructive",
      });
    },
  });

  // Quick start training mutation
  const quickStartMutation = useMutation({
    mutationFn: async () => {
      const quickConfig = {
        name: "Quick Training Session",
        description: "Fast bio-inspired MARL training with default parameters",
        total_episodes: 50,
        max_steps_per_episode: 100,
        learning_rate: 0.01,
        batch_size: 16,
        hidden_dim: 128,
        breakthrough_threshold: 0.6,
      };
      const response = await apiRequest('POST', '/api/training/start', {
        config: quickConfig,
      });
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/experiments'] });
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
      setSelectedExperiment(data.experimentId);
      toast({
        title: "Quick Training Started",
        description: "Fast bio-inspired MARL training session started with optimized parameters.",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: `Failed to start quick training: ${error.message}`,
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
      queryClient.invalidateQueries({ queryKey: ['/api/training/status'] });
      toast({
        title: "Training Stopped",
        description: "Bio-inspired MARL training has been stopped successfully.",
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
    startTrainingMutation.mutate(newExperimentConfig);
  };
  
  const handleQuickStart = () => {
    quickStartMutation.mutate();
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
        {/* Ray Status Alert */}
        {rayConfigTemplateMutation.data && !rayConfigTemplateMutation.data.ray_available && (
          <Alert className="mb-6" variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Ray RLlib Not Available</AlertTitle>
            <AlertDescription>
              {rayConfigTemplateMutation.data.ray_error?.includes('PyArrow') ? (
                <>
                  <strong>PyArrow Compatibility Issue:</strong> There's a version conflict between Ray and PyArrow. 
                  To fix this, update PyArrow to a compatible version or use the fallback training system.
                  <br />
                  <code className="text-xs mt-2 block bg-muted p-2 rounded">
                    pip install pyarrow==12.0.0  # or compatible version
                  </code>
                </>
              ) : (
                <>
                  <strong>Ray RLlib dependency issue:</strong> {rayConfigTemplateMutation.data.ray_error}
                  <br />
                  <span className="text-sm">The system will automatically use the fallback training method.</span>
                </>
              )}
            </AlertDescription>
          </Alert>
        )}
        
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
              <Tabs defaultValue="standard" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="standard">Standard Training</TabsTrigger>
                  <TabsTrigger value="ray">Ray RLlib Training</TabsTrigger>
                </TabsList>
                
                <TabsContent value="standard" className="space-y-4">
                  <TrainingControls
                    experimentId={selectedExperiment}
                    trainingStatus={trainingStatus}
                    onStart={handleStartTraining}
                    onQuickStart={handleQuickStart}
                    onStop={handleStopTraining}
                    isStarting={startTrainingMutation.isPending}
                    isStopping={stopTrainingMutation.isPending}
                    realtimeMetrics={wsData}
                  />
                </TabsContent>
                
                <TabsContent value="ray" className="space-y-4">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold">Ray RLlib Training</h3>
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => rayConfigTemplateMutation.mutate()}
                          disabled={rayConfigTemplateMutation.isPending}
                        >
                          <FileText className="h-4 w-4 mr-2" />
                          Load Template
                        </Button>
                        <Button 
                          onClick={() => startRayTrainingMutation.mutate(rayConfig)}
                          disabled={startRayTrainingMutation.isPending}
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Start Ray Training
                        </Button>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="ray_name">Experiment Name</Label>
                        <Input
                          id="ray_name"
                          value={rayConfig.name}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, name: e.target.value }))}
                          placeholder="Ray RLlib Experiment"
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_episodes">Episodes</Label>
                        <Input
                          id="ray_episodes"
                          type="number"
                          value={rayConfig.total_episodes}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, total_episodes: parseInt(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_lr">Learning Rate</Label>
                        <Input
                          id="ray_lr"
                          type="number"
                          step="0.0001"
                          value={rayConfig.learning_rate}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_workers">Rollout Workers</Label>
                        <Input
                          id="ray_workers"
                          type="number"
                          value={rayConfig.num_rollout_workers}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, num_rollout_workers: parseInt(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_batch">Batch Size</Label>
                        <Input
                          id="ray_batch"
                          type="number"
                          value={rayConfig.batch_size}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_train_batch">Train Batch Size</Label>
                        <Input
                          id="ray_train_batch"
                          type="number"
                          value={rayConfig.train_batch_size}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, train_batch_size: parseInt(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_hidden">Hidden Dimension</Label>
                        <Input
                          id="ray_hidden"
                          type="number"
                          value={rayConfig.hidden_dim}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, hidden_dim: parseInt(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_attention">Attention Heads</Label>
                        <Input
                          id="ray_attention"
                          type="number"
                          value={rayConfig.num_attention_heads}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, num_attention_heads: parseInt(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_pheromone">Pheromone Decay</Label>
                        <Input
                          id="ray_pheromone"
                          type="number"
                          step="0.01"
                          value={rayConfig.pheromone_decay}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, pheromone_decay: parseFloat(e.target.value) }))}
                        />
                      </div>
                      
                      <div>
                        <Label htmlFor="ray_plasticity">Neural Plasticity Rate</Label>
                        <Input
                          id="ray_plasticity"
                          type="number"
                          step="0.01"
                          value={rayConfig.neural_plasticity_rate}
                          onChange={(e) => setRayConfig(prev => ({ ...prev, neural_plasticity_rate: parseFloat(e.target.value) }))}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <Label htmlFor="ray_description">Description</Label>
                      <Textarea
                        id="ray_description"
                        value={rayConfig.description}
                        onChange={(e) => setRayConfig(prev => ({ ...prev, description: e.target.value }))}
                        placeholder="Ray RLlib training with bio-inspired features"
                      />
                    </div>
                    
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">Ray RLlib Features</h4>
                      <ul className="text-sm text-blue-800 dark:text-blue-200 mt-2 space-y-1">
                        <li>• Distributed multi-worker training</li>
                        <li>• Production-ready Algorithm and Learner classes</li>
                        <li>• Bio-inspired neural networks with attention mechanisms</li>
                        <li>• Automatic checkpointing and evaluation</li>
                        <li>• Enhanced 3D environment with pheromone trails</li>
                        <li>• Real-time metrics and breakthrough detection</li>
                      </ul>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
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
