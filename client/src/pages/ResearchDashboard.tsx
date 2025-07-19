import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { AlertCircle, CheckCircle, Clock, Play, Pause, BarChart3, Brain, Beaker } from 'lucide-react';
import { toast } from '@/hooks/use-toast';

interface ResearchStatus {
  status: string;
  framework_initialized: boolean;
  active_experiments: number;
  total_hypotheses: number;
  hypotheses: Record<string, {
    title: string;
    status: string;
    confidence_threshold: number;
  }>;
  available_experiment_types: string[];
}

interface Experiment {
  experiment_id: string;
  experiment_name: string;
  hypothesis_id: string;
  status: string;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  duration?: number;
}

interface ExperimentConfig {
  experiment_name: string;
  hypothesis_id: string;
  description: string;
  environment_type: string;
  num_agents: number;
  grid_size: [number, number, number];
  hidden_dim: number;
  attention_heads: number;
  training_steps: number;
  learning_rate: number;
  batch_size: number;
}

export default function ResearchDashboard() {
  const [selectedTab, setSelectedTab] = useState('overview');
  const [newExperimentConfig, setNewExperimentConfig] = useState<Partial<ExperimentConfig>>({
    experiment_name: '',
    hypothesis_id: 'H1_pheromone_emergence',
    description: '',
    environment_type: 'ForagingEnvironment',
    num_agents: 8,
    grid_size: [6, 6, 1],
    hidden_dim: 256,
    attention_heads: 8,
    training_steps: 1000,
    learning_rate: 0.0003,
    batch_size: 128
  });

  const queryClient = useQueryClient();

  // Fetch research framework status
  const { data: researchStatus, isLoading: statusLoading } = useQuery<ResearchStatus>({
    queryKey: ['/api/research/status'],
    refetchInterval: 5000
  });

  // Fetch experiments list
  const { data: experiments, isLoading: experimentsLoading } = useQuery<{
    experiments: Experiment[];
    total_count: number;
    status_summary: Record<string, number>;
  }>({
    queryKey: ['/api/research/experiments'],
    refetchInterval: 3000
  });

  // Fetch hypotheses summary
  const { data: hypotheses } = useQuery({
    queryKey: ['/api/research/hypotheses'],
    refetchInterval: 10000
  });

  // Create experiment mutation
  const createExperimentMutation = useMutation({
    mutationFn: async (config: ExperimentConfig) => {
      const response = await fetch('/api/research/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      return response.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        toast({
          title: "Experiment Created",
          description: `Experiment "${data.experiment_name}" created successfully`
        });
        queryClient.invalidateQueries({ queryKey: ['/api/research/experiments'] });
        setNewExperimentConfig({
          experiment_name: '',
          hypothesis_id: 'H1_pheromone_emergence',
          description: '',
          environment_type: 'ForagingEnvironment',
          num_agents: 8,
          grid_size: [6, 6, 1],
          hidden_dim: 256,
          attention_heads: 8,
          training_steps: 1000,
          learning_rate: 0.0003,
          batch_size: 128
        });
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to create experiment",
          variant: "destructive"
        });
      }
    }
  });

  // Run experiment mutation
  const runExperimentMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      const response = await fetch(`/api/research/experiments/${experimentId}/run`, {
        method: 'POST'
      });
      return response.json();
    },
    onSuccess: (data) => {
      if (data.success) {
        toast({
          title: "Experiment Started",
          description: "Research experiment is now running"
        });
        queryClient.invalidateQueries({ queryKey: ['/api/research/experiments'] });
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to start experiment",
          variant: "destructive"
        });
      }
    }
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'running':
        return <Clock className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'running':
        return 'secondary';
      case 'failed':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-blue-500" />
            Research Dashboard
          </h1>
          <p className="text-muted-foreground">
            Systematic multi-agent reinforcement learning research framework
          </p>
        </div>
        {researchStatus && (
          <Badge variant={researchStatus.framework_initialized ? 'default' : 'destructive'}>
            {researchStatus.framework_initialized ? 'Active' : 'Inactive'}
          </Badge>
        )}
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="experiments">Experiments</TabsTrigger>
          <TabsTrigger value="hypotheses">Hypotheses</TabsTrigger>
          <TabsTrigger value="create">Create</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Experiments</CardTitle>
                <Beaker className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {researchStatus?.active_experiments || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Currently running research experiments
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Hypotheses</CardTitle>
                <Brain className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {researchStatus?.total_hypotheses || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Research hypotheses being tested
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Completion Rate</CardTitle>
                <BarChart3 className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {experiments?.status_summary?.completed || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  Successfully completed experiments
                </p>
              </CardContent>
            </Card>
          </div>

          {researchStatus?.hypotheses && (
            <Card>
              <CardHeader>
                <CardTitle>Research Hypotheses Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(researchStatus.hypotheses).map(([id, hypothesis]) => (
                    <div key={id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex-1">
                        <h3 className="font-medium">{hypothesis.title}</h3>
                        <p className="text-sm text-muted-foreground">
                          Confidence threshold: {(hypothesis.confidence_threshold * 100).toFixed(0)}%
                        </p>
                      </div>
                      <Badge variant={getStatusBadgeVariant(hypothesis.status)}>
                        {hypothesis.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="experiments" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Research Experiments</CardTitle>
              <p className="text-sm text-muted-foreground">
                Track and manage your research experiments
              </p>
            </CardHeader>
            <CardContent>
              {experimentsLoading ? (
                <div className="text-center py-8">Loading experiments...</div>
              ) : experiments?.experiments.length ? (
                <ScrollArea className="h-96">
                  <div className="space-y-4">
                    {experiments.experiments.map((experiment) => (
                      <div key={experiment.experiment_id} className="flex items-center justify-between p-4 border rounded-lg">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(experiment.status)}
                            <h3 className="font-medium">{experiment.experiment_name}</h3>
                            <Badge variant="outline">{experiment.hypothesis_id}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">
                            Created: {new Date(experiment.created_at * 1000).toLocaleString()}
                            {experiment.duration && ` • Duration: ${formatDuration(experiment.duration)}`}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={getStatusBadgeVariant(experiment.status)}>
                            {experiment.status}
                          </Badge>
                          {experiment.status === 'created' && (
                            <Button
                              size="sm"
                              onClick={() => runExperimentMutation.mutate(experiment.experiment_id)}
                              disabled={runExperimentMutation.isPending}
                            >
                              <Play className="w-4 h-4 mr-1" />
                              Run
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No experiments found. Create your first research experiment.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hypotheses" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Hypothesis Validation</CardTitle>
              <p className="text-sm text-muted-foreground">
                Track the validation status of research hypotheses
              </p>
            </CardHeader>
            <CardContent>
              {hypotheses ? (
                <div className="space-y-4">
                  {Object.entries(hypotheses.hypothesis_summary || {}).map(([id, data]: [string, any]) => (
                    <div key={id} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-medium">{data.hypothesis?.title}</h3>
                        <Badge variant={data.overall_validated ? 'default' : 'secondary'}>
                          {data.overall_validated ? 'Validated' : 'Testing'}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">
                        {data.hypothesis?.description}
                      </p>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Experiments: </span>
                          {data.experiment_count || 0}
                        </div>
                        <div>
                          <span className="font-medium">Confidence: </span>
                          {data.overall_confidence ? (data.overall_confidence * 100).toFixed(1) + '%' : 'N/A'}
                        </div>
                      </div>
                      {data.overall_confidence && (
                        <Progress value={data.overall_confidence * 100} className="mt-3" />
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  Loading hypothesis validation data...
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create New Experiment</CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure and launch a new research experiment
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="experiment_name">Experiment Name</Label>
                  <Input
                    id="experiment_name"
                    value={newExperimentConfig.experiment_name || ''}
                    onChange={(e) => setNewExperimentConfig(prev => ({
                      ...prev,
                      experiment_name: e.target.value
                    }))}
                    placeholder="Enter experiment name"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="hypothesis_id">Research Hypothesis</Label>
                  <Select
                    value={newExperimentConfig.hypothesis_id}
                    onValueChange={(value) => setNewExperimentConfig(prev => ({
                      ...prev,
                      hypothesis_id: value
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select hypothesis" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="H1_pheromone_emergence">Pheromone Communication</SelectItem>
                      <SelectItem value="H2_swarm_coordination">Swarm Coordination</SelectItem>
                      <SelectItem value="H3_environmental_pressure">Environmental Adaptation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  value={newExperimentConfig.description || ''}
                  onChange={(e) => setNewExperimentConfig(prev => ({
                    ...prev,
                    description: e.target.value
                  }))}
                  placeholder="Describe the experiment objectives and methodology"
                  rows={3}
                />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="num_agents">Number of Agents</Label>
                  <Input
                    id="num_agents"
                    type="number"
                    value={newExperimentConfig.num_agents || 8}
                    onChange={(e) => setNewExperimentConfig(prev => ({
                      ...prev,
                      num_agents: parseInt(e.target.value) || 8
                    }))}
                    min="2"
                    max="32"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="training_steps">Training Steps</Label>
                  <Input
                    id="training_steps"
                    type="number"
                    value={newExperimentConfig.training_steps || 1000}
                    onChange={(e) => setNewExperimentConfig(prev => ({
                      ...prev,
                      training_steps: parseInt(e.target.value) || 1000
                    }))}
                    min="100"
                    max="10000"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="hidden_dim">Hidden Dimension</Label>
                  <Input
                    id="hidden_dim"
                    type="number"
                    value={newExperimentConfig.hidden_dim || 256}
                    onChange={(e) => setNewExperimentConfig(prev => ({
                      ...prev,
                      hidden_dim: parseInt(e.target.value) || 256
                    }))}
                    min="64"
                    max="1024"
                  />
                </div>
              </div>

              <Separator />

              <Button
                onClick={() => createExperimentMutation.mutate(newExperimentConfig as ExperimentConfig)}
                disabled={createExperimentMutation.isPending || !newExperimentConfig.experiment_name}
                className="w-full"
              >
                {createExperimentMutation.isPending ? 'Creating...' : 'Create Experiment'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}