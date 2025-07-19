import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { toast } from '@/hooks/use-toast';
import { 
  BookOpen, 
  Repeat2, 
  Brain, 
  TrendingUp, 
  Settings, 
  Play, 
  Pause,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';
import { apiRequest } from '@/lib/queryClient';

interface AdvancedLearningStatus {
  success: boolean;
  status: {
    curriculum: {
      enabled: boolean;
      stage: string;
      progress: number;
    };
    transfer: {
      enabled: boolean;
      models_available: number;
      compatibility: number;
    };
    meta: {
      enabled: boolean;
      adaptation_rate: number;
      tasks_completed: number;
    };
  };
  metrics: {
    curriculum_progress: number;
    transfer_efficiency: number;
    meta_adaptation_speed: number;
    overall_performance: number;
  };
  timestamp: number;
}

interface CurriculumProgress {
  success: boolean;
  current_stage: string;
  stage_index: number;
  total_stages: number;
  progress: number;
  metrics: {
    completion_rate: number;
    success_threshold: number;
    adaptive_difficulty: boolean;
  };
}

interface TransferRecommendations {
  success: boolean;
  available_models: Array<{
    name: string;
    compatibility: number;
    performance_gain: number;
    transfer_components: string[];
  }>;
  recommendations: string[];
}

interface MetaInsights {
  success: boolean;
  adaptation_metrics: {
    learning_rate: number;
    few_shot_performance: number;
    generalization_score: number;
    meta_optimization_steps: number;
  };
  insights: string[];
  recommendations: string[];
}

export default function AdvancedLearning() {
  const [isInitialized, setIsInitialized] = useState(false);
  const queryClient = useQueryClient();

  // Fetch advanced learning status
  const { data: status, isLoading: statusLoading } = useQuery<AdvancedLearningStatus>({
    queryKey: ['/api/advanced/status'],
    refetchInterval: 5000
  });

  // Fetch curriculum progress
  const { data: curriculumProgress } = useQuery<CurriculumProgress>({
    queryKey: ['/api/advanced/curriculum_progress'],
    enabled: status?.status?.curriculum?.enabled,
    refetchInterval: 3000
  });

  // Fetch transfer recommendations
  const { data: transferRecommendations } = useQuery<TransferRecommendations>({
    queryKey: ['/api/advanced/transfer_recommendations'],
    enabled: status?.status?.transfer?.enabled
  });

  // Fetch meta-learning insights
  const { data: metaInsights } = useQuery<MetaInsights>({
    queryKey: ['/api/advanced/meta_insights'],
    enabled: status?.status?.meta?.enabled,
    refetchInterval: 5000
  });

  // Initialize advanced learning
  const initializeMutation = useMutation({
    mutationFn: (config: any) => apiRequest('/api/advanced/initialize', 'POST', config),
    onSuccess: () => {
      setIsInitialized(true);
      queryClient.invalidateQueries({ queryKey: ['/api/advanced/status'] });
      toast({
        title: "Advanced Learning Initialized",
        description: "All advanced learning features have been activated."
      });
    },
    onError: (error: any) => {
      toast({
        title: "Initialization Failed",
        description: error.message || "Failed to initialize advanced learning features.",
        variant: "destructive"
      });
    }
  });

  // Start advanced training
  const startTrainingMutation = useMutation({
    mutationFn: (config: any) => apiRequest('/api/advanced/start_training', 'POST', config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/advanced/status'] });
      toast({
        title: "Advanced Training Started",
        description: "Training session has begun with advanced learning features."
      });
    },
    onError: (error: any) => {
      toast({
        title: "Training Failed",
        description: error.message || "Failed to start advanced training.",
        variant: "destructive"
      });
    }
  });

  const handleInitialize = () => {
    const config = {
      advanced: {
        curriculum_learning: {
          enabled: true,
          stages: [
            {
              name: "basic_coordination",
              episodes: 200,
              difficulty: 0.3,
              success_threshold: 0.75
            },
            {
              name: "advanced_coordination", 
              episodes: 500,
              difficulty: 0.7,
              success_threshold: 0.85
            },
            {
              name: "expert_coordination",
              episodes: 300,
              difficulty: 1.0,
              success_threshold: 0.9
            }
          ]
        },
        transfer_learning: {
          enabled: true,
          transfer_components: ["attention", "memory", "communication"]
        },
        meta_learning: {
          enabled: true,
          adaptation_steps: 5,
          meta_lr: 0.01,
          inner_lr: 0.1
        }
      }
    };

    initializeMutation.mutate(config);
  };

  const handleStartTraining = () => {
    const trainingConfig = {
      experiment_name: "Advanced Learning Session",
      total_episodes: 1000,
      max_steps_per_episode: 500,
      learning_rate: 0.001,
      use_curriculum: true,
      use_transfer: true,
      use_meta_learning: true
    };

    startTrainingMutation.mutate(trainingConfig);
  };

  if (statusLoading) {
    return (
      <div className="container mx-auto p-4">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
            <p>Loading advanced learning status...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Advanced Learning Features</h1>
          <p className="text-muted-foreground">
            Curriculum Learning, Transfer Learning, and Meta-Learning for enhanced agent training
          </p>
        </div>
        <div className="flex gap-2">
          {!status?.success ? (
            <Button 
              onClick={handleInitialize}
              disabled={initializeMutation.isPending}
              className="flex items-center gap-2"
            >
              <Settings className="h-4 w-4" />
              {initializeMutation.isPending ? 'Initializing...' : 'Initialize'}
            </Button>
          ) : (
            <Button 
              onClick={handleStartTraining}
              disabled={startTrainingMutation.isPending}
              className="flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              {startTrainingMutation.isPending ? 'Starting...' : 'Start Advanced Training'}
            </Button>
          )}
        </div>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Learning Mode</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Advanced</div>
            <p className="text-xs text-muted-foreground">
              Stage: {status?.status?.curriculum?.stage || 'Ready to train'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Curriculum Learning</CardTitle>
            <BookOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {status?.status?.curriculum?.enabled ? (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium">Enabled</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm font-medium">Disabled</span>
                </>
              )}
            </div>
            {curriculumProgress?.current_stage && (
              <p className="text-xs text-muted-foreground mt-1">
                Stage: {curriculumProgress.current_stage}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Transfer Learning</CardTitle>
            <Repeat2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {status?.status?.transfer?.enabled ? (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium">Enabled</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm font-medium">Disabled</span>
                </>
              )}
            </div>
            {transferRecommendations && (
              <p className="text-xs text-muted-foreground mt-1">
                {transferRecommendations.available_models.length} models available
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Meta-Learning</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {status?.status?.meta?.enabled ? (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium">Enabled</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm font-medium">Disabled</span>
                </>
              )}
            </div>
            {metaInsights?.adaptation_metrics && (
              <p className="text-xs text-muted-foreground mt-1">
                {metaInsights.adaptation_metrics.meta_optimization_steps} steps
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="curriculum" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="curriculum" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            Curriculum Learning
          </TabsTrigger>
          <TabsTrigger value="transfer" className="flex items-center gap-2">
            <Repeat2 className="h-4 w-4" />
            Transfer Learning
          </TabsTrigger>
          <TabsTrigger value="meta" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Meta-Learning
          </TabsTrigger>
        </TabsList>

        {/* Curriculum Learning Tab */}
        <TabsContent value="curriculum" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Curriculum Learning Progress</CardTitle>
              <CardDescription>
                Progressive difficulty training for enhanced learning efficiency
              </CardDescription>
            </CardHeader>
            <CardContent>
              {curriculumProgress?.success ? (
                <div className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-medium">
                        Current Stage: {curriculumProgress.current_stage}
                      </h4>
                      <Badge variant="outline">
                        Stage {curriculumProgress.stage_index + 1} of {curriculumProgress.total_stages}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Stage Progress</span>
                        <span>
                          {(curriculumProgress.progress * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress 
                        value={curriculumProgress.progress * 100}
                        className="w-full"
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Completion Rate</span>
                        <span>{(curriculumProgress.metrics.completion_rate * 100).toFixed(1)}%</span>
                      </div>
                      <Progress 
                        value={curriculumProgress.metrics.completion_rate * 100}
                        className="w-full"
                      />
                      <div className="text-xs text-muted-foreground">
                        Threshold: {curriculumProgress.metrics.success_threshold.toFixed(2)}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div className="text-center p-3 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">{curriculumProgress.stage_index + 1}</div>
                        <div className="text-xs text-muted-foreground">Current Stage</div>
                      </div>
                      <div className="text-center p-3 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">
                          {curriculumProgress.metrics.adaptive_difficulty ? '✓' : '✗'}
                        </div>
                        <div className="text-xs text-muted-foreground">Adaptive Difficulty</div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>Curriculum Learning Disabled</AlertTitle>
                  <AlertDescription>
                    Initialize advanced learning to enable curriculum-based training.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Transfer Learning Tab */}
        <TabsContent value="transfer" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Transfer Learning Recommendations</CardTitle>
              <CardDescription>
                Leverage knowledge from previous experiments to accelerate learning
              </CardDescription>
            </CardHeader>
            <CardContent>
              {transferRecommendations?.success ? (
                <div className="space-y-4">
                  {transferRecommendations.available_models.length > 0 ? (
                    <div className="space-y-3">
                      {transferRecommendations.available_models.map((model, index) => (
                        <Card key={index} className="p-4">
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-medium">{model.name}</h4>
                            <Badge variant="secondary">
                              {(model.compatibility * 100).toFixed(0)}% compatible
                            </Badge>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <div className="text-muted-foreground">Performance Gain</div>
                              <div className="font-medium">+{(model.performance_gain * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                              <div className="text-muted-foreground">Compatibility</div>
                              <div className="font-medium">{(model.compatibility * 100).toFixed(1)}%</div>
                            </div>
                          </div>
                          
                          <div className="mt-2">
                            <div className="text-sm text-muted-foreground">Components:</div>
                            <div className="flex gap-1 mt-1">
                              {model.transfer_components.map((comp, i) => (
                                <Badge key={i} variant="outline" className="text-xs">
                                  {comp}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertTitle>No Transfer Models</AlertTitle>
                      <AlertDescription>
                        No compatible models found for transfer learning. Train some models first.
                      </AlertDescription>
                    </Alert>
                  )}
                  
                  {transferRecommendations.recommendations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium">Recommendations</h4>
                      <div className="space-y-2">
                        {transferRecommendations.recommendations.map((rec, index) => (
                          <Alert key={index}>
                            <Info className="h-4 w-4" />
                            <AlertDescription>{rec}</AlertDescription>
                          </Alert>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>Transfer Learning Disabled</AlertTitle>
                  <AlertDescription>
                    Initialize advanced learning to enable transfer learning capabilities.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Meta-Learning Tab */}
        <TabsContent value="meta" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Meta-Learning Insights</CardTitle>
              <CardDescription>
                Learning to learn - adaptive algorithms that improve from experience
              </CardDescription>
            </CardHeader>
            <CardContent>
              {metaInsights?.success ? (
                <div className="space-y-4">
                  {metaInsights.adaptation_metrics ? (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">{metaInsights.adaptation_metrics.meta_optimization_steps}</div>
                          <div className="text-xs text-muted-foreground">Optimization Steps</div>
                        </div>
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {metaInsights.adaptation_metrics.learning_rate.toFixed(2)}
                          </div>
                          <div className="text-xs text-muted-foreground">Learning Rate</div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {(metaInsights.adaptation_metrics.few_shot_performance * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-muted-foreground">Few-Shot Performance</div>
                        </div>
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {(metaInsights.adaptation_metrics.generalization_score * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-muted-foreground">Generalization Score</div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertTitle>No Meta-Learning Data</AlertTitle>
                      <AlertDescription>
                        Start training to collect meta-learning insights and adaptations.
                      </AlertDescription>
                    </Alert>
                  )}

                  {metaInsights.insights && metaInsights.insights.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium">Insights</h4>
                      <ScrollArea className="h-24">
                        <div className="space-y-2">
                          {metaInsights.insights.map((insight, index) => (
                            <Alert key={index}>
                              <Info className="h-4 w-4" />
                              <AlertDescription>{insight}</AlertDescription>
                            </Alert>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}
                  
                  {metaInsights.recommendations && metaInsights.recommendations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium">Recommendations</h4>
                      <ScrollArea className="h-24">
                        <div className="space-y-2">
                          {metaInsights.recommendations.map((rec, index) => (
                            <Alert key={index}>
                              <Info className="h-4 w-4" />
                              <AlertDescription>{rec}</AlertDescription>
                            </Alert>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}
                </div>
              ) : (
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>Meta-Learning Disabled</AlertTitle>
                  <AlertDescription>
                    Initialize advanced learning to enable meta-learning capabilities.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}