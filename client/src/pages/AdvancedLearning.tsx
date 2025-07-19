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
  initialized: boolean;
  learning_mode: string;
  is_training: boolean;
  current_phase: string;
  enabled_features: {
    curriculum_learning: boolean;
    transfer_learning: boolean;
    meta_learning: boolean;
  };
  curriculum_status?: any;
  transfer_status?: any;
  meta_status?: any;
}

interface CurriculumProgress {
  enabled: boolean;
  current_stage?: {
    name: string;
    index: number;
    total_stages: number;
    difficulty: number;
    episodes_completed: number;
    episodes_target: number;
  };
  performance?: {
    mastery_score: number;
    success_threshold: number;
    ready_to_advance: boolean;
  };
}

interface TransferRecommendations {
  enabled: boolean;
  recommendations: Array<{
    source_experiment: string;
    compatibility_score: number;
    recommended_components: string[];
    expected_benefit: {
      training_speedup: number;
      final_performance_boost: number;
      convergence_episodes: number;
    };
  }>;
  available_models: number;
}

interface MetaInsights {
  enabled: boolean;
  insights?: {
    total_adaptations: number;
    average_adaptation_time: number;
    average_performance_improvement: number;
    adaptation_success_rate: number;
  };
  recommendations?: string[];
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
    enabled: status?.enabled_features?.curriculum_learning,
    refetchInterval: 3000
  });

  // Fetch transfer recommendations
  const { data: transferRecommendations } = useQuery<TransferRecommendations>({
    queryKey: ['/api/advanced/transfer_recommendations'],
    enabled: status?.enabled_features?.transfer_learning
  });

  // Fetch meta-learning insights
  const { data: metaInsights } = useQuery<MetaInsights>({
    queryKey: ['/api/advanced/meta_insights'],
    enabled: status?.enabled_features?.meta_learning,
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
          {!status?.initialized ? (
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
              disabled={startTrainingMutation.isPending || status?.is_training}
              className="flex items-center gap-2"
            >
              {status?.is_training ? (
                <>
                  <Pause className="h-4 w-4" />
                  Training Active
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  {startTrainingMutation.isPending ? 'Starting...' : 'Start Advanced Training'}
                </>
              )}
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
            <div className="text-2xl font-bold">{status?.learning_mode || 'Standard'}</div>
            <p className="text-xs text-muted-foreground">
              {status?.is_training ? `Phase: ${status.current_phase}` : 'Ready to train'}
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
              {status?.enabled_features?.curriculum_learning ? (
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
                Stage: {curriculumProgress.current_stage.name}
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
              {status?.enabled_features?.transfer_learning ? (
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
                {transferRecommendations.available_models} models available
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
              {status?.enabled_features?.meta_learning ? (
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
            {metaInsights?.insights && (
              <p className="text-xs text-muted-foreground mt-1">
                {metaInsights.insights.total_adaptations} adaptations
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
              {curriculumProgress?.enabled ? (
                <div className="space-y-4">
                  {curriculumProgress.current_stage && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-medium">
                          Current Stage: {curriculumProgress.current_stage.name}
                        </h4>
                        <Badge variant="outline">
                          Stage {curriculumProgress.current_stage.index + 1} of {curriculumProgress.current_stage.total_stages}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Episodes Progress</span>
                          <span>
                            {curriculumProgress.current_stage.episodes_completed} / {curriculumProgress.current_stage.episodes_target}
                          </span>
                        </div>
                        <Progress 
                          value={(curriculumProgress.current_stage.episodes_completed / curriculumProgress.current_stage.episodes_target) * 100}
                          className="w-full"
                        />
                      </div>

                      {curriculumProgress.performance && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span>Mastery Score</span>
                            <span>{curriculumProgress.performance.mastery_score.toFixed(3)}</span>
                          </div>
                          <Progress 
                            value={curriculumProgress.performance.mastery_score * 100}
                            className="w-full"
                          />
                          <div className="text-xs text-muted-foreground">
                            Threshold: {curriculumProgress.performance.success_threshold.toFixed(2)}
                          </div>
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-4 mt-4">
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">{curriculumProgress.current_stage.difficulty.toFixed(1)}</div>
                          <div className="text-xs text-muted-foreground">Difficulty Level</div>
                        </div>
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {curriculumProgress.performance?.ready_to_advance ? '✓' : '⏳'}
                          </div>
                          <div className="text-xs text-muted-foreground">Ready to Advance</div>
                        </div>
                      </div>
                    </div>
                  )}
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
              {transferRecommendations?.enabled ? (
                <div className="space-y-4">
                  {transferRecommendations.recommendations.length > 0 ? (
                    <div className="space-y-3">
                      {transferRecommendations.recommendations.map((rec, index) => (
                        <Card key={index} className="p-4">
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-medium">{rec.source_experiment}</h4>
                            <Badge variant="secondary">
                              {(rec.compatibility_score * 100).toFixed(0)}% compatible
                            </Badge>
                          </div>
                          
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <div className="text-muted-foreground">Training Speedup</div>
                              <div className="font-medium">{rec.expected_benefit.training_speedup.toFixed(1)}x</div>
                            </div>
                            <div>
                              <div className="text-muted-foreground">Performance Boost</div>
                              <div className="font-medium">+{(rec.expected_benefit.final_performance_boost * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                              <div className="text-muted-foreground">Convergence</div>
                              <div className="font-medium">{rec.expected_benefit.convergence_episodes} episodes</div>
                            </div>
                          </div>
                          
                          <div className="mt-2">
                            <div className="text-sm text-muted-foreground">Components:</div>
                            <div className="flex gap-1 mt-1">
                              {rec.recommended_components.map((comp, i) => (
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
                      <AlertTitle>No Transfer Recommendations</AlertTitle>
                      <AlertDescription>
                        No compatible models found for transfer learning. Train some models first.
                      </AlertDescription>
                    </Alert>
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
              {metaInsights?.enabled ? (
                <div className="space-y-4">
                  {metaInsights.insights ? (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">{metaInsights.insights.total_adaptations}</div>
                          <div className="text-xs text-muted-foreground">Total Adaptations</div>
                        </div>
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {metaInsights.insights.average_adaptation_time.toFixed(1)}s
                          </div>
                          <div className="text-xs text-muted-foreground">Avg Adaptation Time</div>
                        </div>
                      </div>
                      
                      <div className="space-y-3">
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {(metaInsights.insights.average_performance_improvement * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-muted-foreground">Avg Performance Gain</div>
                        </div>
                        <div className="text-center p-3 bg-muted rounded-lg">
                          <div className="text-2xl font-bold">
                            {(metaInsights.insights.adaptation_success_rate * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-muted-foreground">Success Rate</div>
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

                  {metaInsights.recommendations && metaInsights.recommendations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium">Recommendations</h4>
                      <ScrollArea className="h-32">
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