import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Play, RefreshCw, Zap, Settings } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";

interface InitializationResponse {
  success: boolean;
  message: string;
  data: {
    totalAgents: number;
    coordinatorCount: number;
    experimentId: number;
  };
}

export function SystemControl() {
  const [isInitialized, setIsInitialized] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const initializeSystemMutation = useMutation({
    mutationFn: async (): Promise<InitializationResponse> => {
      const response = await apiRequest('POST', '/api/init-agents', {});
      return response.json();
    },
    onSuccess: (data) => {
      setIsInitialized(true);
      toast({
        title: "System Initialized",
        description: `Successfully created ${data.data.totalAgents} agents with ${data.data.coordinatorCount} coordinators`,
      });
      
      // Invalidate all queries to refresh the UI
      queryClient.invalidateQueries();
    },
    onError: (error: any) => {
      console.error('Initialization error:', error);
      toast({
        title: "Initialization Failed",
        description: "Failed to initialize the agent system. Please check the console for details.",
        variant: "destructive",
      });
    },
  });

  const startTrainingMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/train/start', {});
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Training Started",
        description: "Multi-agent training session has been initiated",
      });
      queryClient.invalidateQueries();
    },
    onError: () => {
      toast({
        title: "Training Failed",
        description: "Failed to start training session",
        variant: "destructive",
      });
    },
  });

  const simulateCommunicationMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest('POST', '/api/simulate-communication', {});
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Communication Simulated",
        description: `Generated ${data.data.messages} messages across ${data.data.rounds} rounds with ${(data.data.efficiency * 100).toFixed(1)}% efficiency`,
      });
      queryClient.invalidateQueries();
    },
    onError: () => {
      toast({
        title: "Communication Failed",
        description: "Failed to simulate communication patterns",
        variant: "destructive",
      });
    },
  });

  const handleInitialize = () => {
    initializeSystemMutation.mutate();
  };

  const handleStartTraining = () => {
    startTrainingMutation.mutate();
  };

  const handleSimulateCommunication = () => {
    simulateCommunicationMutation.mutate();
  };

  const handleReset = () => {
    setIsInitialized(false);
    queryClient.invalidateQueries();
    toast({
      title: "System Reset",
      description: "The system has been reset to initial state",
    });
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="w-5 h-5" />
          System Control
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <Badge variant={isInitialized ? "default" : "secondary"}>
            {isInitialized ? "Initialized" : "Not Initialized"}
          </Badge>
          <Badge variant="outline">
            Brain-Inspired MARL v1.0
          </Badge>
        </div>

        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">
            Phase 1: Agent Initialization & Grid Population
          </div>
          <Button 
            onClick={handleInitialize}
            disabled={initializeSystemMutation.isPending}
            className="w-full"
            variant={isInitialized ? "outline" : "default"}
          >
            {initializeSystemMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Initializing Agents...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                {isInitialized ? "Reinitialize" : "Initialize"} Agent System
              </>
            )}
          </Button>
        </div>

        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">
            Phase 2: Communication Simulation
          </div>
          <Button 
            onClick={handleSimulateCommunication}
            disabled={!isInitialized || simulateCommunicationMutation.isPending}
            className="w-full"
            variant="default"
          >
            {simulateCommunicationMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Simulating Communication...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Simulate Communication
              </>
            )}
          </Button>
        </div>

        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">
            Phase 3: Start Training Experiment
          </div>
          <Button 
            onClick={handleStartTraining}
            disabled={!isInitialized || startTrainingMutation.isPending}
            className="w-full"
            variant="default"
          >
            {startTrainingMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Starting Training...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Start MARL Training
              </>
            )}
          </Button>
        </div>

        <div className="pt-4 border-t">
          <div className="text-sm text-muted-foreground mb-2">
            System Configuration
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Grid Size: 4×3×3</div>
            <div>Total Agents: 30</div>
            <div>Coordinators: 3</div>
            <div>Regular Agents: 27</div>
          </div>
        </div>

        <div className="pt-2">
          <Button 
            onClick={handleReset}
            variant="outline"
            size="sm"
            className="w-full"
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Reset System
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}