import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ConfigurationForm } from "../components/ConfigurationForm";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { TrainingConfig } from "@/types/training";
import { Settings, Brain, Network, Database, Save, Download, Upload } from "lucide-react";

const defaultConfig: TrainingConfig = {
  maxEpisodes: 1000,
  stepsPerEpisode: 200,
  learningRate: 0.001,
  batchSize: 32,
  replayBufferSize: 10000,
  explorationRate: 0.1,
  communicationRange: 2,
  breakthroughThreshold: 0.8,
  memorySize: 1000,
  coordinatorRatio: 0.1
};

export default function Config() {
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const [presetName, setPresetName] = useState<string>("");
  const [savedPresets, setSavedPresets] = useState<Record<string, TrainingConfig>>({});

  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: experiments } = useQuery({
    queryKey: ["/api/experiments"],
    staleTime: 10000,
  });

  const createExperimentMutation = useMutation({
    mutationFn: async (experimentData: any) => {
      const res = await apiRequest("POST", "/api/experiments", experimentData);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/experiments"] });
      toast({
        title: "Success",
        description: "Experiment created successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to create experiment",
        variant: "destructive",
      });
    },
  });

  const handleConfigChange = (newConfig: Partial<TrainingConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  };

  const handleCreateExperiment = async () => {
    if (!presetName.trim()) {
      toast({
        title: "Error",
        description: "Please provide an experiment name",
        variant: "destructive",
      });
      return;
    }

    const experimentData = {
      name: presetName,
      description: `Brain-inspired MARL experiment with ${config.maxEpisodes} episodes`,
      config: config,
      status: "created"
    };

    createExperimentMutation.mutate(experimentData);
  };

  const handleSavePreset = () => {
    if (!presetName.trim()) {
      toast({
        title: "Error",
        description: "Please provide a preset name",
        variant: "destructive",
      });
      return;
    }

    const newPresets = { ...savedPresets, [presetName]: config };
    setSavedPresets(newPresets);
    localStorage.setItem("brain-marl-presets", JSON.stringify(newPresets));
    
    toast({
      title: "Success",
      description: `Preset "${presetName}" saved successfully`,
    });
  };

  const handleLoadPreset = (name: string) => {
    const preset = savedPresets[name];
    if (preset) {
      setConfig(preset);
      setPresetName(name);
      toast({
        title: "Success",
        description: `Preset "${name}" loaded successfully`,
      });
    }
  };

  const handleExportConfig = () => {
    const exportData = {
      config,
      presets: savedPresets,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brain-marl-config-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importData = JSON.parse(e.target?.result as string);
        if (importData.config) {
          setConfig(importData.config);
        }
        if (importData.presets) {
          setSavedPresets(importData.presets);
          localStorage.setItem("brain-marl-presets", JSON.stringify(importData.presets));
        }
        toast({
          title: "Success",
          description: "Configuration imported successfully",
        });
      } catch (error) {
        toast({
          title: "Error",
          description: "Invalid configuration file",
          variant: "destructive",
        });
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  // Load saved presets on mount
  useState(() => {
    const saved = localStorage.getItem("brain-marl-presets");
    if (saved) {
      try {
        setSavedPresets(JSON.parse(saved));
      } catch (error) {
        console.error("Failed to load saved presets:", error);
      }
    }
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Configuration</h1>
          <p className="text-muted-foreground mt-2">
            Configure training parameters and experiment settings
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={handleExportConfig} variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <label className="cursor-pointer">
            <input
              type="file"
              accept=".json"
              onChange={handleImportConfig}
              className="hidden"
            />
            <Button variant="outline" size="sm" asChild>
              <span>
                <Upload className="h-4 w-4 mr-2" />
                Import
              </span>
            </Button>
          </label>
        </div>
      </div>

      {/* Experiment Creation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Create New Experiment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <input
              type="text"
              placeholder="Experiment name"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              className="flex-1 px-3 py-2 border rounded-md"
            />
            <Button
              onClick={handleCreateExperiment}
              disabled={createExperimentMutation.isPending}
            >
              {createExperimentMutation.isPending ? "Creating..." : "Create Experiment"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Configuration Tabs */}
      <Tabs defaultValue="training" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="training" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Training
          </TabsTrigger>
          <TabsTrigger value="network" className="flex items-center gap-2">
            <Network className="h-4 w-4" />
            Network
          </TabsTrigger>
          <TabsTrigger value="memory" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Memory
          </TabsTrigger>
          <TabsTrigger value="presets" className="flex items-center gap-2">
            <Save className="h-4 w-4" />
            Presets
          </TabsTrigger>
        </TabsList>

        <TabsContent value="training" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Training Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <ConfigurationForm
                config={config}
                onConfigChange={handleConfigChange}
                section="training"
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Network Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <ConfigurationForm
                config={config}
                onConfigChange={handleConfigChange}
                section="network"
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="memory" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Memory & Communication</CardTitle>
            </CardHeader>
            <CardContent>
              <ConfigurationForm
                config={config}
                onConfigChange={handleConfigChange}
                section="memory"
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="presets" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Save Current Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Preset name"
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                  />
                  <Button onClick={handleSavePreset} className="w-full">
                    <Save className="h-4 w-4 mr-2" />
                    Save Preset
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Saved Presets</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.keys(savedPresets).length > 0 ? (
                    Object.keys(savedPresets).map((name) => (
                      <div key={name} className="flex items-center justify-between p-2 bg-muted rounded">
                        <span className="font-medium">{name}</span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleLoadPreset(name)}
                        >
                          Load
                        </Button>
                      </div>
                    ))
                  ) : (
                    <p className="text-muted-foreground">No saved presets</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Current Configuration Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Current Configuration Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {config.maxEpisodes}
              </div>
              <div className="text-sm text-muted-foreground">Max Episodes</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {config.stepsPerEpisode}
              </div>
              <div className="text-sm text-muted-foreground">Steps/Episode</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {config.learningRate}
              </div>
              <div className="text-sm text-muted-foreground">Learning Rate</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {config.communicationRange}
              </div>
              <div className="text-sm text-muted-foreground">Comm Range</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
