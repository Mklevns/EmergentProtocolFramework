import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrainingConfig } from "@/types/training";

// Form schema based on TrainingConfig
const configSchema = z.object({
  maxEpisodes: z.number().min(100).max(10000),
  stepsPerEpisode: z.number().min(50).max(1000),
  learningRate: z.number().min(0.0001).max(0.1),
  batchSize: z.number().min(8).max(256),
  replayBufferSize: z.number().min(1000).max(100000),
  explorationRate: z.number().min(0.01).max(1.0),
  communicationRange: z.number().min(1).max(10),
  breakthroughThreshold: z.number().min(0.1).max(1.0),
  memorySize: z.number().min(100).max(10000),
  coordinatorRatio: z.number().min(0.05).max(0.5),
});

interface ConfigurationFormProps {
  config: TrainingConfig;
  onConfigChange: (config: Partial<TrainingConfig>) => void;
  section: "training" | "network" | "memory";
}

export function ConfigurationForm({ config, onConfigChange, section }: ConfigurationFormProps) {
  const form = useForm<TrainingConfig>({
    resolver: zodResolver(configSchema),
    defaultValues: config,
  });

  const handleFieldChange = (field: keyof TrainingConfig, value: number) => {
    form.setValue(field, value);
    onConfigChange({ [field]: value });
  };

  const renderTrainingSection = () => (
    <div className="space-y-6">
      <FormField
        control={form.control}
        name="maxEpisodes"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Max Episodes</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("maxEpisodes", parseInt(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value]}
                  onValueChange={(values) => handleFieldChange("maxEpisodes", values[0])}
                  min={100}
                  max={10000}
                  step={100}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Total number of training episodes (100-10000)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="stepsPerEpisode"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Steps Per Episode</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("stepsPerEpisode", parseInt(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value]}
                  onValueChange={(values) => handleFieldChange("stepsPerEpisode", values[0])}
                  min={50}
                  max={1000}
                  step={10}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Number of steps per training episode (50-1000)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="learningRate"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Learning Rate</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  step="0.0001"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("learningRate", parseFloat(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value * 10000]} // Scale for slider
                  onValueChange={(values) => handleFieldChange("learningRate", values[0] / 10000)}
                  min={1}
                  max={1000}
                  step={1}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Learning rate for the neural network (0.0001-0.1)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="batchSize"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Batch Size</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("batchSize", parseInt(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value]}
                  onValueChange={(values) => handleFieldChange("batchSize", values[0])}
                  min={8}
                  max={256}
                  step={8}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Training batch size (8-256)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="explorationRate"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Exploration Rate</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  step="0.01"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("explorationRate", parseFloat(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value * 100]} // Scale for slider
                  onValueChange={(values) => handleFieldChange("explorationRate", values[0] / 100)}
                  min={1}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Exploration rate for epsilon-greedy policy (0.01-1.0)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />
    </div>
  );

  const renderNetworkSection = () => (
    <div className="space-y-6">
      <FormField
        control={form.control}
        name="communicationRange"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Communication Range</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("communicationRange", parseInt(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value]}
                  onValueChange={(values) => handleFieldChange("communicationRange", values[0])}
                  min={1}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Agent communication range in grid units (1-10)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="coordinatorRatio"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Coordinator Ratio</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  step="0.01"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("coordinatorRatio", parseFloat(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value * 100]} // Scale for slider
                  onValueChange={(values) => handleFieldChange("coordinatorRatio", values[0] / 100)}
                  min={5}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Ratio of coordinator agents to regular agents (0.05-0.5)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="breakthroughThreshold"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Breakthrough Threshold</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  step="0.01"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("breakthroughThreshold", parseFloat(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value * 100]} // Scale for slider
                  onValueChange={(values) => handleFieldChange("breakthroughThreshold", values[0] / 100)}
                  min={10}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Threshold for detecting learning breakthroughs (0.1-1.0)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />
    </div>
  );

  const renderMemorySection = () => (
    <div className="space-y-6">
      <FormField
        control={form.control}
        name="memorySize"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Memory Size</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("memorySize", parseInt(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value]}
                  onValueChange={(values) => handleFieldChange("memorySize", values[0])}
                  min={100}
                  max={10000}
                  step={100}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Agent memory capacity for experiences (100-10000)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="replayBufferSize"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Replay Buffer Size</FormLabel>
            <FormControl>
              <div className="space-y-2">
                <Input
                  type="number"
                  {...field}
                  value={field.value}
                  onChange={(e) => handleFieldChange("replayBufferSize", parseInt(e.target.value) || 0)}
                />
                <Slider
                  value={[field.value]}
                  onValueChange={(values) => handleFieldChange("replayBufferSize", values[0])}
                  min={1000}
                  max={100000}
                  step={1000}
                  className="w-full"
                />
              </div>
            </FormControl>
            <FormDescription>
              Experience replay buffer size (1000-100000)
            </FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />
    </div>
  );

  return (
    <Form {...form}>
      <form className="space-y-4">
        {section === "training" && renderTrainingSection()}
        {section === "network" && renderNetworkSection()}
        {section === "memory" && renderMemorySection()}
      </form>
    </Form>
  );
}