import { Express } from "express";
import { spawn } from "child_process";
import path from "path";
import { storage } from "../storage";
import { insertExperimentSchema } from "@shared/schema";

export function setupPersistentTrainingRoutes(app: Express, broadcast?: (data: any) => void) {
  // Default broadcast function if not provided
  const safeBroadcast = broadcast || ((data: any) => {
    console.log('Broadcasting:', data.type);
  });
  
  // Start persistent training with database persistence
  app.post('/api/training/persistent/start', async (req, res) => {
    try {
      const config = req.body;
      
      // Create experiment first
      const experiment = await storage.createExperiment({
        name: config?.name || 'Persistent MARL Training Session',
        description: 'Long-running MARL experiment with database persistence',
        config: config,
        status: 'initializing'
      });
      
      const pythonPath = path.join(process.cwd(), 'server/services/persistent_training_service.py');
      
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });
      
      // Send configuration to Python script
      const persistentConfig = {
        experiment_id: experiment.id,
        experiment_name: experiment.name,
        total_episodes: config?.total_episodes || 200,
        max_steps_per_episode: config?.max_steps_per_episode || 300,
        learning_rate: config?.learning_rate || 3e-4,
        batch_size: config?.batch_size || 32,
        hidden_dim: config?.hidden_dim || 256,
        breakthrough_threshold: config?.breakthrough_threshold || 0.75,
        checkpoint_interval: config?.checkpoint_interval || 10,
        agents: await storage.getAllAgents()
      };
      
      pythonProcess.stdin.write(JSON.stringify(persistentConfig));
      pythonProcess.stdin.end();
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        console.log(`Persistent Training stdout: ${data}`);
        
        // Try to parse and broadcast metrics
        try {
          const lines = data.toString().split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{') && line.trim().endsWith('}')) {
              const metrics = JSON.parse(line.trim());
              if (metrics.type === 'persistent_training_metrics') {
                safeBroadcast({ 
                  type: 'persistent_training_metrics', 
                  data: {
                    ...metrics,
                    experiment_id: experiment.id,
                    database_persistence: true
                  }
                });
              }
            }
          }
        } catch (e) {
          // Not JSON, just regular log
          safeBroadcast({ 
            type: 'persistent_training_log', 
            data: {
              message: data.toString(),
              experiment_id: experiment.id,
              timestamp: new Date().toISOString()
            }
          });
        }
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error(`Persistent Training stderr: ${data}`);
        safeBroadcast({ 
          type: 'persistent_training_error', 
          data: {
            error: data.toString(),
            experiment_id: experiment.id,
            timestamp: new Date().toISOString()
          }
        });
      });
      
      pythonProcess.on('close', async (code) => {
        try {
          if (code === 0) {
            try {
              const trimmedOutput = stdout.trim();
              const lastJsonLine = trimmedOutput.split('\n').reverse().find(line => 
                line.trim().startsWith('{') && line.trim().endsWith('}')
              );
              
              if (lastJsonLine) {
                const trainingResult = JSON.parse(lastJsonLine);
                
                if (trainingResult.success) {
                  // Update experiment with final results
                  await storage.updateExperimentMetrics(experiment.id, trainingResult.final_metrics);
                  await storage.updateExperimentStatus(experiment.id, 'completed');
                  
                  // Broadcast completion
                  safeBroadcast({
                    type: 'persistent_training_completed',
                    data: {
                      experiment_id: experiment.id,
                      final_metrics: trainingResult.final_metrics,
                      database_persistence: true,
                      timestamp: new Date().toISOString()
                    }
                  });
                  
                  // Don't send response if headers already sent
                  if (!res.headersSent) {
                    res.json({
                      success: true,
                      message: 'Persistent training completed successfully',
                      experiment_id: experiment.id,
                      final_metrics: trainingResult.final_metrics,
                      database_persistence: true
                    });
                  }
                } else {
                  await storage.updateExperimentStatus(experiment.id, 'failed');
                  if (!res.headersSent) {
                    res.status(500).json({
                      success: false,
                      error: trainingResult.error || 'Unknown error',
                      experiment_id: experiment.id,
                      database_persistence: true
                    });
                  }
                }
              } else {
                await storage.updateExperimentStatus(experiment.id, 'failed');
                if (!res.headersSent) {
                  res.status(500).json({
                    success: false,
                    error: 'No valid result from training process',
                    experiment_id: experiment.id,
                    stderr: stderr
                  });
                }
              }
            } catch (parseError) {
              console.error('Failed to parse training result:', parseError);
              await storage.updateExperimentStatus(experiment.id, 'failed');
              if (!res.headersSent) {
                res.status(500).json({
                  success: false,
                  error: 'Failed to parse training results',
                  experiment_id: experiment.id,
                  details: parseError.message
                });
              }
            }
          } else {
            console.error('Persistent training process failed:', stderr);
            await storage.updateExperimentStatus(experiment.id, 'failed');
            if (!res.headersSent) {
              res.status(500).json({
                success: false,
                error: 'Training process failed',
                experiment_id: experiment.id,
                details: stderr,
                exit_code: code
              });
            }
          }
        } catch (error) {
          console.error('Error in process close handler:', error);
          if (!res.headersSent) {
            res.status(500).json({
              success: false,
              error: 'Internal server error',
              experiment_id: experiment.id
            });
          }
        }
      });
      
      // Update experiment status to running
      await storage.updateExperimentStatus(experiment.id, 'running');
      
      // Return immediate response
      res.json({
        success: true,
        message: 'Persistent training started successfully',
        experiment_id: experiment.id,
        database_persistence: true
      });
      
    } catch (error) {
      console.error('Failed to start persistent training:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to start persistent training', 
        details: error
      });
    }
  });

  // Get persistent training experiment status
  app.get('/api/training/persistent/status/:experimentId', async (req, res) => {
    try {
      const experimentId = parseInt(req.params.experimentId);
      const trainingStatus = await storage.getTrainingStatus(experimentId);
      
      if (!trainingStatus) {
        return res.status(404).json({ error: 'Experiment not found' });
      }
      
      // Get additional persistence-specific information
      const experiment = trainingStatus.experiment;
      const recentMetrics = trainingStatus.recentMetrics;
      
      // Calculate additional statistics
      const episodeMetrics = recentMetrics.filter(m => m.metricType === 'episode_reward');
      const networkEfficiencyMetrics = recentMetrics.filter(m => m.metricType === 'network_efficiency');
      const memoryMetrics = recentMetrics.filter(m => m.metricType === 'memory_utilization');
      
      const persistentStatus = {
        ...trainingStatus,
        database_persistence: true,
        statistics: {
          total_episodes_completed: episodeMetrics.length,
          avg_episode_reward: episodeMetrics.length > 0 ? 
            episodeMetrics.reduce((sum, m) => sum + m.value, 0) / episodeMetrics.length : 0,
          avg_network_efficiency: networkEfficiencyMetrics.length > 0 ? 
            networkEfficiencyMetrics.reduce((sum, m) => sum + m.value, 0) / networkEfficiencyMetrics.length : 0,
          avg_memory_utilization: memoryMetrics.length > 0 ? 
            memoryMetrics.reduce((sum, m) => sum + m.value, 0) / memoryMetrics.length : 0,
          metrics_count: recentMetrics.length
        }
      };
      
      res.json(persistentStatus);
      
    } catch (error) {
      console.error('Failed to get persistent training status:', error);
      res.status(500).json({ error: 'Failed to get training status' });
    }
  });

  // Get all persistent experiments
  app.get('/api/training/persistent/experiments', async (req, res) => {
    try {
      const experiments = await storage.getAllExperiments();
      
      // Add statistics for each experiment
      const experimentsWithStats = await Promise.all(
        experiments.map(async (experiment) => {
          const metrics = await storage.getMetricsByExperiment(experiment.id);
          const breakthroughs = await storage.getRecentBreakthroughs(100)
            .then(all => all.filter(b => b.timestamp && 
              experiment.startTime && 
              b.timestamp >= experiment.startTime &&
              (!experiment.endTime || b.timestamp <= experiment.endTime)
            ));
          
          return {
            ...experiment,
            database_persistence: true,
            statistics: {
              total_metrics: metrics.length,
              total_breakthroughs: breakthroughs.length,
              last_activity: metrics.length > 0 ? 
                Math.max(...metrics.map(m => m.timestamp?.getTime() || 0)) : null
            }
          };
        })
      );
      
      res.json(experimentsWithStats);
      
    } catch (error) {
      console.error('Failed to get persistent experiments:', error);
      res.status(500).json({ error: 'Failed to get experiments' });
    }
  });

  // Get experiment metrics with pagination
  app.get('/api/training/persistent/experiments/:experimentId/metrics', async (req, res) => {
    try {
      const experimentId = parseInt(req.params.experimentId);
      const limit = parseInt(req.query.limit as string) || 100;
      const metricType = req.query.type as string;
      
      let metrics = await storage.getMetricsByExperiment(experimentId);
      
      // Filter by type if specified
      if (metricType) {
        metrics = metrics.filter(m => m.metricType === metricType);
      }
      
      // Apply limit
      metrics = metrics.slice(0, limit);
      
      res.json({
        experiment_id: experimentId,
        metrics,
        total_count: metrics.length,
        database_persistence: true
      });
      
    } catch (error) {
      console.error('Failed to get experiment metrics:', error);
      res.status(500).json({ error: 'Failed to get experiment metrics' });
    }
  });

  // Get memory vectors for an experiment (if stored during training)
  app.get('/api/training/persistent/experiments/:experimentId/memory', async (req, res) => {
    try {
      const experimentId = parseInt(req.params.experimentId);
      const limit = parseInt(req.query.limit as string) || 50;
      
      // Get all memory vectors and filter by experiment context
      const allVectors = await storage.getAllMemoryVectors();
      const experimentVectors = allVectors
        .filter(v => v.content && typeof v.content === 'object' && 
                    (v.content as any).episode !== undefined)
        .slice(0, limit);
      
      res.json({
        experiment_id: experimentId,
        memory_vectors: experimentVectors.map(v => ({
          vector_id: v.vectorId,
          vector_type: v.vectorType,
          importance: v.importance,
          access_count: v.accessCount,
          created_at: v.createdAt,
          content_summary: v.content ? {
            episode: (v.content as any).episode,
            reward: (v.content as any).reward,
            breakthrough_count: (v.content as any).breakthrough_count
          } : null
        })),
        total_count: experimentVectors.length,
        database_persistence: true
      });
      
    } catch (error) {
      console.error('Failed to get experiment memory vectors:', error);
      res.status(500).json({ error: 'Failed to get memory vectors' });
    }
  });

  // Test database connectivity
  app.get('/api/training/persistent/health', async (req, res) => {
    try {
      // Test basic operations
      const agents = await storage.getAllAgents();
      const experiments = await storage.getAllExperiments();
      const recentMetrics = await storage.getRecentMetrics(10);
      const memoryVectors = await storage.getAllMemoryVectors();
      
      res.json({
        status: 'healthy',
        database_persistence: true,
        statistics: {
          total_agents: agents.length,
          total_experiments: experiments.length,
          recent_metrics: recentMetrics.length,
          memory_vectors: memoryVectors.length
        },
        database_url_configured: !!process.env.DATABASE_URL,
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      console.error('Database health check failed:', error);
      res.status(500).json({ 
        status: 'unhealthy',
        database_persistence: false,
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });
}