import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer } from "ws";
import { storage } from "./storage";
import { insertAgentSchema, insertExperimentSchema, insertMessageSchema } from "@shared/schema";
import { spawn } from "child_process";
import path from "path";
import { setupPersistentTrainingRoutes } from "./routes/persistent_training_routes";

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // Setup WebSocket server but handle it carefully to avoid conflicts
  let wss: WebSocketServer | null = null;
  let broadcast: (data: any) => void = () => {};
  
  try {
    // Use a different path for WebSocket to avoid conflicts with Vite HMR
    wss = new WebSocketServer({ 
      server: httpServer,
      path: '/api/ws'
    });

    // WebSocket for real-time updates
    wss.on('connection', (ws) => {
      console.log('Client connected to WebSocket');
      
      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message.toString());
          
          // Handle different message types
          switch (data.type) {
            case 'subscribe':
              // Subscribe to specific data feeds
              break;
            case 'unsubscribe':
              // Unsubscribe from data feeds
              break;
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      });
      
      ws.on('close', () => {
        console.log('Client disconnected from WebSocket');
      });
      
      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });
    });

    // Broadcast function for real-time updates
    broadcast = (data: any) => {
      wss?.clients.forEach((client) => {
        if (client.readyState === 1) { // WebSocket.OPEN
          client.send(JSON.stringify(data));
        }
      });
    };
  } catch (error) {
    console.error('Failed to setup WebSocket server:', error);
    // Continue without WebSocket if it fails
  }

  // Agent initialization endpoint
  app.post('/api/init-agents', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/agent_initialization.py');
      
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', async (code) => {
        if (code !== 0) {
          console.error('Python agent initialization failed:', stderr);
          return res.status(500).json({ error: 'Failed to initialize agents', details: stderr });
        }
        
        try {
          // Parse the full JSON output from Python
          const trimmedOutput = stdout.trim();
          
          if (!trimmedOutput.startsWith('{')) {
            console.error('Invalid JSON output from Python:', stdout);
            return res.status(500).json({ error: 'Invalid Python output format' });
          }
          
          const systemState = JSON.parse(trimmedOutput);
          
          // Clear existing data and populate with new agents
          await storage.clearAll();
          
          // Create agents
          for (const agentData of systemState.agents) {
            try {
              await storage.createAgent({
                agentId: agentData.agent_id,
                type: agentData.agent_type,
                positionX: agentData.position.x,
                positionY: agentData.position.y,
                positionZ: agentData.position.z,
                coordinatorId: agentData.coordinator_id,
                status: agentData.status,
                isActive: agentData.is_active,
                hiddenDim: 256
              });
            } catch (agentError) {
              console.error('Error creating agent:', agentData.agent_id, agentError);
              throw agentError;
            }
          }
          
          // Create initial experiment
          const experiment = await storage.createExperiment({
            name: systemState.experimentConfig.name,
            description: systemState.experimentConfig.description,
            config: systemState.experimentConfig,
            status: 'initialized'
          });
          
          // Broadcast initialization update via WebSocket
          broadcast({
            type: 'system_initialized',
            data: {
              agents: systemState.agents,
              experiment: experiment,
              gridDimensions: systemState.gridDimensions,
              timestamp: new Date().toISOString()
            }
          });
          
          res.json({
            success: true,
            message: 'Agents initialized successfully',
            data: {
              totalAgents: systemState.totalAgents,
              coordinatorCount: systemState.coordinatorCount,
              experimentId: experiment.id
            }
          });
          
        } catch (parseError) {
          console.error('Failed to parse Python output:', parseError);
          console.error('Python stdout:', stdout);
          console.error('Python stderr:', stderr);
          res.status(500).json({ error: 'Failed to parse initialization data', details: parseError.message });
        }
      });
      
    } catch (error) {
      console.error('Error initializing agents:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  // Communication system endpoint
  app.post('/api/simulate-communication', async (req, res) => {
    try {
      const agents = await storage.getAllAgents();
      const pythonPath = path.join(process.cwd(), 'server/services/communication_system.py');
      
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });
      
      // Send agents data to Python script
      pythonProcess.stdin.write(JSON.stringify(agents));
      pythonProcess.stdin.end();
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', async (code) => {
        if (code !== 0) {
          console.error('Python communication simulation failed:', stderr);
          return res.status(500).json({ error: 'Failed to simulate communication', details: stderr });
        }
        
        try {
          const trimmedOutput = stdout.trim();
          
          if (!trimmedOutput.startsWith('{')) {
            console.error('Invalid JSON output from Python:', stdout);
            return res.status(500).json({ error: 'Invalid Python output format' });
          }
          
          const communicationResult = JSON.parse(trimmedOutput);
          
          if (communicationResult.success) {
            // Store messages in the database
            const messages = [];
            for (const round of communicationResult.communication_rounds) {
              for (const message of round.delivered_messages) {
                const storedMessage = await storage.createMessage({
                  fromAgentId: message.from_agent_id,
                  toAgentId: message.to_agent_id,
                  messageType: message.message_type,
                  content: message.content,
                  memoryPointer: message.content.memory_pointer || null,
                  isProcessed: true
                });
                messages.push(storedMessage);
              }
            }
            
            // Update communication patterns
            for (const pattern of communicationResult.final_state.communication_patterns) {
              await storage.updateCommunicationPattern(
                pattern.from_agent_id,
                pattern.to_agent_id,
                pattern.frequency,
                pattern.efficiency
              );
            }
            
            // Broadcast communication update via WebSocket
            broadcast({
              type: 'communication_simulated',
              data: {
                rounds: communicationResult.communication_rounds.length,
                messages: messages.length,
                patterns: communicationResult.final_state.communication_patterns.length,
                efficiency: communicationResult.final_state.network_metrics.network_efficiency,
                timestamp: new Date().toISOString()
              }
            });
            
            res.json({
              success: true,
              message: 'Communication simulation completed',
              data: {
                rounds: communicationResult.communication_rounds.length,
                messages: messages.length,
                patterns: communicationResult.final_state.communication_patterns.length,
                efficiency: communicationResult.final_state.network_metrics.network_efficiency,
                summary: communicationResult.summary
              }
            });
          } else {
            res.status(400).json({ error: communicationResult.error });
          }
          
        } catch (parseError) {
          console.error('Failed to parse communication results:', parseError);
          console.error('Python stdout:', stdout);
          console.error('Python stderr:', stderr);
          res.status(500).json({ error: 'Failed to parse communication results', details: parseError.message });
        }
      });
      
    } catch (error) {
      console.error('Error simulating communication:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  // Agent routes
  app.get('/api/agents', async (req, res) => {
    try {
      const agents = await storage.getAllAgents();
      res.json(agents);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch agents' });
    }
  });

  app.post('/api/agents', async (req, res) => {
    try {
      const validatedData = insertAgentSchema.parse(req.body);
      const agent = await storage.createAgent(validatedData);
      broadcast({ type: 'agent_created', data: agent });
      res.json(agent);
    } catch (error) {
      res.status(400).json({ error: 'Invalid agent data' });
    }
  });

  app.get('/api/agents/:id', async (req, res) => {
    try {
      const agent = await storage.getAgent(req.params.id);
      if (!agent) {
        return res.status(404).json({ error: 'Agent not found' });
      }
      res.json(agent);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch agent' });
    }
  });

  app.put('/api/agents/:id/status', async (req, res) => {
    try {
      const { status } = req.body;
      await storage.updateAgentStatus(req.params.id, status);
      broadcast({ type: 'agent_status_updated', data: { agentId: req.params.id, status } });
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to update agent status' });
    }
  });

  // Grid data route
  app.get('/api/grid', async (req, res) => {
    try {
      const gridData = await storage.getAgentGridData();
      res.json(gridData);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch grid data' });
    }
  });

  // Message routes
  app.get('/api/messages', async (req, res) => {
    try {
      const messages = await storage.getRecentMessages();
      res.json(messages);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch messages' });
    }
  });

  app.post('/api/messages', async (req, res) => {
    try {
      const validatedData = insertMessageSchema.parse(req.body);
      const message = await storage.createMessage(validatedData);
      broadcast({ type: 'message_created', data: message });
      res.json(message);
    } catch (error) {
      res.status(400).json({ error: 'Invalid message data' });
    }
  });

  // Memory routes
  app.get('/api/memory', async (req, res) => {
    try {
      const memoryState = await storage.getMemoryState();
      res.json(memoryState);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch memory state' });
    }
  });

  app.get('/api/memory/:vectorId', async (req, res) => {
    try {
      const vector = await storage.getMemoryVector(req.params.vectorId);
      if (!vector) {
        return res.status(404).json({ error: 'Memory vector not found' });
      }
      await storage.updateMemoryAccess(req.params.vectorId);
      res.json(vector);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch memory vector' });
    }
  });

  // Breakthrough routes
  app.get('/api/breakthroughs', async (req, res) => {
    try {
      const breakthroughs = await storage.getRecentBreakthroughs();
      res.json(breakthroughs);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch breakthroughs' });
    }
  });

  app.get('/api/breakthroughs/agent/:agentId', async (req, res) => {
    try {
      const breakthroughs = await storage.getBreakthroughsByAgent(req.params.agentId);
      res.json(breakthroughs);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch agent breakthroughs' });
    }
  });

  // Experiment routes
  app.get('/api/experiments', async (req, res) => {
    try {
      const experiments = await storage.getAllExperiments();
      res.json(experiments);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch experiments' });
    }
  });

  app.post('/api/experiments', async (req, res) => {
    try {
      const validatedData = insertExperimentSchema.parse(req.body);
      const experiment = await storage.createExperiment(validatedData);
      res.json(experiment);
    } catch (error) {
      res.status(400).json({ error: 'Invalid experiment data' });
    }
  });

  app.get('/api/experiments/:id', async (req, res) => {
    try {
      const experiment = await storage.getExperiment(parseInt(req.params.id));
      if (!experiment) {
        return res.status(404).json({ error: 'Experiment not found' });
      }
      res.json(experiment);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch experiment' });
    }
  });

  app.get('/api/experiments/:id/status', async (req, res) => {
    try {
      const trainingStatus = await storage.getTrainingStatus(parseInt(req.params.id));
      if (!trainingStatus) {
        return res.status(404).json({ error: 'Training status not found' });
      }
      res.json(trainingStatus);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch training status' });
    }
  });

  // Training control routes (multiple endpoints for compatibility)
  app.post('/api/train/start', async (req, res) => {
    try {
      const { config } = req.body;
      
      // Create a new experiment for quick training
      const experiment = await storage.createExperiment({
        name: config?.name || 'Training Session',
        description: config?.description || 'Bio-inspired MARL training',
        status: 'created',
        config: config || {},
        metrics: {}
      });
      
      // Start Python training process
      const pythonPath = path.join(process.cwd(), 'server/services/training_execution.py');
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });
      
      // Send training configuration to Python script
      const trainingConfig = {
        experiment_id: experiment.id,
        experiment_name: experiment.name,
        total_episodes: config?.total_episodes || 50,
        max_steps_per_episode: config?.max_steps_per_episode || 100,
        learning_rate: config?.learning_rate || 0.01,
        batch_size: config?.batch_size || 16,
        hidden_dim: config?.hidden_dim || 128,
        breakthrough_threshold: config?.breakthrough_threshold || 0.6,
        agents: await storage.getAllAgents()
      };
      
      pythonProcess.stdin.write(JSON.stringify(trainingConfig));
      pythonProcess.stdin.end();
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        console.log(`Training stdout: ${data}`);
        
        // Try to parse and broadcast metrics
        try {
          const lines = data.toString().split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{') && line.trim().endsWith('}')) {
              const metrics = JSON.parse(line.trim());
              if (metrics.type === 'training_metrics') {
                broadcast({ type: 'training_metrics', data: metrics });
              }
            }
          }
        } catch (e) {
          // Not JSON, just regular log
          broadcast({ type: 'training_log', data: data.toString() });
        }
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error(`Training stderr: ${data}`);
        broadcast({ type: 'training_error', data: data.toString() });
      });
      
      pythonProcess.on('close', async (code) => {
        if (code === 0) {
          try {
            const trimmedOutput = stdout.trim();
            const lastJsonLine = trimmedOutput.split('\n').reverse().find(line => 
              line.trim().startsWith('{') && line.trim().endsWith('}')
            );
            
            if (lastJsonLine) {
              const trainingResult = JSON.parse(lastJsonLine);
              
              // Update experiment with final metrics
              await storage.updateExperimentMetrics(experiment.id, trainingResult.final_metrics);
              await storage.updateExperimentStatus(experiment.id, 'completed');
              
              broadcast({ 
                type: 'training_completed', 
                data: { 
                  experimentId: experiment.id, 
                  metrics: trainingResult.final_metrics 
                } 
              });
            }
          } catch (parseError) {
            console.error('Failed to parse training results:', parseError);
            await storage.updateExperimentStatus(experiment.id, 'failed');
            broadcast({ type: 'training_failed', data: { experimentId: experiment.id } });
          }
        } else {
          console.error('Training process failed:', stderr);
          await storage.updateExperimentStatus(experiment.id, 'failed');
          broadcast({ type: 'training_failed', data: { experimentId: experiment.id } });
        }
      });
      
      await storage.updateExperimentStatus(experiment.id, 'running');
      broadcast({ type: 'training_started', data: { experimentId: experiment.id } });
      
      res.json({ 
        success: true, 
        message: 'Training started',
        experimentId: experiment.id,
        experiment: experiment
      });
    } catch (error) {
      console.error('Error starting training:', error);
      res.status(500).json({ error: 'Failed to start training: ' + error.message });
    }
  });

  // Ray RLlib integration routes
  app.post('/api/training/ray/start', async (req, res) => {
    try {
      const { config } = req.body;
      
      // Create a new experiment for Ray training
      const experiment = await storage.createExperiment({
        name: config?.name || 'Ray RLlib Training',
        description: config?.description || 'Bio-inspired MARL training with Ray RLlib',
        status: 'created',
        config: { ...config, use_ray: true },
        metrics: {}
      });
      
      // Start Ray training process (with fallback)
      const pythonPath = path.join(process.cwd(), 'server/services/ray_fallback.py');
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });
      
      // Send Ray training configuration to Python script
      const rayConfig = {
        experiment_id: experiment.id,
        experiment_name: experiment.name,
        use_ray: true,
        total_episodes: config?.total_episodes || 100,
        max_steps_per_episode: config?.max_steps_per_episode || 500,
        learning_rate: config?.learning_rate || 3e-4,
        batch_size: config?.batch_size || 128,
        train_batch_size: config?.train_batch_size || 4000,
        hidden_dim: config?.hidden_dim || 256,
        num_rollout_workers: config?.num_rollout_workers || 4,
        num_attention_heads: config?.num_attention_heads || 8,
        pheromone_decay: config?.pheromone_decay || 0.95,
        neural_plasticity_rate: config?.neural_plasticity_rate || 0.1,
        communication_range: config?.communication_range || 2.0,
        breakthrough_threshold: config?.breakthrough_threshold || 0.7,
        agents: await storage.getAllAgents()
      };
      
      pythonProcess.stdin.write(JSON.stringify(rayConfig));
      pythonProcess.stdin.end();
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        console.log(`Ray Training stdout: ${data}`);
        
        // Try to parse and broadcast Ray metrics
        try {
          const lines = data.toString().split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{') && line.trim().endsWith('}')) {
              const metrics = JSON.parse(line.trim());
              if (metrics.type === 'ray_training_metrics') {
                broadcast({ type: 'ray_training_metrics', data: metrics });
              }
            }
          }
        } catch (e) {
          // Not JSON, just regular log
          broadcast({ type: 'ray_training_log', data: data.toString() });
        }
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error(`Ray Training stderr: ${data}`);
        broadcast({ type: 'ray_training_error', data: data.toString() });
      });
      
      pythonProcess.on('close', async (code) => {
        if (code === 0) {
          try {
            const trimmedOutput = stdout.trim();
            const lastJsonLine = trimmedOutput.split('\n').reverse().find(line => 
              line.trim().startsWith('{') && line.trim().endsWith('}')
            );
            
            if (lastJsonLine) {
              const rayResult = JSON.parse(lastJsonLine);
              
              if (rayResult.success) {
                // Handle both Ray RLlib and fallback system results
                const finalMetrics = rayResult.result?.final_metrics || rayResult.metrics;
                
                if (finalMetrics) {
                  await storage.updateExperimentMetrics(experiment.id, finalMetrics);
                }
                await storage.updateExperimentStatus(experiment.id, 'completed');
                
                broadcast({ 
                  type: 'ray_training_completed', 
                  data: { 
                    experimentId: experiment.id, 
                    metrics: finalMetrics,
                    trainingMethod: rayResult.ray_available ? 'ray_rllib' : 'fallback',
                    message: rayResult.message
                  } 
                });
              } else {
                await storage.updateExperimentStatus(experiment.id, 'failed');
                broadcast({ type: 'ray_training_failed', data: { experimentId: experiment.id, error: rayResult.error } });
              }
            }
          } catch (parseError) {
            console.error('Failed to parse Ray training results:', parseError);
            await storage.updateExperimentStatus(experiment.id, 'failed');
            broadcast({ type: 'ray_training_failed', data: { experimentId: experiment.id } });
          }
        } else {
          console.error('Ray training process failed:', stderr);
          await storage.updateExperimentStatus(experiment.id, 'failed');
          broadcast({ type: 'ray_training_failed', data: { experimentId: experiment.id } });
        }
      });
      
      await storage.updateExperimentStatus(experiment.id, 'running');
      broadcast({ type: 'ray_training_started', data: { experimentId: experiment.id } });
      
      res.json({ 
        success: true, 
        message: 'Ray training started',
        experimentId: experiment.id,
        experiment: experiment,
        trainingMethod: 'ray_rllib'
      });
    } catch (error) {
      console.error('Error starting Ray training:', error);
      res.status(500).json({ error: 'Failed to start Ray training: ' + error.message });
    }
  });

  app.get('/api/training/ray/config-template', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/ray_fallback.py');
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const template = JSON.parse(stdout.trim());
            res.json(template);
          } catch (parseError) {
            console.error('Failed to parse Ray config template:', parseError);
            res.status(500).json({ error: 'Failed to parse Ray config template' });
          }
        } else {
          console.error('Ray config template generation failed:', stderr);
          res.status(500).json({ error: 'Failed to generate Ray config template' });
        }
      });
    } catch (error) {
      console.error('Error generating Ray config template:', error);
      res.status(500).json({ error: 'Failed to generate Ray config template' });
    }
  });

  app.post('/api/training/start', async (req, res) => {
    try {
      const { experimentId, config } = req.body;
      
      // Create a new experiment if needed
      let experiment;
      if (!experimentId) {
        experiment = await storage.createExperiment({
          name: config.name || 'New Training Session',
          description: config.description || 'Bio-inspired MARL training',
          status: 'created',
          config: config,
          metrics: {}
        });
      } else {
        experiment = await storage.getExperiment(experimentId);
        if (!experiment) {
          return res.status(404).json({ error: 'Experiment not found' });
        }
      }
      
      // Start Python training process
      const pythonPath = path.join(process.cwd(), 'server/services/training_execution.py');
      const pythonProcess = spawn('python3', [pythonPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });
      
      // Send training configuration to Python script
      const trainingConfig = {
        experiment_id: experiment.id,
        experiment_name: experiment.name,
        total_episodes: config.total_episodes || 100,
        max_steps_per_episode: config.max_steps_per_episode || 200,
        learning_rate: config.learning_rate || 0.001,
        batch_size: config.batch_size || 32,
        hidden_dim: config.hidden_dim || 256,
        breakthrough_threshold: config.breakthrough_threshold || 0.7,
        agents: await storage.getAllAgents()
      };
      
      pythonProcess.stdin.write(JSON.stringify(trainingConfig));
      pythonProcess.stdin.end();
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        console.log(`Training stdout: ${data}`);
        
        // Try to parse and broadcast metrics
        try {
          const lines = data.toString().split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{') && line.trim().endsWith('}')) {
              const metrics = JSON.parse(line.trim());
              if (metrics.type === 'training_metrics') {
                broadcast({ type: 'training_metrics', data: metrics });
              }
            }
          }
        } catch (e) {
          // Not JSON, just regular log
          broadcast({ type: 'training_log', data: data.toString() });
        }
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error(`Training stderr: ${data}`);
        broadcast({ type: 'training_error', data: data.toString() });
      });
      
      pythonProcess.on('close', async (code) => {
        if (code === 0) {
          try {
            const trimmedOutput = stdout.trim();
            const lastJsonLine = trimmedOutput.split('\n').reverse().find(line => 
              line.trim().startsWith('{') && line.trim().endsWith('}')
            );
            
            if (lastJsonLine) {
              const trainingResult = JSON.parse(lastJsonLine);
              
              // Update experiment with final metrics
              await storage.updateExperimentMetrics(experiment.id, trainingResult.final_metrics);
              await storage.updateExperimentStatus(experiment.id, 'completed');
              
              broadcast({ 
                type: 'training_completed', 
                data: { 
                  experimentId: experiment.id, 
                  metrics: trainingResult.final_metrics 
                } 
              });
            }
          } catch (parseError) {
            console.error('Failed to parse training results:', parseError);
            await storage.updateExperimentStatus(experiment.id, 'failed');
            broadcast({ type: 'training_failed', data: { experimentId: experiment.id } });
          }
        } else {
          console.error('Training process failed:', stderr);
          await storage.updateExperimentStatus(experiment.id, 'failed');
          broadcast({ type: 'training_failed', data: { experimentId: experiment.id } });
        }
      });
      
      await storage.updateExperimentStatus(experiment.id, 'running');
      broadcast({ type: 'training_started', data: { experimentId: experiment.id } });
      
      res.json({ 
        success: true, 
        message: 'Training started',
        experimentId: experiment.id,
        experiment: experiment
      });
    } catch (error) {
      console.error('Error starting training:', error);
      res.status(500).json({ error: 'Failed to start training' });
    }
  });

  app.post('/api/training/stop', async (req, res) => {
    try {
      const { experimentId } = req.body;
      await storage.updateExperimentStatus(experimentId, 'stopped');
      broadcast({ type: 'training_stopped', data: { experimentId } });
      res.json({ success: true, message: 'Training stopped' });
    } catch (error) {
      res.status(500).json({ error: 'Failed to stop training' });
    }
  });

  // Advanced Learning API endpoints
  app.post('/api/advanced/initialize', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/advanced_learning_api.py');
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${path.join(process.cwd(), 'server/services')}')
from advanced_learning_api import handle_advanced_learning_request
import json

request_data = json.loads(input())
result = handle_advanced_learning_request('initialize', request_data)
print(json.dumps(result))
`], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      pythonProcess.stdin.write(JSON.stringify(req.body));
      pythonProcess.stdin.end();

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            broadcast({ type: 'advanced_learning_initialized', data: result });
            res.json(result);
          } catch (error) {
            console.error('Failed to parse advanced learning result:', error);
            res.status(500).json({ error: 'Failed to parse result' });
          }
        } else {
          console.error('Advanced learning initialization failed:', stderr);
          res.status(500).json({ error: 'Failed to initialize advanced learning' });
        }
      });
    } catch (error) {
      console.error('Error initializing advanced learning:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/advanced/status', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/advanced_learning_simple.py');
      const pythonProcess = spawn('python3', [pythonPath, 'status'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (error) {
            console.error('Failed to parse status result:', error);
            res.status(500).json({ error: 'Failed to parse status' });
          }
        } else {
          console.error('Advanced learning status failed:', stderr);
          res.status(500).json({ error: 'Failed to get status' });
        }
      });
    } catch (error) {
      console.error('Error getting advanced learning status:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.post('/api/advanced/start_training', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/advanced_learning_api.py');
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${path.join(process.cwd(), 'server/services')}')
from advanced_learning_api import handle_advanced_learning_request
import json

request_data = json.loads(input())
result = handle_advanced_learning_request('start_training', request_data)
print(json.dumps(result))
`], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      pythonProcess.stdin.write(JSON.stringify(req.body));
      pythonProcess.stdin.end();

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        // Try to parse and broadcast real-time metrics
        try {
          const lines = data.toString().split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{') && line.trim().endsWith('}')) {
              const metrics = JSON.parse(line.trim());
              if (metrics.type === 'advanced_learning_metrics') {
                broadcast({ type: 'advanced_metrics', data: metrics });
              }
            }
          }
        } catch (e) {
          // Not JSON, regular log
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            broadcast({ type: 'advanced_training_started', data: result });
            res.json(result);
          } catch (error) {
            console.error('Failed to parse training result:', error);
            res.status(500).json({ error: 'Failed to parse result' });
          }
        } else {
          console.error('Advanced training failed:', stderr);
          res.status(500).json({ error: 'Failed to start advanced training' });
        }
      });
    } catch (error) {
      console.error('Error starting advanced training:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/advanced/curriculum_progress', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/advanced_learning_simple.py');
      const pythonProcess = spawn('python3', [pythonPath, 'curriculum_progress'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      let stdout = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (error) {
            res.status(500).json({ error: 'Failed to parse curriculum progress' });
          }
        } else {
          res.status(500).json({ error: 'Failed to get curriculum progress' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/advanced/transfer_recommendations', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/advanced_learning_simple.py');
      const pythonProcess = spawn('python3', [pythonPath, 'transfer_recommendations'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      let stdout = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (error) {
            res.status(500).json({ error: 'Failed to parse transfer recommendations' });
          }
        } else {
          res.status(500).json({ error: 'Failed to get transfer recommendations' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/advanced/meta_insights', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/advanced_learning_simple.py');
      const pythonProcess = spawn('python3', [pythonPath, 'meta_insights'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      let stdout = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (error) {
            res.status(500).json({ error: 'Failed to parse meta-learning insights' });
          }
        } else {
          res.status(500).json({ error: 'Failed to get meta-learning insights' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/training/status', async (req, res) => {
    try {
      const experiments = await storage.getAllExperiments();
      const runningExperiment = experiments.find(exp => exp.status === 'running');
      
      if (runningExperiment) {
        const trainingStatus = await storage.getTrainingStatus(runningExperiment.id);
        res.json(trainingStatus);
      } else {
        res.json({ 
          experiment: null, 
          currentEpisode: 0, 
          currentStep: 0, 
          recentMetrics: [], 
          isRunning: false 
        });
      }
    } catch (error) {
      res.status(500).json({ error: 'Failed to get training status' });
    }
  });

  // Communication patterns
  app.get('/api/communication-patterns', async (req, res) => {
    try {
      const patterns = await storage.getCommunicationPatterns();
      res.json(patterns);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch communication patterns' });
    }
  });

  // Metrics routes
  app.get('/api/metrics', async (req, res) => {
    try {
      const metrics = await storage.getRecentMetrics();
      res.json(metrics);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch metrics' });
    }
  });

  app.get('/api/metrics/experiment/:id', async (req, res) => {
    try {
      const metrics = await storage.getMetricsByExperiment(parseInt(req.params.id));
      res.json(metrics);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch experiment metrics' });
    }
  });

  // Enhanced Communication & Memory APIs
  app.get('/api/communication/enhanced-stats', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/enhanced_api.py');
      const pythonProcess = spawn('python3', [pythonPath, 'get_enhanced_communication_stats'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (parseError) {
            res.status(500).json({ error: 'Failed to parse enhanced communication stats' });
          }
        } else {
          console.error('Enhanced communication stats error:', stderr);
          res.status(500).json({ error: 'Failed to get enhanced communication statistics' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch enhanced communication stats' });
    }
  });

  app.get('/api/memory/advanced-stats', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/enhanced_api.py');
      const pythonProcess = spawn('python3', [pythonPath, 'get_advanced_memory_stats'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (parseError) {
            res.status(500).json({ error: 'Failed to parse advanced memory stats' });
          }
        } else {
          console.error('Advanced memory stats error:', stderr);
          res.status(500).json({ error: 'Failed to get advanced memory statistics' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch advanced memory stats' });
    }
  });

  app.post('/api/memory/query-semantic', async (req, res) => {
    try {
      const { query_text, max_results = 10, threshold = 0.7 } = req.body;
      
      const pythonPath = path.join(process.cwd(), 'server/services/enhanced_api.py');
      const pythonProcess = spawn('python3', [pythonPath, 'semantic_memory_query'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });

      const queryData = {
        query_text,
        max_results,
        threshold
      };

      pythonProcess.stdin.write(JSON.stringify(queryData));
      pythonProcess.stdin.end();

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (parseError) {
            res.status(500).json({ error: 'Failed to parse semantic query results' });
          }
        } else {
          console.error('Semantic query error:', stderr);
          res.status(500).json({ error: 'Failed to execute semantic memory query' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to process semantic memory query' });
    }
  });

  app.post('/api/memory/query-associative', async (req, res) => {
    try {
      const { vector_id, max_depth = 3, max_results = 15 } = req.body;
      
      const pythonPath = path.join(process.cwd(), 'server/services/enhanced_api.py');
      const pythonProcess = spawn('python3', [pythonPath, 'associative_memory_query'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });

      const queryData = {
        vector_id,
        max_depth,
        max_results
      };

      pythonProcess.stdin.write(JSON.stringify(queryData));
      pythonProcess.stdin.end();

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (parseError) {
            res.status(500).json({ error: 'Failed to parse associative query results' });
          }
        } else {
          console.error('Associative query error:', stderr);
          res.status(500).json({ error: 'Failed to execute associative memory query' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to process associative memory query' });
    }
  });

  app.get('/api/communication/bandwidth-usage', async (req, res) => {
    try {
      const pythonPath = path.join(process.cwd(), 'server/services/enhanced_api.py');
      const pythonProcess = spawn('python3', [pythonPath, 'get_bandwidth_usage'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (parseError) {
            res.status(500).json({ error: 'Failed to parse bandwidth usage data' });
          }
        } else {
          console.error('Bandwidth usage error:', stderr);
          res.status(500).json({ error: 'Failed to get bandwidth usage statistics' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch bandwidth usage data' });
    }
  });

  app.post('/api/memory/predictive-prefetch', async (req, res) => {
    try {
      const { access_sequence, max_predictions = 5 } = req.body;
      
      const pythonPath = path.join(process.cwd(), 'server/services/enhanced_api.py');
      const pythonProcess = spawn('python3', [pythonPath, 'predictive_prefetch'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONPATH: process.cwd() }
      });

      const queryData = {
        access_sequence,
        max_predictions
      };

      pythonProcess.stdin.write(JSON.stringify(queryData));
      pythonProcess.stdin.end();

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
          } catch (parseError) {
            res.status(500).json({ error: 'Failed to parse predictive prefetch results' });
          }
        } else {
          console.error('Predictive prefetch error:', stderr);
          res.status(500).json({ error: 'Failed to execute predictive prefetching' });
        }
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to process predictive prefetch request' });
    }
  });

  // Initialize demo data
  app.post('/api/initialize-demo', async (req, res) => {
    try {
      // Initialize 4x3x3 grid with 27 regular agents and 3 coordinators
      const agents = [];
      let agentCounter = 0;
      
      // Create regular agents
      for (let x = 0; x < 4; x++) {
        for (let y = 0; y < 3; y++) {
          for (let z = 0; z < 3; z++) {
            if (agentCounter < 27) {
              const agentId = `agent_${agentCounter}`;
              const coordinatorId = `coordinator_${Math.floor(agentCounter / 9)}`;
              
              agents.push({
                agentId,
                type: 'regular',
                positionX: x,
                positionY: y,
                positionZ: z,
                coordinatorId,
                status: 'idle',
                hiddenDim: 256,
                isActive: true,
              });
              agentCounter++;
            }
          }
        }
      }
      
      // Create 3 coordinator agents at strategic positions
      const coordinatorPositions = [
        { x: 1, y: 1, z: 1 },
        { x: 2, y: 1, z: 1 },
        { x: 3, y: 1, z: 1 },
      ];
      
      for (let i = 0; i < 3; i++) {
        const coordinatorId = `coordinator_${i}`;
        agents.push({
          agentId: coordinatorId,
          type: 'coordinator',
          positionX: coordinatorPositions[i].x,
          positionY: coordinatorPositions[i].y,
          positionZ: coordinatorPositions[i].z,
          coordinatorId: null,
          status: 'idle',
          hiddenDim: 512,
          isActive: true,
        });
      }
      
      // Create agents in storage
      const createdAgents = [];
      for (const agent of agents) {
        const created = await storage.createAgent(agent);
        createdAgents.push(created);
      }
      
      broadcast({ type: 'demo_initialized', data: createdAgents });
      res.json({ success: true, agents: createdAgents });
    } catch (error) {
      res.status(500).json({ error: 'Failed to initialize demo data' });
    }
  });

  // Setup persistent training routes with database access
  setupPersistentTrainingRoutes(app, broadcast);

  // Enhanced Research Framework API Routes
  app.get('/api/research/status', async (req, res) => {
    try {
      const { spawn } = require('child_process');
      const path = require('path');
      
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${process.cwd()}/server/services')
from enhanced_research_api import get_research_api
import asyncio
import json

async def main():
    api = get_research_api()
    status = await api.get_research_status()
    print(json.dumps(status))

asyncio.run(main())
`], { stdio: ['pipe', 'pipe', 'pipe'] });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Research API failed:', stderr);
          return res.status(500).json({ error: 'Research framework unavailable', details: stderr });
        }

        try {
          const status = JSON.parse(stdout.trim());
          res.json(status);
        } catch (parseError) {
          console.error('Failed to parse research status:', parseError);
          res.status(500).json({ error: 'Invalid response format' });
        }
      });
    } catch (error) {
      console.error('Research status error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.post('/api/research/experiments', async (req, res) => {
    try {
      const experimentConfig = req.body;
      const { spawn } = require('child_process');
      
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${process.cwd()}/server/services')
from enhanced_research_api import get_research_api
import asyncio
import json

async def main():
    api = get_research_api()
    config = ${JSON.stringify(experimentConfig)}
    result = await api.create_experiment_from_config(config)
    print(json.dumps(result))

asyncio.run(main())
`], { stdio: ['pipe', 'pipe', 'pipe'] });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Experiment creation failed:', stderr);
          return res.status(500).json({ error: 'Failed to create experiment', details: stderr });
        }

        try {
          const result = JSON.parse(stdout.trim());
          if (result.success) {
            broadcast({ type: 'research_experiment_created', data: result });
          }
          res.json(result);
        } catch (parseError) {
          console.error('Failed to parse experiment result:', parseError);
          res.status(500).json({ error: 'Invalid response format' });
        }
      });
    } catch (error) {
      console.error('Experiment creation error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.post('/api/research/experiments/:id/run', async (req, res) => {
    try {
      const experimentId = req.params.id;
      const { spawn } = require('child_process');
      
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${process.cwd()}/server/services')
from enhanced_research_api import get_research_api
import asyncio
import json

async def main():
    api = get_research_api()
    result = await api.run_experiment('${experimentId}')
    print(json.dumps(result))

asyncio.run(main())
`], { stdio: ['pipe', 'pipe', 'pipe'] });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Experiment execution failed:', stderr);
          return res.status(500).json({ error: 'Experiment execution failed', details: stderr });
        }

        try {
          const result = JSON.parse(stdout.trim());
          if (result.success) {
            broadcast({ type: 'research_experiment_completed', data: result });
          }
          res.json(result);
        } catch (parseError) {
          console.error('Failed to parse experiment result:', parseError);
          res.status(500).json({ error: 'Invalid response format' });
        }
      });
    } catch (error) {
      console.error('Experiment execution error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/research/experiments', async (req, res) => {
    try {
      const { spawn } = require('child_process');
      
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${process.cwd()}/server/services')
from enhanced_research_api import get_research_api
import asyncio
import json

async def main():
    api = get_research_api()
    result = await api.list_experiments()
    print(json.dumps(result))

asyncio.run(main())
`], { stdio: ['pipe', 'pipe', 'pipe'] });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Failed to list experiments:', stderr);
          return res.status(500).json({ error: 'Failed to list experiments', details: stderr });
        }

        try {
          const result = JSON.parse(stdout.trim());
          res.json(result);
        } catch (parseError) {
          console.error('Failed to parse experiments list:', parseError);
          res.status(500).json({ error: 'Invalid response format' });
        }
      });
    } catch (error) {
      console.error('List experiments error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/api/research/hypotheses', async (req, res) => {
    try {
      const { spawn } = require('child_process');
      
      const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.append('${process.cwd()}/server/services')
from enhanced_research_api import get_research_api
import asyncio
import json

async def main():
    api = get_research_api()
    result = await api.get_hypothesis_validation_summary()
    print(json.dumps(result))

asyncio.run(main())
`], { stdio: ['pipe', 'pipe', 'pipe'] });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Failed to get hypothesis summary:', stderr);
          return res.status(500).json({ error: 'Failed to get hypothesis summary', details: stderr });
        }

        try {
          const result = JSON.parse(stdout.trim());
          res.json(result);
        } catch (parseError) {
          console.error('Failed to parse hypothesis summary:', parseError);
          res.status(500).json({ error: 'Invalid response format' });
        }
      });
    } catch (error) {
      console.error('Hypothesis summary error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  return httpServer;
}
