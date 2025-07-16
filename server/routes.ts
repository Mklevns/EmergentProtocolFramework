import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer } from "ws";
import { storage } from "./storage";
import { insertAgentSchema, insertExperimentSchema, insertMessageSchema } from "@shared/schema";
import { spawn } from "child_process";
import path from "path";

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

  // Training control routes
  app.post('/api/training/start', async (req, res) => {
    try {
      const { experimentId, config } = req.body;
      
      // Start Python training process
      const pythonProcess = spawn('python', [
        path.join(__dirname, 'services', 'training_orchestrator.py'),
        '--experiment-id', experimentId.toString(),
        '--config', JSON.stringify(config)
      ]);
      
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Training stdout: ${data}`);
        broadcast({ type: 'training_log', data: data.toString() });
      });
      
      pythonProcess.stderr.on('data', (data) => {
        console.error(`Training stderr: ${data}`);
        broadcast({ type: 'training_error', data: data.toString() });
      });
      
      await storage.updateExperimentStatus(experimentId, 'running');
      broadcast({ type: 'training_started', data: { experimentId } });
      
      res.json({ success: true, message: 'Training started' });
    } catch (error) {
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

  return httpServer;
}
