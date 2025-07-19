# Brain-Inspired Multi-Agent Reinforcement Learning System

A cutting-edge multi-agent reinforcement learning (MARL) platform that enables sophisticated, hierarchical agent interactions through advanced communication and spatial positioning technologies.

## 🌟 Features

- **3D Agent Grid System**: 30 intelligent agents organized in a 4×3×3 grid structure
- **Hierarchical Communication**: Multi-level message routing with emergent protocols
- **Database Persistence**: Robust PostgreSQL integration for long-running experiments
- **Real-time Visualization**: Live data streaming via WebSocket connections
- **Bio-inspired Algorithms**: Pheromone trails, neural plasticity, and swarm coordination
- **Advanced Learning Systems**: Curriculum learning, meta-learning, and transfer learning
- **Breakthrough Detection**: Automatic identification of significant agent achievements

## 🏗️ Architecture

### Frontend
- **Framework**: React with TypeScript and Vite
- **Styling**: Tailwind CSS with shadcn/ui components
- **State Management**: TanStack Query for server state
- **Routing**: Wouter for client-side navigation
- **Real-time**: WebSocket integration for live updates

### Backend
- **Runtime**: Node.js with Express.js and TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **Storage**: Hybrid system with database persistence and in-memory caching
- **Python Services**: Direct database access via psycopg2 for robust data persistence
- **WebSocket**: Real-time communication for live updates

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v20 or higher)
- **Python** (3.11 or higher)
- **PostgreSQL** (12 or higher)
- **npm** or **yarn** package manager

### 1. Environment Setup

Create a `.env` file in the root directory:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/marl_system
PGHOST=localhost
PGPORT=5432
PGUSER=your_username
PGPASSWORD=your_password
PGDATABASE=marl_system

# Optional: Session Configuration
SESSION_SECRET=your-super-secret-session-key
```

### 2. Database Setup

Create a PostgreSQL database:

```sql
-- Connect to PostgreSQL and create database
CREATE DATABASE marl_system;
CREATE USER marl_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE marl_system TO marl_user;
```

### 3. Install Dependencies

#### Node.js Dependencies
```bash
# Install all Node.js packages
npm install
```

#### Python Dependencies
The system uses Python for advanced MARL algorithms. Install the core dependencies:

```bash
# Core database dependencies (required)
pip install psycopg2-binary==2.9.9
pip install SQLAlchemy==2.0.23
pip install alembic==1.13.1
pip install flask==3.0.0

# Optional: Machine Learning libraries (may have compatibility issues in some environments)
# Uncomment only if your environment supports them:
# pip install torch==2.1.1
# pip install numpy==1.26.2
# pip install ray[rllib]==2.8.1
# pip install pettingzoo==1.24.3
```

**Note**: The system is designed to work without heavy ML dependencies and will gracefully fallback to simplified algorithms if packages like NumPy or Ray are unavailable.

### 4. Database Schema Migration

Push the database schema:

```bash
npm run db:push
```

### 5. Start the Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:5000`

## 📱 Usage

### Basic Operations

1. **Initialize Demo Data**: Click "Initialize Demo" to create sample agents
2. **Start Training**: Use the training controls to begin MARL experiments
3. **Monitor Progress**: Watch real-time metrics and visualizations
4. **View Memory**: Explore shared agent memory and communication patterns

### Advanced Features

#### Persistent Training
Access database-backed training experiments:

```bash
# Health check
curl -X GET "http://localhost:5000/api/training/persistent/health"

# Start persistent training
curl -X POST "http://localhost:5000/api/training/persistent/start" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Long-term MARL Experiment",
    "description": "Robust training with database persistence",
    "total_episodes": 1000,
    "checkpoint_interval": 50
  }'

# View experiment history
curl -X GET "http://localhost:5000/api/training/persistent/experiments"
```

#### Communication Analysis
Monitor emergent communication patterns:

```bash
# Get communication patterns
curl -X GET "http://localhost:5000/api/communication-patterns"

# Analyze memory usage
curl -X GET "http://localhost:5000/api/memory"
```

## 🔧 Configuration

### Experiment Parameters

Configure training experiments via the API or modify the default settings:

```typescript
interface ExperimentConfig {
  name: string;
  description: string;
  total_episodes: number;
  max_steps_per_episode: number;
  learning_rate: number;
  batch_size: number;
  checkpoint_interval: number;
  breakthrough_threshold: number;
}
```

### Agent Grid Settings

Customize the 3D agent grid in `client/src/lib/grid.ts`:

```typescript
export const GRID_CONFIG = {
  dimensions: { x: 4, y: 3, z: 3 }, // 4×3×3 grid
  agentCount: 30,
  coordinatorCount: 3,
  communicationRange: 2
};
```

## 🧠 Core Components

### 1. Agent Types
- **Regular Agents**: Handle local tasks and peer communication
- **Coordinator Agents**: Manage regional oversight and inter-coordinator communication

### 2. Communication Protocol
- **Message Types**: pointer, broadcast, breakthrough, coordination, heartbeat
- **Hierarchical Routing**: Multi-level message distribution
- **Emergent Patterns**: System learns efficient communication strategies

### 3. Memory System
- **Vectorized Storage**: Efficient shared information lookup
- **Persistence Levels**: Critical, high, medium, low priority data
- **Compression**: Automatic vector compression for storage optimization

### 4. Training Orchestration
- **Episode Management**: Coordinated training sessions
- **Metrics Collection**: Comprehensive performance tracking
- **Checkpoint System**: Automatic saving at configured intervals

## 🐛 Troubleshooting

### Common Issues

#### Python Import Errors
If you encounter NumPy or other ML library import issues:

1. The system is designed to work without heavy ML dependencies
2. It will automatically use simplified algorithms as fallbacks
3. For full ML features, ensure proper environment setup

#### Database Connection Issues
1. Verify PostgreSQL is running: `systemctl status postgresql`
2. Check connection string in `.env` file
3. Ensure database exists and user has proper permissions

#### Port Conflicts
1. Default port is 5000
2. Change port in `server/index.ts` if needed
3. Update client connections accordingly

### Development Mode

For development with hot reloading:

```bash
# Start with development logging
NODE_ENV=development npm run dev

# Check database status
curl -X GET "http://localhost:5000/api/training/persistent/health"
```

## 📊 Performance Monitoring

The system provides comprehensive metrics:

- **Training Progress**: Episode rewards, convergence rates
- **Communication Efficiency**: Message routing success rates
- **Memory Utilization**: Vector storage and retrieval performance
- **Breakthrough Detection**: Novel strategy identification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with modern web technologies and AI/ML best practices
- Inspired by biological neural networks and swarm intelligence
- Designed for research and educational purposes in multi-agent systems

## 📚 Additional Resources

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [PettingZoo Multi-Agent Environments](https://pettingzoo.farama.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [React + TypeScript Guide](https://react-typescript-cheatsheet.netlify.app/)

---

**Note**: This system is designed to be robust and work in various environments, including cloud platforms like Replit, with automatic fallbacks for dependencies that may not be available.