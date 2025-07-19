"""
Python Database Layer for Direct Database Access
Provides robust database persistence for Python services with connection pooling,
async support, and comprehensive error handling.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extras import RealDictCursor, Json
import threading
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        self.pghost = os.environ.get('PGHOST', 'localhost')
        self.pgport = int(os.environ.get('PGPORT', 5432))
        self.pguser = os.environ.get('PGUSER', 'postgres')
        self.pgpassword = os.environ.get('PGPASSWORD', '')
        self.pgdatabase = os.environ.get('PGDATABASE', 'postgres')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for psycopg2"""
        return {
            'host': self.pghost,
            'port': self.pgport,
            'user': self.pguser,
            'password': self.pgpassword,
            'database': self.pgdatabase,
        }

class DatabaseConnectionPool:
    """Thread-safe database connection pool"""
    
    def __init__(self, config: DatabaseConfig, min_connections: int = 2, max_connections: int = 10):
        self.config = config
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool = None
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            connection_params = self.config.get_connection_params()
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                **connection_params
            )
            logger.info(f"Database connection pool initialized with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            with self._lock:
                conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                with self._lock:
                    self._pool.putconn(conn)
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("All database connections closed")

@dataclass
class AgentData:
    """Agent data structure matching the schema"""
    agent_id: str
    type: str
    position_x: int
    position_y: int
    position_z: int
    status: str = "idle"
    coordinator_id: Optional[str] = None
    hidden_dim: int = 256
    is_active: bool = True
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class ExperimentData:
    """Experiment data structure"""
    name: str
    config: Dict[str, Any]
    description: Optional[str] = None
    status: str = "pending"
    metrics: Optional[Dict[str, Any]] = None
    id: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: Optional[datetime] = None

@dataclass
class MetricData:
    """Training metrics data structure"""
    experiment_id: int
    episode: int
    step: int
    metric_type: str
    value: float
    agent_id: Optional[str] = None
    id: Optional[int] = None
    timestamp: Optional[datetime] = None

@dataclass
class MemoryVectorData:
    """Memory vector data structure"""
    vector_id: str
    content: Dict[str, Any]
    vector_type: str
    coordinates: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None

@dataclass
class BreakthroughData:
    """Breakthrough data structure"""
    agent_id: str
    breakthrough_type: str
    description: Optional[str] = None
    vector_id: Optional[str] = None
    confidence: float = 0.0
    was_shared: bool = False
    id: Optional[int] = None
    timestamp: Optional[datetime] = None

@dataclass
class MessageData:
    """Message data structure"""
    from_agent_id: str
    to_agent_id: str
    message_type: str
    content: Dict[str, Any]
    memory_pointer: Optional[str] = None
    is_processed: bool = False
    id: Optional[int] = None
    timestamp: Optional[datetime] = None

class PythonDatabaseManager:
    """Main database manager for Python services"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.pool = DatabaseConnectionPool(self.config)
        logger.info("Python database manager initialized")
    
    def _convert_to_dict_row(self, data: Any) -> Optional[Dict[str, Any]]:
        """Convert database row to dictionary"""
        if data is None:
            return None
        if hasattr(data, '_asdict'):
            return data._asdict()
        return dict(data) if data else None
    
    def _serialize_for_db(self, data: Any) -> Any:
        """Serialize data for database storage"""
        if isinstance(data, dict):
            return Json(data)
        return data
    
    # Agent operations
    def create_agent(self, agent: AgentData) -> Optional[AgentData]:
        """Create a new agent"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO agents (agent_id, type, position_x, position_y, position_z,
                                          status, coordinator_id, hidden_dim, is_active)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                    """, (
                        agent.agent_id, agent.type, agent.position_x, agent.position_y,
                        agent.position_z, agent.status, agent.coordinator_id,
                        agent.hidden_dim, agent.is_active
                    ))
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return AgentData(**row)
                    return None
        except Exception as e:
            logger.error(f"Failed to create agent {agent.agent_id}: {e}")
            return None
    
    def get_agent(self, agent_id: str) -> Optional[AgentData]:
        """Get agent by ID"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM agents WHERE agent_id = %s", (agent_id,))
                    row = cur.fetchone()
                    return AgentData(**row) if row else None
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    def get_all_agents(self) -> List[AgentData]:
        """Get all agents"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM agents ORDER BY agent_id")
                    rows = cur.fetchall()
                    return [AgentData(**row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all agents: {e}")
            return []
    
    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE agents 
                        SET status = %s, updated_at = CURRENT_TIMESTAMP 
                        WHERE agent_id = %s
                    """, (status, agent_id))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id} status: {e}")
            return False
    
    # Experiment operations
    def create_experiment(self, experiment: ExperimentData) -> Optional[ExperimentData]:
        """Create a new experiment"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO experiments (name, description, config, status, metrics)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING *
                    """, (
                        experiment.name, experiment.description,
                        self._serialize_for_db(experiment.config),
                        experiment.status,
                        self._serialize_for_db(experiment.metrics)
                    ))
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return ExperimentData(**row)
                    return None
        except Exception as e:
            logger.error(f"Failed to create experiment {experiment.name}: {e}")
            return None
    
    def get_experiment(self, experiment_id: int) -> Optional[ExperimentData]:
        """Get experiment by ID"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM experiments WHERE id = %s", (experiment_id,))
                    row = cur.fetchone()
                    return ExperimentData(**row) if row else None
        except Exception as e:
            logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    def update_experiment_status(self, experiment_id: int, status: str) -> bool:
        """Update experiment status"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    now = datetime.now(timezone.utc)
                    if status == "running":
                        cur.execute("""
                            UPDATE experiments 
                            SET status = %s, start_time = %s 
                            WHERE id = %s
                        """, (status, now, experiment_id))
                    elif status in ["completed", "failed"]:
                        cur.execute("""
                            UPDATE experiments 
                            SET status = %s, end_time = %s 
                            WHERE id = %s
                        """, (status, now, experiment_id))
                    else:
                        cur.execute("""
                            UPDATE experiments 
                            SET status = %s 
                            WHERE id = %s
                        """, (status, experiment_id))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update experiment {experiment_id} status: {e}")
            return False
    
    def update_experiment_metrics(self, experiment_id: int, metrics: Dict[str, Any]) -> bool:
        """Update experiment metrics"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE experiments 
                        SET metrics = %s 
                        WHERE id = %s
                    """, (self._serialize_for_db(metrics), experiment_id))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update experiment {experiment_id} metrics: {e}")
            return False
    
    # Metrics operations
    def create_metric(self, metric: MetricData) -> Optional[MetricData]:
        """Create a training metric"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO metrics (experiment_id, episode, step, metric_type, value, agent_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING *
                    """, (
                        metric.experiment_id, metric.episode, metric.step,
                        metric.metric_type, metric.value, metric.agent_id
                    ))
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return MetricData(**row)
                    return None
        except Exception as e:
            logger.error(f"Failed to create metric: {e}")
            return None
    
    def get_metrics_by_experiment(self, experiment_id: int, limit: int = 100) -> List[MetricData]:
        """Get metrics for an experiment"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM metrics 
                        WHERE experiment_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (experiment_id, limit))
                    rows = cur.fetchall()
                    return [MetricData(**row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get metrics for experiment {experiment_id}: {e}")
            return []
    
    # Memory vector operations
    def create_memory_vector(self, vector: MemoryVectorData) -> Optional[MemoryVectorData]:
        """Create a memory vector"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO memory_vectors (vector_id, content, vector_type, coordinates, importance)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING *
                    """, (
                        vector.vector_id, self._serialize_for_db(vector.content),
                        vector.vector_type, vector.coordinates, vector.importance
                    ))
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return MemoryVectorData(**row)
                    return None
        except Exception as e:
            logger.error(f"Failed to create memory vector {vector.vector_id}: {e}")
            return None
    
    def get_memory_vector(self, vector_id: str) -> Optional[MemoryVectorData]:
        """Get memory vector by ID"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM memory_vectors WHERE vector_id = %s", (vector_id,))
                    row = cur.fetchone()
                    return MemoryVectorData(**row) if row else None
        except Exception as e:
            logger.error(f"Failed to get memory vector {vector_id}: {e}")
            return None
    
    def update_memory_access(self, vector_id: str) -> bool:
        """Update memory vector access count and timestamp"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE memory_vectors 
                        SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP 
                        WHERE vector_id = %s
                    """, (vector_id,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update memory vector {vector_id} access: {e}")
            return False
    
    # Breakthrough operations
    def create_breakthrough(self, breakthrough: BreakthroughData) -> Optional[BreakthroughData]:
        """Create a breakthrough event"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO breakthroughs (agent_id, breakthrough_type, description, vector_id, confidence, was_shared)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING *
                    """, (
                        breakthrough.agent_id, breakthrough.breakthrough_type,
                        breakthrough.description, breakthrough.vector_id,
                        breakthrough.confidence, breakthrough.was_shared
                    ))
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return BreakthroughData(**row)
                    return None
        except Exception as e:
            logger.error(f"Failed to create breakthrough: {e}")
            return None
    
    def get_breakthroughs_by_agent(self, agent_id: str, limit: int = 20) -> List[BreakthroughData]:
        """Get breakthroughs for an agent"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM breakthroughs 
                        WHERE agent_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (agent_id, limit))
                    rows = cur.fetchall()
                    return [BreakthroughData(**row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get breakthroughs for agent {agent_id}: {e}")
            return []
    
    # Message operations
    def create_message(self, message: MessageData) -> Optional[MessageData]:
        """Create a message"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        INSERT INTO messages (from_agent_id, to_agent_id, message_type, content, memory_pointer)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING *
                    """, (
                        message.from_agent_id, message.to_agent_id, message.message_type,
                        self._serialize_for_db(message.content), message.memory_pointer
                    ))
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return MessageData(**row)
                    return None
        except Exception as e:
            logger.error(f"Failed to create message: {e}")
            return None
    
    def get_unprocessed_messages(self, limit: int = 100) -> List[MessageData]:
        """Get unprocessed messages"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM messages 
                        WHERE is_processed = FALSE 
                        ORDER BY timestamp ASC 
                        LIMIT %s
                    """, (limit,))
                    rows = cur.fetchall()
                    return [MessageData(**row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get unprocessed messages: {e}")
            return []
    
    def mark_message_processed(self, message_id: int) -> bool:
        """Mark message as processed"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE messages 
                        SET is_processed = TRUE 
                        WHERE id = %s
                    """, (message_id,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to mark message {message_id} as processed: {e}")
            return False
    
    # Utility methods
    def execute_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """Execute a custom query and return results"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
    
    def execute_update(self, query: str, params: Tuple = None) -> int:
        """Execute an update/insert/delete query and return affected rows"""
        try:
            with self.pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    conn.commit()
                    return cur.rowcount
        except Exception as e:
            logger.error(f"Failed to execute update: {e}")
            return 0
    
    def close(self):
        """Close database connections"""
        self.pool.close_all_connections()

# Singleton instance for easy access
_db_manager = None

def get_db_manager() -> PythonDatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = PythonDatabaseManager()
    return _db_manager

def close_db():
    """Close database connections"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None