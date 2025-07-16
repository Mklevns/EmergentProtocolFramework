export interface VectorData {
  embedding: number[];
  metadata: Record<string, any>;
  timestamp: Date;
}

export interface MemoryEntry {
  id: string;
  experimentId: string;
  memoryKey: string;
  vectorData: VectorData;
  accessCount: number;
  lastAccessed: Date;
  createdAt: Date;
}

export interface MemoryPointer {
  key: string;
  coordinates: { x: number; y: number; z: number };
  type: 'breakthrough' | 'pattern' | 'context';
  significance: number;
}

export interface MemoryUtilization {
  totalEntries: number;
  activeEntries: number;
  memoryUsage: number;
  accessPatterns: Array<{
    key: string;
    frequency: number;
    lastAccess: Date;
  }>;
}
