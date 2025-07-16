import { MemoryEntry, MemoryPointer, MemoryUtilization, VectorData } from "@/types/memory";

export class SharedMemorySystem {
  private memoryStore: Map<string, MemoryEntry>;
  private pointers: Map<string, MemoryPointer>;
  private accessLog: Map<string, Date[]>;

  constructor() {
    this.memoryStore = new Map();
    this.pointers = new Map();
    this.accessLog = new Map();
  }

  public storeVector(
    key: string,
    vectorData: VectorData,
    experimentId: string
  ): MemoryEntry {
    const entry: MemoryEntry = {
      id: this.generateId(),
      experimentId,
      memoryKey: key,
      vectorData,
      accessCount: 0,
      lastAccessed: new Date(),
      createdAt: new Date()
    };

    this.memoryStore.set(key, entry);
    this.accessLog.set(key, []);
    
    return entry;
  }

  public retrieveVector(key: string): MemoryEntry | undefined {
    const entry = this.memoryStore.get(key);
    
    if (entry) {
      entry.accessCount++;
      entry.lastAccessed = new Date();
      
      const log = this.accessLog.get(key) || [];
      log.push(new Date());
      this.accessLog.set(key, log);
    }
    
    return entry;
  }

  public createPointer(
    key: string,
    coordinates: { x: number; y: number; z: number },
    type: 'breakthrough' | 'pattern' | 'context',
    significance: number
  ): MemoryPointer {
    const pointer: MemoryPointer = {
      key,
      coordinates,
      type,
      significance
    };

    this.pointers.set(key, pointer);
    return pointer;
  }

  public getPointer(key: string): MemoryPointer | undefined {
    return this.pointers.get(key);
  }

  public getAllPointers(): Map<string, MemoryPointer> {
    return this.pointers;
  }

  public getMemoryUtilization(): MemoryUtilization {
    const totalEntries = this.memoryStore.size;
    const activeEntries = Array.from(this.memoryStore.values()).filter(
      entry => entry.accessCount > 0
    ).length;

    const memoryUsage = Array.from(this.memoryStore.values()).reduce(
      (sum, entry) => sum + this.calculateMemorySize(entry.vectorData), 0
    );

    const accessPatterns = Array.from(this.accessLog.entries()).map(
      ([key, accesses]) => ({
        key,
        frequency: accesses.length,
        lastAccess: accesses[accesses.length - 1] || new Date(0)
      })
    );

    return {
      totalEntries,
      activeEntries,
      memoryUsage,
      accessPatterns
    };
  }

  public encodeBreakthrough(
    breakthroughData: any,
    context: Record<string, any>
  ): VectorData {
    // Simple encoding - in production, this would use a proper embedding model
    const embedding = this.generateEmbedding(breakthroughData);
    
    return {
      embedding,
      metadata: {
        type: 'breakthrough',
        context,
        timestamp: new Date(),
        significance: this.calculateSignificance(breakthroughData)
      },
      timestamp: new Date()
    };
  }

  public findSimilarVectors(
    queryVector: number[],
    threshold: number = 0.8
  ): MemoryEntry[] {
    const similar: MemoryEntry[] = [];

    for (const entry of this.memoryStore.values()) {
      const similarity = this.calculateSimilarity(
        queryVector,
        entry.vectorData.embedding
      );

      if (similarity >= threshold) {
        similar.push(entry);
      }
    }

    return similar.sort((a, b) => 
      this.calculateSimilarity(queryVector, b.vectorData.embedding) -
      this.calculateSimilarity(queryVector, a.vectorData.embedding)
    );
  }

  private generateEmbedding(data: any): number[] {
    // Simple hash-based embedding for demonstration
    const str = JSON.stringify(data);
    const embedding = new Array(128).fill(0);
    
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      embedding[i % 128] += char / 255;
    }

    return embedding.map(v => Math.tanh(v));
  }

  private calculateSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length) return 0;

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  private calculateSignificance(data: any): number {
    // Simple significance calculation
    const complexity = JSON.stringify(data).length;
    return Math.min(complexity / 1000, 1.0);
  }

  private calculateMemorySize(vectorData: VectorData): number {
    return vectorData.embedding.length * 8 + // 8 bytes per float
           JSON.stringify(vectorData.metadata).length;
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  public cleanup(maxAge: number = 3600000) { // 1 hour default
    const now = new Date();
    const cutoff = new Date(now.getTime() - maxAge);

    for (const [key, entry] of this.memoryStore.entries()) {
      if (entry.lastAccessed < cutoff && entry.accessCount === 0) {
        this.memoryStore.delete(key);
        this.accessLog.delete(key);
        this.pointers.delete(key);
      }
    }
  }
}
