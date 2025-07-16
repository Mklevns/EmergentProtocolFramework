import { Message, CommunicationProtocol, CommunicationFlow } from "@/types/communication";
import { AgentNetwork } from "@/types/agents";

export class CommunicationSystem {
  private messages: Map<string, Message>;
  private protocols: Map<string, CommunicationProtocol>;
  private flows: Map<string, CommunicationFlow>;
  private emergentPatterns: Map<string, any>;

  constructor() {
    this.messages = new Map();
    this.protocols = new Map();
    this.flows = new Map();
    this.emergentPatterns = new Map();
  }

  public sendMessage(
    fromAgentId: string,
    toAgentId: string,
    messageType: 'pointer' | 'broadcast' | 'direct',
    payload: any,
    memoryPointer?: string
  ): Message {
    const message: Message = {
      id: this.generateId(),
      fromAgentId,
      toAgentId,
      messageType,
      payload,
      memoryPointer,
      timestamp: new Date()
    };

    this.messages.set(message.id, message);
    this.updateCommunicationFlow(fromAgentId, toAgentId, messageType);
    
    return message;
  }

  public broadcastMessage(
    fromAgentId: string,
    recipients: string[],
    payload: any,
    memoryPointer?: string
  ): Message[] {
    const messages: Message[] = [];

    for (const recipient of recipients) {
      const message = this.sendMessage(
        fromAgentId,
        recipient,
        'broadcast',
        payload,
        memoryPointer
      );
      messages.push(message);
    }

    return messages;
  }

  public sendPointerMessage(
    fromAgentId: string,
    toAgentId: string,
    memoryPointer: string,
    context?: any
  ): Message {
    return this.sendMessage(
      fromAgentId,
      toAgentId,
      'pointer',
      { context },
      memoryPointer
    );
  }

  public getMessagesForAgent(agentId: string): Message[] {
    return Array.from(this.messages.values()).filter(
      msg => msg.toAgentId === agentId
    );
  }

  public getRecentMessages(timeWindow: number = 60000): Message[] {
    const cutoff = new Date(Date.now() - timeWindow);
    return Array.from(this.messages.values()).filter(
      msg => msg.timestamp > cutoff
    );
  }

  public analyzeCommunicationPatterns(network: AgentNetwork): void {
    const patterns = new Map<string, any>();

    // Analyze message frequency patterns
    const messageFrequency = new Map<string, number>();
    for (const message of this.messages.values()) {
      const key = `${message.fromAgentId}-${message.toAgentId}`;
      messageFrequency.set(key, (messageFrequency.get(key) || 0) + 1);
    }

    // Detect emergent communication protocols
    const protocolSignatures = new Map<string, number>();
    for (const message of this.messages.values()) {
      const signature = this.generateProtocolSignature(message);
      protocolSignatures.set(signature, (protocolSignatures.get(signature) || 0) + 1);
    }

    // Identify frequently used protocols
    for (const [signature, count] of protocolSignatures.entries()) {
      if (count > 10) { // Threshold for protocol emergence
        const protocol = this.createProtocol(signature, count);
        this.protocols.set(protocol.id, protocol);
      }
    }

    this.emergentPatterns = patterns;
  }

  public getCommunicationEfficiency(network: AgentNetwork): number {
    const totalMessages = this.messages.size;
    const successfulDeliveries = Array.from(this.messages.values()).filter(
      msg => this.isMessageDelivered(msg, network)
    ).length;

    const redundantMessages = this.calculateRedundancy();
    const pointerEfficiency = this.calculatePointerEfficiency();

    return totalMessages > 0 
      ? ((successfulDeliveries / totalMessages) * 0.5 + 
         (1 - redundantMessages / totalMessages) * 0.3 + 
         pointerEfficiency * 0.2)
      : 0;
  }

  public getNetworkTopology(network: AgentNetwork) {
    const nodes = Array.from(network.agents.values()).map(agent => ({
      id: agent.id,
      type: agent.type,
      position: agent.position,
      active: agent.isActive
    }));

    const edges: Array<{
      source: string;
      target: string;
      weight: number;
      type: string;
    }> = [];

    for (const [agentId, neighbors] of network.connections.entries()) {
      for (const neighborId of neighbors) {
        const flow = this.flows.get(`${agentId}-${neighborId}`);
        edges.push({
          source: agentId,
          target: neighborId,
          weight: flow?.frequency || 1,
          type: flow?.messageType || 'direct'
        });
      }
    }

    return { nodes, edges };
  }

  private updateCommunicationFlow(
    fromAgentId: string,
    toAgentId: string,
    messageType: string
  ) {
    const key = `${fromAgentId}-${toAgentId}`;
    const existing = this.flows.get(key);

    if (existing) {
      existing.frequency++;
      existing.latency = this.calculateLatency(fromAgentId, toAgentId);
    } else {
      this.flows.set(key, {
        source: fromAgentId,
        target: toAgentId,
        messageType,
        frequency: 1,
        latency: this.calculateLatency(fromAgentId, toAgentId),
        reliability: 1.0
      });
    }
  }

  private generateProtocolSignature(message: Message): string {
    return `${message.messageType}-${typeof message.payload}-${message.memoryPointer ? 'pointer' : 'direct'}`;
  }

  private createProtocol(signature: string, usage: number): CommunicationProtocol {
    return {
      id: this.generateId(),
      name: `Protocol-${signature}`,
      description: `Emergent protocol with signature: ${signature}`,
      efficiency: Math.min(usage / 100, 1.0),
      usage,
      emergenceTime: new Date()
    };
  }

  private isMessageDelivered(message: Message, network: AgentNetwork): boolean {
    const recipient = network.agents.get(message.toAgentId);
    return recipient ? recipient.isActive : false;
  }

  private calculateRedundancy(): number {
    const uniqueMessages = new Set();
    let redundant = 0;

    for (const message of this.messages.values()) {
      const key = `${message.fromAgentId}-${message.toAgentId}-${JSON.stringify(message.payload)}`;
      if (uniqueMessages.has(key)) {
        redundant++;
      } else {
        uniqueMessages.add(key);
      }
    }

    return redundant;
  }

  private calculatePointerEfficiency(): number {
    const pointerMessages = Array.from(this.messages.values()).filter(
      msg => msg.messageType === 'pointer'
    );

    const totalMessages = this.messages.size;
    return totalMessages > 0 ? pointerMessages.length / totalMessages : 0;
  }

  private calculateLatency(fromAgentId: string, toAgentId: string): number {
    // Simple distance-based latency calculation
    // In a real system, this would consider network topology
    return Math.random() * 10 + 5; // 5-15ms latency
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  public getProtocols(): Map<string, CommunicationProtocol> {
    return this.protocols;
  }

  public getCommunicationFlows(): Map<string, CommunicationFlow> {
    return this.flows;
  }

  public getEmergentPatterns(): Map<string, any> {
    return this.emergentPatterns;
  }
}
