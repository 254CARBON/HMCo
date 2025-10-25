import crypto from 'crypto';

export type SupportedRuntime = 'spark' | 'flink' | 'seatunnel';

export interface JobConfig {
  name: string;
  runtime: SupportedRuntime;
  spec: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface RunnerOptions {
  timeoutMs?: number;
}

export interface RunnerTestResult {
  success: boolean;
  latencyMs: number;
  message: string;
  details?: Record<string, unknown>;
}

export interface RunnerExecutionResult {
  status: 'success' | 'failed';
  startedAt: Date;
  completedAt: Date;
  durationMs: number;
  recordsIngested: number;
  recordsFailed: number;
  logs: string[];
  metadata: Record<string, unknown>;
  errorMessage?: string;
}

export class Runner {
  private readonly options: RunnerOptions;

  constructor(
    private readonly jobConfig: JobConfig,
    options?: RunnerOptions
  ) {
    this.options = options ?? {};
  }

  async testConnection(): Promise<RunnerTestResult> {
    const started = Date.now();

    // Simulate lightweight validation work
    await Promise.resolve();

    const latencyMs = Math.min(
      750,
      Math.max(50, this.jobConfig.name.length * 25)
    );

    return {
      success: true,
      latencyMs,
      message: `Connection validated for ${this.jobConfig.runtime.toUpperCase()} runner`,
      details: {
        engine: this.jobConfig.runtime,
        jobFingerprint: this.buildFingerprint(),
      },
    };
  }

  async execute(): Promise<RunnerExecutionResult> {
    const startedAt = new Date();

    // Simulate async execution without blocking the event loop.
    await Promise.resolve();

    const durationMs = this.estimateDuration();
    const completedAt = new Date(startedAt.getTime() + durationMs);
    const recordsIngested = this.estimateRecordsIngested();

    return {
      status: 'success',
      startedAt,
      completedAt,
      durationMs,
      recordsIngested,
      recordsFailed: 0,
      logs: this.buildLogs(durationMs, recordsIngested),
      metadata: {
        engine: this.jobConfig.runtime,
        timeoutMs: this.options.timeoutMs ?? null,
        jobFingerprint: this.buildFingerprint(),
      },
    };
  }

  private estimateDuration(): number {
    const base = this.jobConfig.name.length * 42;
    const runtimeWeight =
      this.jobConfig.runtime === 'spark'
        ? 1.2
        : this.jobConfig.runtime === 'flink'
        ? 1.1
        : 1;
    const estimated = Math.round(base * runtimeWeight);
    return Math.max(500, Math.min(estimated, this.options.timeoutMs ?? estimated));
  }

  private estimateRecordsIngested(): number {
    const base = this.jobConfig.name.length * 37;
    return Math.max(10, base);
  }

  private buildLogs(durationMs: number, recordsIngested: number): string[] {
    return [
      `[runner] Starting ${this.jobConfig.runtime} job "${this.jobConfig.name}"`,
      `[runner] Compiled job fingerprint: ${this.buildFingerprint()}`,
      `[runner] Estimated duration ${durationMs} ms`,
      `[runner] Ingested ${recordsIngested} records`,
      '[runner] Job completed successfully',
    ];
  }

  private buildFingerprint(): string {
    const hash = crypto.createHash('sha1');
    hash.update(this.jobConfig.name);
    hash.update(this.jobConfig.runtime);
    hash.update(JSON.stringify(this.jobConfig.spec));
    return hash.digest('hex');
  }
}
