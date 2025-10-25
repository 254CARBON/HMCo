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
export declare class Runner {
    private readonly jobConfig;
    private readonly options;
    constructor(jobConfig: JobConfig, options?: RunnerOptions);
    testConnection(): Promise<RunnerTestResult>;
    execute(): Promise<RunnerExecutionResult>;
    private estimateDuration;
    private estimateRecordsIngested;
    private buildLogs;
    private buildFingerprint;
}
//# sourceMappingURL=index.d.ts.map