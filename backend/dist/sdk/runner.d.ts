export type SupportedRuntime = 'spark' | 'flink' | 'seatunnel';
export interface JobConfig {
    name: string;
    runtime: SupportedRuntime;
    spec: Record<string, unknown>;
    metadata?: Record<string, unknown>;
}
export interface RunnerExecutionResult {
    status: 'success' | 'failed';
    recordsIngested: number;
    recordsFailed: number;
    durationMs: number;
    logs: string[];
    errorMessage?: string;
}
export interface RunnerTestResult {
    ok: boolean;
    details?: string;
}
export declare class Runner {
    private readonly config;
    constructor(config: JobConfig);
    execute(): Promise<RunnerExecutionResult>;
    testConnection(): Promise<RunnerTestResult>;
    private spawnCollect;
}
//# sourceMappingURL=runner.d.ts.map