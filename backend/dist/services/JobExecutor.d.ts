import { JobConfig, Runner, RunnerExecutionResult, RunnerTestResult, SupportedRuntime } from '../sdk/runner';
type UISRuntime = SupportedRuntime;
export interface UISSpec extends Record<string, unknown> {
    name: string;
    runtime: UISRuntime;
    schedule?: {
        cron?: string;
        timezone?: string;
    };
}
export interface UISValidationResult {
    valid: boolean;
    errors: string[];
    spec?: UISSpec;
    normalized?: string;
    runtime?: UISRuntime;
}
export interface JobExecutionSummary extends RunnerExecutionResult {
    jobConfig: JobConfig;
}
type RunnerFactory = (config: JobConfig) => Runner;
export declare class JobExecutor {
    private readonly createRunner;
    constructor(createRunner?: RunnerFactory);
    validateUIS(uis: string): UISValidationResult;
    testConnection(uis: string): Promise<RunnerTestResult>;
    executeRun(providerId: string, uis: string): Promise<JobExecutionSummary>;
    private compile;
    private selectCompiler;
    private sparkCompiler;
    private flinkCompiler;
    private seatunnelCompiler;
    private parseSpec;
    private ensureUISSpec;
}
export declare function validateCronExpression(value: string): boolean;
export {};
//# sourceMappingURL=JobExecutor.d.ts.map