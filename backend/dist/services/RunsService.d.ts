import { Client } from 'pg';
import { RunRow } from '../models/Run';
import { JobExecutor } from './JobExecutor';
export declare class RunsService {
    private readonly db;
    private readonly jobExecutor;
    constructor(db: Client, jobExecutor: JobExecutor);
    listRuns(providerId?: string, status?: string, sortBy?: string, sortOrder?: string, limit?: number): Promise<{
        runs: RunRow[];
    }>;
    createRun(providerId: string, parameters?: Record<string, unknown>): Promise<RunRow>;
    getRun(id: string): Promise<RunRow>;
    updateRun(id: string, status: string, data?: {
        recordsIngested?: number;
        recordsFailed?: number;
        duration?: number;
        logs?: string;
        errorMessage?: string;
    }): Promise<RunRow>;
    private getProvider;
    private updateProviderAfterRun;
}
//# sourceMappingURL=RunsService.d.ts.map