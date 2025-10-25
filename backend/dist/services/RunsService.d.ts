import { Client } from 'pg';
import { RunRow } from '../models/Run';
export declare class RunsService {
    private readonly db;
    constructor(db: Client);
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
}
//# sourceMappingURL=RunsService.d.ts.map