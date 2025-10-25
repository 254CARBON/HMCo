export interface ProviderRow {
    id: string;
    name: string;
    type: string;
    status: string;
    uis: string;
    config: Record<string, unknown>;
    schedule: string | null;
    last_run_at: Date | null;
    next_run_at: Date | null;
    total_runs: number;
    success_rate: number;
    created_at: Date;
    updated_at: Date;
}
//# sourceMappingURL=Provider.d.ts.map