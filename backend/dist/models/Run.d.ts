export interface RunRow {
    id: string;
    provider_id: string;
    status: string;
    started_at: Date;
    completed_at: Date | null;
    records_ingested: number;
    records_failed: number;
    duration: number | null;
    logs: string | null;
    error_message: string | null;
    parameters: Record<string, unknown>;
    created_at: Date;
    provider_name?: string;
}
//# sourceMappingURL=Run.d.ts.map