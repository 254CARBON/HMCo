"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunsService = void 0;
const sortableColumns = new Set(['created_at', 'started_at', 'completed_at', 'duration']);
class RunsService {
    constructor(db) {
        this.db = db;
    }
    async listRuns(providerId, status, sortBy = 'created_at', sortOrder = 'DESC', limit = 50) {
        const appliedLimit = Math.max(1, Math.min(limit, 100));
        const safeSortBy = sortableColumns.has(sortBy) ? sortBy : 'created_at';
        const safeSortOrder = sortOrder.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';
        const conditions = ['1=1'];
        const params = [];
        if (providerId) {
            params.push(providerId);
            conditions.push(`r.provider_id = $${params.length}`);
        }
        if (status) {
            params.push(status);
            conditions.push(`r.status = $${params.length}`);
        }
        params.push(appliedLimit);
        const limitIdx = params.length;
        const query = `
      SELECT r.*, p.name as provider_name
      FROM runs r
      JOIN providers p ON r.provider_id = p.id
      WHERE ${conditions.join(' AND ')}
      ORDER BY r.${safeSortBy} ${safeSortOrder}
      LIMIT $${limitIdx}
    `;
        const result = await this.db.query(query, params);
        return { runs: result.rows };
    }
    async createRun(providerId, parameters) {
        const result = await this.db.query(`INSERT INTO runs (provider_id, status, parameters)
       VALUES ($1, $2, $3)
       RETURNING *`, [providerId, 'running', JSON.stringify(parameters ?? {})]);
        const run = result.rows[0];
        if (!run) {
            throw new Error('Failed to create run');
        }
        return run;
    }
    async getRun(id) {
        const result = await this.db.query(`SELECT r.*, p.name as provider_name
       FROM runs r
       JOIN providers p ON r.provider_id = p.id
       WHERE r.id = $1`, [id]);
        const run = result.rows[0];
        if (!run) {
            throw new Error('Run not found');
        }
        return run;
    }
    async updateRun(id, status, data) {
        const result = await this.db.query(`UPDATE runs SET
         status = $1,
         records_ingested = COALESCE($2, records_ingested),
         records_failed = COALESCE($3, records_failed),
         duration = COALESCE($4, duration),
         logs = COALESCE($5, logs),
         error_message = COALESCE($6, error_message),
         completed_at = CASE WHEN $1 IN ('success', 'failed', 'cancelled') THEN NOW() ELSE completed_at END
       WHERE id = $7
       RETURNING *`, [
            status,
            data?.recordsIngested ?? null,
            data?.recordsFailed ?? null,
            data?.duration ?? null,
            data?.logs ?? null,
            data?.errorMessage ?? null,
            id,
        ]);
        const run = result.rows[0];
        if (!run) {
            throw new Error('Run not found');
        }
        return run;
    }
}
exports.RunsService = RunsService;
//# sourceMappingURL=RunsService.js.map