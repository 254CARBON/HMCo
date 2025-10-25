"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Runner = void 0;
const crypto_1 = __importDefault(require("crypto"));
class Runner {
    constructor(jobConfig, options) {
        this.jobConfig = jobConfig;
        this.options = options ?? {};
    }
    async testConnection() {
        const started = Date.now();
        // Simulate lightweight validation work
        await Promise.resolve();
        const latencyMs = Math.min(750, Math.max(50, this.jobConfig.name.length * 25));
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
    async execute() {
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
    estimateDuration() {
        const base = this.jobConfig.name.length * 42;
        const runtimeWeight = this.jobConfig.runtime === 'spark'
            ? 1.2
            : this.jobConfig.runtime === 'flink'
                ? 1.1
                : 1;
        const estimated = Math.round(base * runtimeWeight);
        return Math.max(500, Math.min(estimated, this.options.timeoutMs ?? estimated));
    }
    estimateRecordsIngested() {
        const base = this.jobConfig.name.length * 37;
        return Math.max(10, base);
    }
    buildLogs(durationMs, recordsIngested) {
        return [
            `[runner] Starting ${this.jobConfig.runtime} job "${this.jobConfig.name}"`,
            `[runner] Compiled job fingerprint: ${this.buildFingerprint()}`,
            `[runner] Estimated duration ${durationMs} ms`,
            `[runner] Ingested ${recordsIngested} records`,
            '[runner] Job completed successfully',
        ];
    }
    buildFingerprint() {
        const hash = crypto_1.default.createHash('sha1');
        hash.update(this.jobConfig.name);
        hash.update(this.jobConfig.runtime);
        hash.update(JSON.stringify(this.jobConfig.spec));
        return hash.digest('hex');
    }
}
exports.Runner = Runner;
//# sourceMappingURL=index.js.map