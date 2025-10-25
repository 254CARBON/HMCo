"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.JobExecutor = void 0;
exports.validateCronExpression = validateCronExpression;
const js_yaml_1 = __importDefault(require("js-yaml"));
const runner_1 = require("../sdk/runner");
const SUPPORTED_RUNTIMES = ['spark', 'flink', 'seatunnel'];
class JobExecutor {
    constructor(createRunner = (config) => new runner_1.Runner(config)) {
        this.createRunner = createRunner;
    }
    validateUIS(uis) {
        const errors = [];
        if (!uis || !uis.trim()) {
            return {
                valid: false,
                errors: ['UIS specification cannot be empty'],
            };
        }
        let spec;
        try {
            spec = this.parseSpec(uis);
        }
        catch (error) {
            const err = error;
            errors.push(err.message);
            return {
                valid: false,
                errors,
            };
        }
        if (!spec) {
            return {
                valid: false,
                errors: ['UIS specification parsed to an empty object'],
            };
        }
        if (!spec.name || typeof spec.name !== 'string') {
            errors.push('UIS specification must include a "name" field.');
        }
        const runtime = typeof spec.runtime === 'string' ? spec.runtime.toLowerCase() : undefined;
        if (!runtime) {
            errors.push('UIS specification must include a "runtime" field.');
        }
        else if (!SUPPORTED_RUNTIMES.includes(runtime)) {
            errors.push(`Unsupported runtime "${spec.runtime}". Supported runtimes: ${SUPPORTED_RUNTIMES.join(', ')}`);
        }
        else {
            spec.runtime = runtime;
        }
        const hasProviderBlock = typeof spec.provider === 'object' && spec.provider !== null;
        const hasSourceBlock = typeof spec.source === 'object' && spec.source !== null;
        if (!hasProviderBlock && !hasSourceBlock) {
            errors.push('UIS specification must include either a "provider" or "source" definition.');
        }
        const normalized = spec ? JSON.stringify(spec, null, 2) : undefined;
        const result = {
            valid: errors.length === 0,
            errors,
            spec,
            runtime: spec?.runtime,
        };
        if (normalized !== undefined) {
            result.normalized = normalized;
        }
        return result;
    }
    async testConnection(uis) {
        const validation = this.validateUIS(uis);
        if (!validation.valid || !validation.spec) {
            throw new Error(`UIS validation failed: ${validation.errors.join('; ')}`);
        }
        const jobConfig = this.compile(validation.spec);
        const runner = this.createRunner(jobConfig);
        return runner.testConnection();
    }
    async executeRun(providerId, uis) {
        const validation = this.validateUIS(uis);
        if (!validation.valid || !validation.spec) {
            throw new Error(`UIS validation failed: ${validation.errors.join('; ')}`);
        }
        const jobConfig = this.compile(validation.spec, { providerId });
        const runner = this.createRunner(jobConfig);
        const result = await runner.execute();
        return {
            ...result,
            jobConfig,
        };
    }
    compile(spec, metadata = {}) {
        const compiler = this.selectCompiler(spec.runtime);
        return compiler(spec, metadata);
    }
    selectCompiler(runtime) {
        switch (runtime) {
            case 'spark':
                return this.sparkCompiler;
            case 'flink':
                return this.flinkCompiler;
            case 'seatunnel':
                return this.seatunnelCompiler;
            default:
                throw new Error(`Unsupported runtime "${runtime}"`);
        }
    }
    sparkCompiler(spec, metadata) {
        return {
            name: spec.name,
            runtime: 'spark',
            spec,
            metadata: {
                engine: 'spark',
                ...metadata,
            },
        };
    }
    flinkCompiler(spec, metadata) {
        return {
            name: spec.name,
            runtime: 'flink',
            spec,
            metadata: {
                engine: 'flink',
                ...metadata,
            },
        };
    }
    seatunnelCompiler(spec, metadata) {
        return {
            name: spec.name,
            runtime: 'seatunnel',
            spec,
            metadata: {
                engine: 'seatunnel',
                ...metadata,
            },
        };
    }
    parseSpec(uis) {
        const source = uis.trim();
        const parseErrors = [];
        if (source.startsWith('{') || source.startsWith('[')) {
            try {
                const parsed = JSON.parse(source);
                return this.ensureUISSpec(parsed);
            }
            catch (error) {
                const err = error;
                parseErrors.push(`JSON parse error: ${err.message}`);
            }
        }
        try {
            const parsed = js_yaml_1.default.load(source, {
                schema: js_yaml_1.default.DEFAULT_SCHEMA,
            });
            return this.ensureUISSpec(parsed);
        }
        catch (error) {
            const err = error;
            parseErrors.push(`YAML parse error: ${err.message}`);
        }
        throw new Error(parseErrors.join('; '));
    }
    ensureUISSpec(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)) {
            throw new Error('UIS specification must parse to an object');
        }
        const spec = { ...value };
        if (typeof spec.runtime === 'string') {
            spec.runtime = spec.runtime.toLowerCase();
        }
        return spec;
    }
}
exports.JobExecutor = JobExecutor;
const CRON_SEGMENT = /^[\d*/?,#\-LWA]+$/i;
const CRON_MACROS = new Set([
    '@yearly',
    '@annually',
    '@monthly',
    '@weekly',
    '@daily',
    '@midnight',
    '@hourly',
]);
function validateCronExpression(value) {
    if (!value) {
        return true;
    }
    const trimmed = value.trim();
    if (CRON_MACROS.has(trimmed.toLowerCase())) {
        return true;
    }
    const segments = trimmed.split(/\s+/);
    if (segments.length < 5 || segments.length > 6) {
        return false;
    }
    return segments.every((segment) => CRON_SEGMENT.test(segment));
}
//# sourceMappingURL=JobExecutor.js.map