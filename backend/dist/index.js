"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cors_1 = __importDefault(require("cors"));
const morgan_1 = __importDefault(require("morgan"));
const express_1 = __importDefault(require("express"));
const env_1 = require("./config/env");
const database_1 = require("./config/database");
const path_1 = __importDefault(require("path"));
const migrate_1 = require("./config/migrate");
const providers_1 = require("./routes/providers");
const runs_1 = require("./routes/runs");
const JobExecutor_1 = require("./services/JobExecutor");
const ProvidersService_1 = require("./services/ProvidersService");
const RunsService_1 = require("./services/RunsService");
async function bootstrap() {
    const dbClient = await (0, database_1.getDbClient)();
    if ((process.env.RUN_MIGRATIONS ?? 'true').toLowerCase() === 'true') {
        try {
            const migrationsDir = path_1.default.resolve(__dirname, '../migrations');
            await (0, migrate_1.runBasicMigrations)(dbClient, migrationsDir);
            console.log('Migrations applied (idempotent).');
        }
        catch (mErr) {
            console.warn('Migration step failed (continuing):', mErr.message);
        }
    }
    const jobExecutor = new JobExecutor_1.JobExecutor();
    const providersService = new ProvidersService_1.ProvidersService(dbClient);
    const runsService = new RunsService_1.RunsService(dbClient, jobExecutor);
    const providersRouter = (0, providers_1.createProvidersRouter)(providersService, jobExecutor);
    const runsRouter = (0, runs_1.createRunsRouter)(runsService);
    const app = (0, express_1.default)();
    app.use((0, cors_1.default)());
    app.use(express_1.default.json());
    app.use((0, morgan_1.default)('combined'));
    const { register, metricsMiddleware } = await Promise.resolve().then(() => __importStar(require('./middleware/metrics')));
    app.use(metricsMiddleware);
    app.get('/healthz', (_req, res) => {
        res.json({ status: 'ok' });
    });
    app.get('/metrics', async (_req, res) => {
        const { register } = await Promise.resolve().then(() => __importStar(require('./middleware/metrics')));
        res.set('Content-Type', register.contentType);
        res.end(await register.metrics());
    });
    // Mount core routes
    app.use('/providers', providersRouter);
    app.use('/runs', runsRouter);
    // API-compatible aliases for the portal proxy (expects /api/*)
    app.use('/api/providers', providersRouter);
    app.use('/api/runs', runsRouter);
    app.use((_req, res) => {
        res.status(404).json({ error: 'Not found' });
    });
    app.listen(env_1.env.port, () => {
        console.log(`Backend API listening on port ${env_1.env.port}`);
    });
}
bootstrap().catch((error) => {
    console.error('Failed to start backend API', error);
    process.exit(1);
});
//# sourceMappingURL=index.js.map