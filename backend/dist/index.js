"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cors_1 = __importDefault(require("cors"));
const express_1 = __importDefault(require("express"));
const env_1 = require("./config/env");
const database_1 = require("./config/database");
const providers_1 = require("./routes/providers");
const runs_1 = require("./routes/runs");
const ProvidersService_1 = require("./services/ProvidersService");
const RunsService_1 = require("./services/RunsService");
async function bootstrap() {
    const dbClient = await (0, database_1.getDbClient)();
    const providersService = new ProvidersService_1.ProvidersService(dbClient);
    const runsService = new RunsService_1.RunsService(dbClient);
    const app = (0, express_1.default)();
    app.use((0, cors_1.default)());
    app.use(express_1.default.json());
    app.get('/healthz', (_req, res) => {
        res.json({ status: 'ok' });
    });
    app.use('/providers', (0, providers_1.createProvidersRouter)(providersService));
    app.use('/runs', (0, runs_1.createRunsRouter)(runsService));
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