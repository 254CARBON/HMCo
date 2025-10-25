"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.env = void 0;
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
function parsePort(value, fallback) {
    if (!value) {
        return fallback;
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? fallback : parsed;
}
function resolveClusterHost() {
    const namespace = process.env.POSTGRES_NAMESPACE ??
        process.env.K8S_NAMESPACE ??
        process.env.NAMESPACE ??
        process.env.POD_NAMESPACE;
    if (namespace && namespace.trim().length > 0) {
        return `postgres-shared-service.${namespace}.svc.cluster.local`;
    }
    return 'postgres-shared-service';
}
function buildDatabaseUrlFromParts() {
    const host = process.env.POSTGRES_HOST ?? process.env.PGHOST ?? resolveClusterHost();
    const port = parsePort(process.env.POSTGRES_PORT ?? process.env.PGPORT, 5432);
    const database = process.env.POSTGRES_DB ?? process.env.PGDATABASE ?? 'datahub';
    const user = process.env.POSTGRES_USER ?? process.env.PGUSER ?? 'datahub';
    const password = process.env.POSTGRES_PASSWORD ?? process.env.PGPASSWORD ?? '';
    if (!password) {
        return undefined;
    }
    const encodedUser = encodeURIComponent(user);
    const encodedPassword = encodeURIComponent(password);
    return `postgresql://${encodedUser}:${encodedPassword}@${host}:${port}/${database}`;
}
exports.env = {
    nodeEnv: process.env.NODE_ENV ?? 'development',
    port: parsePort(process.env.PORT, 3001),
    databaseUrl: process.env.DATABASE_URL ?? buildDatabaseUrlFromParts() ?? '',
};
if (!exports.env.databaseUrl) {
    console.warn('[env] Database connection not configured. Set DATABASE_URL or POSTGRES_* variables (POSTGRES_PASSWORD required).');
}
//# sourceMappingURL=env.js.map