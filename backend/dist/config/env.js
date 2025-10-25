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
exports.env = {
    nodeEnv: process.env.NODE_ENV ?? 'development',
    port: parsePort(process.env.PORT, 4000),
    databaseUrl: process.env.DATABASE_URL ?? '',
};
if (!exports.env.databaseUrl) {
    console.warn('[env] DATABASE_URL not set. Set it to point to your PostgreSQL instance.');
}
//# sourceMappingURL=env.js.map