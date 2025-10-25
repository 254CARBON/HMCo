"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runBasicMigrations = runBasicMigrations;
const fs_1 = require("fs");
const path_1 = __importDefault(require("path"));
async function runBasicMigrations(db, migrationsDir) {
    const files = await fs_1.promises.readdir(migrationsDir);
    const sorted = files
        .filter((f) => f.endsWith('.sql'))
        .sort();
    for (const file of sorted) {
        const full = path_1.default.join(migrationsDir, file);
        const sql = await fs_1.promises.readFile(full, 'utf8');
        // Execute whole file; statements include IF NOT EXISTS guards.
        await db.query(sql);
    }
}
//# sourceMappingURL=migrate.js.map