"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getDbClient = getDbClient;
exports.closeDb = closeDb;
const pg_1 = require("pg");
const env_1 = require("./env");
let dbClient = null;
async function getDbClient() {
    if (dbClient) {
        return dbClient;
    }
    if (!env_1.env.databaseUrl) {
        throw new Error('DATABASE_URL is not configured.');
    }
    const client = new pg_1.Client({
        connectionString: env_1.env.databaseUrl,
    });
    await client.connect();
    dbClient = client;
    return client;
}
async function closeDb() {
    if (!dbClient) {
        return;
    }
    await dbClient.end();
    dbClient = null;
}
//# sourceMappingURL=database.js.map