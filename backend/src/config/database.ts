import { Client } from 'pg';
import { env } from './env';

let dbClient: Client | null = null;

export async function getDbClient(): Promise<Client> {
  if (dbClient) {
    return dbClient;
  }

  if (!env.databaseUrl) {
    throw new Error('DATABASE_URL is not configured.');
  }

  const client = new Client({
    connectionString: env.databaseUrl,
  });

  await client.connect();
  dbClient = client;
  return client;
}

export async function closeDb(): Promise<void> {
  if (!dbClient) {
    return;
  }
  await dbClient.end();
  dbClient = null;
}

