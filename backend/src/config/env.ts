import dotenv from 'dotenv';

dotenv.config();

function parsePort(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

export const env = {
  nodeEnv: process.env.NODE_ENV ?? 'development',
  port: parsePort(process.env.PORT, 4000),
  databaseUrl: process.env.DATABASE_URL ?? '',
};

if (!env.databaseUrl) {
  console.warn(
    '[env] DATABASE_URL not set. Set it to point to your PostgreSQL instance.'
  );
}
