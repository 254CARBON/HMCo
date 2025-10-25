import { promises as fs } from 'fs';
import path from 'path';
import { Client } from 'pg';

export async function runBasicMigrations(db: Client, migrationsDir: string): Promise<void> {
  const files = await fs.readdir(migrationsDir);
  const sorted = files
    .filter((f) => f.endsWith('.sql'))
    .sort();

  for (const file of sorted) {
    const full = path.join(migrationsDir, file);
    const sql = await fs.readFile(full, 'utf8');
    // Execute whole file; statements include IF NOT EXISTS guards.
    await db.query(sql);
  }
}

