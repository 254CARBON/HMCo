import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

const UISSpecSchema = z.object({
  name: z.string().min(1),
  description: z.string(),
  source: z.object({
    type: z.enum(['api', 'database', 's3', 'websocket', 'kafka']),
    config: z.record(z.any())
  }),
  target: z.object({
    type: z.enum(['iceberg', 'clickhouse']),
    database: z.string(),
    table: z.string()
  }),
  schedule: z.string().optional(),
  data_quality: z.object({
    suite: z.string(),
    enabled: z.boolean()
  }).optional()
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const result = UISSpecSchema.safeParse(body);
    
    if (!result.success) {
      return NextResponse.json(
        { error: 'Validation failed', details: result.error.errors },
        { status: 400 }
      );
    }
    
    return NextResponse.json({ 
      valid: true,
      spec: result.data
    });
    
  } catch (error) {
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
