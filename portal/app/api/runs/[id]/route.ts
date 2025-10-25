import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.API_URL || 'http://localhost:3001';

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const response = await fetch(`${API_BASE}/api/runs/${params.id}`, {
      headers: {
        'Authorization': req.headers.get('authorization') || '',
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Run not found' },
        { status: response.status }
      );
    }

    const r = await response.json();
    const run = {
      id: r.id,
      providerId: r.provider_id,
      providerName: r.provider_name,
      status: r.status,
      startedAt: r.started_at,
      completedAt: r.completed_at,
      recordsIngested: r.records_ingested ?? 0,
      recordsFailed: r.records_failed ?? 0,
      duration: r.duration ?? 0,
      logs: typeof r.logs === 'string' ? r.logs.split('\n') : Array.isArray(r.logs) ? r.logs : [],
      errorMessage: r.error_message,
      parameters: r.parameters,
      createdAt: r.created_at,
    };
    return NextResponse.json(run);
  } catch (error) {
    console.error('Run detail error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
