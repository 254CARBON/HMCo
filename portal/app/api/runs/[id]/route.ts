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

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Run detail error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
