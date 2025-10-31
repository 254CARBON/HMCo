import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const spec = await request.json();
    
    const preview = {
      schema: {
        fields: [
          { name: 'id', type: 'string', nullable: false },
          { name: 'timestamp', type: 'datetime', nullable: false },
          { name: 'value', type: 'float64', nullable: true }
        ]
      },
      sample: [
        { id: '1', timestamp: '2025-10-31T00:00:00Z', value: 42.5 },
        { id: '2', timestamp: '2025-10-31T01:00:00Z', value: 43.2 }
      ]
    };
    
    return NextResponse.json(preview);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to generate preview' }, { status: 500 });
  }
}
