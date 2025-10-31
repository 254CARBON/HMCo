import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const spec = await request.json();
    const workflowId = `${spec.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    
    const workflow = {
      name: workflowId,
      description: spec.description,
      schedule: spec.schedule || '0 0 * * *',
      tasks: [
        {
          name: 'compile_uis',
          type: 'SHELL',
          command: `python /app/sdk/uis/compilers/seatunnel/compiler.py --spec /app/sdk/uis/specs/${spec.name}.json`
        },
        {
          name: 'run_ingestion',
          type: 'SEATUNNEL',
          dependsOn: ['compile_uis']
        }
      ]
    };
    
    return NextResponse.json({ workflow_id: workflowId, workflow });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to generate DAG' }, { status: 500 });
  }
}
