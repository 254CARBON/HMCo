import yaml from 'js-yaml';
import {
  JobConfig,
  Runner,
  RunnerExecutionResult,
  RunnerTestResult,
  SupportedRuntime,
} from '../sdk/runner';

type UISRuntime = SupportedRuntime;

export interface UISSpec extends Record<string, unknown> {
  name: string;
  runtime: UISRuntime;
  schedule?: {
    cron?: string;
    timezone?: string;
  };
}

export interface UISValidationResult {
  valid: boolean;
  errors: string[];
  spec?: UISSpec;
  normalized?: string;
  runtime?: UISRuntime;
}

export interface JobExecutionSummary extends RunnerExecutionResult {
  jobConfig: JobConfig;
}

type RunnerFactory = (config: JobConfig) => Runner;

const SUPPORTED_RUNTIMES: UISRuntime[] = ['spark', 'flink', 'seatunnel'];

export class JobExecutor {
  constructor(
    private readonly createRunner: RunnerFactory = (config) => new Runner(config)
  ) {}

  validateUIS(uis: string): UISValidationResult {
    const errors: string[] = [];

    if (!uis || !uis.trim()) {
      return {
        valid: false,
        errors: ['UIS specification cannot be empty'],
      };
    }

    let spec: UISSpec | undefined;

    try {
      spec = this.parseSpec(uis);
    } catch (error) {
      const err = error as Error;
      errors.push(err.message);
      return {
        valid: false,
        errors,
      };
    }

    if (!spec) {
      return {
        valid: false,
        errors: ['UIS specification parsed to an empty object'],
      };
    }

    if (!spec.name || typeof spec.name !== 'string') {
      errors.push('UIS specification must include a "name" field.');
    }

    const runtime =
      typeof spec.runtime === 'string' ? spec.runtime.toLowerCase() : undefined;
    if (!runtime) {
      errors.push('UIS specification must include a "runtime" field.');
    } else if (!SUPPORTED_RUNTIMES.includes(runtime as UISRuntime)) {
      errors.push(
        `Unsupported runtime "${spec.runtime}". Supported runtimes: ${SUPPORTED_RUNTIMES.join(', ')}`
      );
    } else {
      spec.runtime = runtime as UISRuntime;
    }

    const hasProviderBlock =
      typeof spec.provider === 'object' && spec.provider !== null;
    const hasSourceBlock =
      typeof spec.source === 'object' && spec.source !== null;

    if (!hasProviderBlock && !hasSourceBlock) {
      errors.push(
        'UIS specification must include either a "provider" or "source" definition.'
      );
    }

    const normalized = spec ? JSON.stringify(spec, null, 2) : undefined;

    const result: UISValidationResult = {
      valid: errors.length === 0,
      errors,
      spec,
      runtime: spec?.runtime,
    };
    if (normalized !== undefined) {
      (result as any).normalized = normalized;
    }
    return result;
  }

  async testConnection(uis: string): Promise<RunnerTestResult> {
    const validation = this.validateUIS(uis);
    if (!validation.valid || !validation.spec) {
      throw new Error(
        `UIS validation failed: ${validation.errors.join('; ')}`
      );
    }

    const jobConfig = this.compile(validation.spec);
    const runner = this.createRunner(jobConfig);
    return runner.testConnection();
  }

  async executeRun(
    providerId: string,
    uis: string
  ): Promise<JobExecutionSummary> {
    const validation = this.validateUIS(uis);
    if (!validation.valid || !validation.spec) {
      throw new Error(
        `UIS validation failed: ${validation.errors.join('; ')}`
      );
    }

    const jobConfig = this.compile(validation.spec, { providerId });
    const runner = this.createRunner(jobConfig);
    const result = await runner.execute();

    return {
      ...result,
      jobConfig,
    };
  }

  private compile(
    spec: UISSpec,
    metadata: Record<string, unknown> = {}
  ): JobConfig {
    const compiler = this.selectCompiler(spec.runtime);
    return compiler(spec, metadata);
  }

  private selectCompiler(runtime: UISRuntime) {
    switch (runtime) {
      case 'spark':
        return this.sparkCompiler;
      case 'flink':
        return this.flinkCompiler;
      case 'seatunnel':
        return this.seatunnelCompiler;
      default:
        throw new Error(`Unsupported runtime "${runtime}"`);
    }
  }

  private sparkCompiler(
    spec: UISSpec,
    metadata: Record<string, unknown>
  ): JobConfig {
    return {
      name: spec.name,
      runtime: 'spark',
      spec,
      metadata: {
        engine: 'spark',
        ...metadata,
      },
    };
  }

  private flinkCompiler(
    spec: UISSpec,
    metadata: Record<string, unknown>
  ): JobConfig {
    return {
      name: spec.name,
      runtime: 'flink',
      spec,
      metadata: {
        engine: 'flink',
        ...metadata,
      },
    };
  }

  private seatunnelCompiler(
    spec: UISSpec,
    metadata: Record<string, unknown>
  ): JobConfig {
    return {
      name: spec.name,
      runtime: 'seatunnel',
      spec,
      metadata: {
        engine: 'seatunnel',
        ...metadata,
      },
    };
  }

  private parseSpec(uis: string): UISSpec {
    const source = uis.trim();

    const parseErrors: string[] = [];

    if (source.startsWith('{') || source.startsWith('[')) {
      try {
        const parsed = JSON.parse(source) as Record<string, unknown>;
        return this.ensureUISSpec(parsed);
      } catch (error) {
        const err = error as Error;
        parseErrors.push(`JSON parse error: ${err.message}`);
      }
    }

    try {
      const parsed = yaml.load(source, {
        schema: yaml.DEFAULT_SCHEMA,
      }) as Record<string, unknown>;
      return this.ensureUISSpec(parsed);
    } catch (error) {
      const err = error as Error;
      parseErrors.push(`YAML parse error: ${err.message}`);
    }

    throw new Error(parseErrors.join('; '));
  }

  private ensureUISSpec(value: unknown): UISSpec {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('UIS specification must parse to an object');
    }

    const spec = { ...(value as Record<string, unknown>) };

    if (typeof spec.runtime === 'string') {
      spec.runtime = spec.runtime.toLowerCase();
    }

    return spec as UISSpec;
  }
}

const CRON_SEGMENT = /^[\d*/?,#\-LWA]+$/i;
const CRON_MACROS = new Set([
  '@yearly',
  '@annually',
  '@monthly',
  '@weekly',
  '@daily',
  '@midnight',
  '@hourly',
]);

export function validateCronExpression(value: string): boolean {
  if (!value) {
    return true;
  }

  const trimmed = value.trim();
  if (CRON_MACROS.has(trimmed.toLowerCase())) {
    return true;
  }

  const segments = trimmed.split(/\s+/);
  if (segments.length < 5 || segments.length > 6) {
    return false;
  }

  return segments.every((segment) => CRON_SEGMENT.test(segment));
}
