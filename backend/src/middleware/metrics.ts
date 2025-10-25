import client from 'prom-client';
import type { Request, Response, NextFunction } from 'express';

export const register = new client.Registry();
client.collectDefaultMetrics({ register });

const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status'] as const,
});

register.registerMetric(httpRequestsTotal);

export function metricsMiddleware(req: Request, res: Response, next: NextFunction) {
  const method = req.method;
  const route = (req.route && req.route.path) || req.path || 'unknown';
  res.on('finish', () => {
    const status = String(res.statusCode);
    httpRequestsTotal.labels(method, route, status).inc();
  });
  next();
}

