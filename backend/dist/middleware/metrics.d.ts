import client from 'prom-client';
import type { Request, Response, NextFunction } from 'express';
export declare const register: client.Registry<"text/plain; version=0.0.4; charset=utf-8">;
export declare function metricsMiddleware(req: Request, res: Response, next: NextFunction): void;
//# sourceMappingURL=metrics.d.ts.map