"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.register = void 0;
exports.metricsMiddleware = metricsMiddleware;
const prom_client_1 = __importDefault(require("prom-client"));
exports.register = new prom_client_1.default.Registry();
prom_client_1.default.collectDefaultMetrics({ register: exports.register });
const httpRequestsTotal = new prom_client_1.default.Counter({
    name: 'http_requests_total',
    help: 'Total number of HTTP requests',
    labelNames: ['method', 'route', 'status'],
});
exports.register.registerMetric(httpRequestsTotal);
function metricsMiddleware(req, res, next) {
    const method = req.method;
    const route = (req.route && req.route.path) || req.path || 'unknown';
    res.on('finish', () => {
        const status = String(res.statusCode);
        httpRequestsTotal.labels(method, route, status).inc();
    });
    next();
}
//# sourceMappingURL=metrics.js.map