"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createProvidersRouter = createProvidersRouter;
const express_1 = __importDefault(require("express"));
function parseQueryNumber(value, fallback) {
    if (typeof value !== 'string') {
        return fallback;
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? fallback : parsed;
}
function createProvidersRouter(providersService) {
    const router = express_1.default.Router();
    router.get('/', async (req, res) => {
        try {
            const { status } = req.query;
            const limit = parseQueryNumber(req.query.limit, 50);
            const offset = parseQueryNumber(req.query.offset, 0);
            const result = await providersService.listProviders(typeof status === 'string' ? status : undefined, limit, offset);
            res.json(result);
        }
        catch (error) {
            const err = error;
            res.status(500).json({ error: err.message });
        }
    });
    router.post('/', async (req, res) => {
        try {
            const { name, type, uis, config, schedule } = req.body;
            if (!name || !type || !uis) {
                return res.status(400).json({ error: 'Missing required fields' });
            }
            const provider = await providersService.createProvider({
                name,
                type,
                uis,
                config,
                schedule,
            });
            res.status(201).json(provider);
        }
        catch (error) {
            const err = error;
            res.status(500).json({ error: err.message });
        }
    });
    router.get('/:id', async (req, res) => {
        try {
            const provider = await providersService.getProvider(req.params.id);
            res.json(provider);
        }
        catch (error) {
            const err = error;
            res.status(404).json({ error: err.message });
        }
    });
    router.patch('/:id', async (req, res) => {
        try {
            const provider = await providersService.updateProvider(req.params.id, req.body ?? {});
            res.json(provider);
        }
        catch (error) {
            const err = error;
            res.status(500).json({ error: err.message });
        }
    });
    router.delete('/:id', async (req, res) => {
        try {
            await providersService.deleteProvider(req.params.id);
            res.json({ success: true });
        }
        catch (error) {
            const err = error;
            res.status(500).json({ error: err.message });
        }
    });
    return router;
}
//# sourceMappingURL=providers.js.map