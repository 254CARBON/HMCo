"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Runner = void 0;
const fs_1 = require("fs");
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
const child_process_1 = require("child_process");
class Runner {
    constructor(config) {
        this.config = config;
    }
    async execute() {
        const tmpDir = await fs_1.promises.mkdtemp(path_1.default.join(os_1.default.tmpdir(), 'hmco-uis-'));
        const specPath = path_1.default.join(tmpDir, 'spec.json');
        await fs_1.promises.writeFile(specPath, JSON.stringify(this.config.spec), 'utf8');
        const pythonBin = process.env.RUNNER_PYTHON || process.env.RUNNER_PYTHON_BIN || 'python3';
        const runnerDir = path_1.default.resolve(__dirname, '../../sdk/runner');
        const runnerMain = path_1.default.join(runnerDir, 'main.py');
        const args = [runnerMain, 'sample', specPath];
        const configPath = process.env.RUNNER_CONFIG || process.env.RUNNER_CONFIG_PATH;
        if (configPath) {
            args.push('--config', configPath);
        }
        const start = Date.now();
        const { code, stdout, stderr } = await this.spawnCollect(pythonBin, args, {
            cwd: runnerDir,
            env: { ...process.env },
        });
        const durationMs = Date.now() - start;
        const logsCombined = [stdout, stderr].filter(Boolean).join('\n');
        const status = code === 0 ? 'success' : 'failed';
        const result = {
            status,
            recordsIngested: 0,
            recordsFailed: status === 'success' ? 0 : 1,
            durationMs,
            logs: logsCombined.split('\n'),
        };
        if (code !== 0) {
            result.errorMessage = stderr || 'Execution failed';
        }
        return result;
    }
    async testConnection() {
        const pythonBin = process.env.RUNNER_PYTHON || process.env.RUNNER_PYTHON_BIN || 'python3';
        const runnerDir = path_1.default.resolve(__dirname, '../../sdk/runner');
        const runnerMain = path_1.default.join(runnerDir, 'main.py');
        const { code, stdout, stderr } = await this.spawnCollect(pythonBin, [runnerMain, 'health'], {
            cwd: runnerDir,
            env: { ...process.env },
        });
        return { ok: code === 0, details: stdout || stderr };
    }
    spawnCollect(cmd, args, opts) {
        return new Promise((resolve, reject) => {
            const child = (0, child_process_1.spawn)(cmd, args, { cwd: opts.cwd, env: opts.env });
            let stdout = '';
            let stderr = '';
            child.stdout.on('data', (d) => {
                stdout += d.toString();
            });
            child.stderr.on('data', (d) => {
                stderr += d.toString();
            });
            child.on('error', (err) => reject(err));
            child.on('close', (code) => resolve({ code, stdout, stderr }));
        });
    }
}
exports.Runner = Runner;
//# sourceMappingURL=runner.js.map