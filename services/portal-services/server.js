import express from 'express';
import morgan from 'morgan';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import * as k8s from '@kubernetes/client-node';

const app = express();
app.disable('x-powered-by');
app.use(express.json());
app.use(morgan('tiny'));

const CONFIG_PATH = process.env.CONFIG_PATH || '/config/services.json';
const PORT = process.env.PORT || 8080;
const CF_ACCESS_CLIENT_ID = process.env.CF_ACCESS_CLIENT_ID || '';
const CF_ACCESS_CLIENT_SECRET = process.env.CF_ACCESS_CLIENT_SECRET || '';
const DEFAULT_TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 6000);

let registry = { services: [] };
let kube = null;
let coreApi = null;
let netApi = null;
const secretCache = new Map(); // key: namespace/name -> { id, secret, ts }
const servicePortCache = new Map(); // key: namespace/name:portName -> port number

const CATEGORY_DEFS = {
  monitoring: {
    label: 'Monitoring & Visualization',
    aliases: ['monitoring', 'observability', 'visualization', 'bi'],
  },
  data: {
    label: 'Data Governance',
    aliases: ['data', 'catalog', 'data catalog', 'metadata', 'governance'],
  },
  compute: {
    label: 'Compute & Query',
    aliases: ['compute', 'olap', 'sql', 'query engine', 'analytics'],
  },
  storage: {
    label: 'Storage & Secrets',
    aliases: ['storage', 'data lake', 'object storage', 'security', 'secrets'],
  },
  workflow: {
    label: 'Workflow & Orchestration',
    aliases: ['workflow', 'orchestration', 'scheduler', 'automation'],
  },
};

const DEFAULT_CATEGORY = { slug: 'other', label: 'Other' };

function normalizeCategory(rawCategory) {
  if (!rawCategory && rawCategory !== 0) return DEFAULT_CATEGORY;
  const value = String(rawCategory).trim();
  if (!value) return DEFAULT_CATEGORY;
  const normalized = value.toLowerCase();
  for (const [slug, def] of Object.entries(CATEGORY_DEFS)) {
    if (normalized === slug) {
      return { slug, label: def.label };
    }
    if (def.aliases.includes(normalized)) {
      return { slug, label: def.label };
    }
  }
  const slug = normalized
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    || DEFAULT_CATEGORY.slug;
  const label = value.charAt(0).toUpperCase() + value.slice(1);
  return { slug, label };
}

function normalizeService(service) {
  if (!service || typeof service !== 'object') return service;
  const normalized = { ...service };
  const { slug, label } = normalizeCategory(service.category ?? service.categoryLabel);
  normalized.category = slug;
  normalized.categoryLabel = label;
  return normalized;
}

function normalizeServices(list) {
  if (!Array.isArray(list)) return [];
  return list.map(normalizeService);
}

function loadRegistry() {
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf-8');
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      registry.services = normalizeServices(parsed);
    } else if (parsed && Array.isArray(parsed.services)) {
      registry.services = normalizeServices(parsed.services);
    } else {
      throw new Error('Invalid services.json format');
    }
    console.log(`Loaded ${registry.services.length} services from ${CONFIG_PATH}`);
  } catch (e) {
    console.error('Failed to load registry:', e.message);
    registry.services = [];
  }
}

loadRegistry();

// Optional auto-discovery at startup
if (String(process.env.DISCOVER_ON_START || 'false').toLowerCase() === 'true') {
  (async () => {
    try {
      const namespaces = (process.env.DISCOVERY_NAMESPACES || 'data-platform,monitoring,vault-prod')
        .split(',').map(s => s.trim()).filter(Boolean);
      const discovered = await discoverFromIngress(namespaces);
      registry.services = mergeServices(registry.services, discovered);
      console.log(`Auto-discovered ${discovered.length} services from ingress (${namespaces.join(',')})`);
    } catch (e) {
      console.warn('Auto-discovery failed:', e.message);
    }
  })();
}

app.get('/healthz', (req, res) => {
  res.json({ ok: true, services: registry.services.length });
});

app.get('/api/services', (req, res) => {
  const services = registry.services.map(s => {
    const normalized = normalizeService(s);
    return {
      id: normalized.id,
      name: normalized.name,
      url: normalized.url,
      category: normalized.category,
      categoryLabel: normalized.categoryLabel,
      description: normalized.description,
      icon: normalized.icon,
    };
  });
  res.json(services);
});

app.post('/api/services/reload', (req, res) => {
  loadRegistry();
  res.json({ reloaded: true, count: registry.services.length });
});

app.get('/api/services/status', async (req, res) => {
  const mode = (req.query.mode || 'auto').toString();
  const results = await Promise.all(
    registry.services.map(svc => checkService(svc, mode).catch(err => ({
      id: svc.id,
      status: 'unknown',
      error: err.message,
      checkedAt: new Date().toISOString(),
    })))
  );
  res.json(results);
});

// Discover services from Kubernetes Ingress annotations and merge or replace registry
// POST /api/services/discover?namespaces=data-platform,monitoring,vault-prod&mode=merge|replace
app.post('/api/services/discover', async (req, res) => {
  try {
    const namespaces = (req.query.namespaces || process.env.DISCOVERY_NAMESPACES || 'data-platform,monitoring,vault-prod')
      .toString()
      .split(',')
      .map(s => s.trim())
      .filter(Boolean);
    const mode = (req.query.mode || 'merge').toString();
    const discovered = await discoverFromIngress(namespaces);
    if (mode === 'replace') {
      registry.services = normalizeServices(discovered);
    } else {
      registry.services = mergeServices(registry.services, discovered);
    }
    res.json({ ok: true, count: registry.services.length, discovered: discovered.length, mode, namespaces });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

async function checkService(svc, mode='auto') {
  const checkedAt = new Date().toISOString();
  const timeoutMs = svc.timeoutMs || DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  const wantsExternal = mode === 'external' || (mode === 'auto' && svc.useCloudflareAccess);

  let url = svc.url;
  let headers = {};
  if (wantsExternal) {
    const hdrs = await getAccessHeadersForService(svc).catch(() => ({}));
    headers = { ...headers, ...hdrs };
  } else if (svc.internalUrl) {
    url = svc.internalUrl;
  }

  const healthUrl = joinUrl(url, svc.healthPath || '/');

  let ok = false, status = 'unknown', code = 0, latencyMs = 0;
  const start = Date.now();
  try {
    const resp = await fetch(healthUrl, { method: 'GET', headers, redirect: 'manual', signal: controller.signal });
    latencyMs = Date.now() - start;
    code = resp.status;
    ok = resp.ok || (resp.status >= 200 && resp.status < 400);
    status = ok ? 'ok' : resp.status >= 500 ? 'err' : 'warn';
  } catch (e) {
    latencyMs = Date.now() - start;
    status = e.name === 'AbortError' ? 'warn' : 'err';
  } finally {
    clearTimeout(t);
  }
  return { id: svc.id, status, code, latencyMs, checkedAt, url: healthUrl };
}

function joinUrl(base, p) {
  try {
    const u = new URL(base);
    const joined = new URL(p, u);
    return joined.toString();
  } catch {
    return base;
  }
}

app.listen(PORT, () => {
  console.log(`portal-services listening on :${PORT}`);
});

// Initialize Kubernetes client lazily (in-cluster) and helpers
function ensureKube() {
  if (coreApi) return coreApi;
  try {
    kube = new k8s.KubeConfig();
    kube.loadFromDefault();
    coreApi = kube.makeApiClient(k8s.CoreV1Api);
    netApi = kube.makeApiClient(k8s.NetworkingV1Api);
  } catch (e) {
    console.warn('Kube client not available:', e.message);
  }
  return coreApi;
}

async function getAccessHeadersForService(svc) {
  // Priority: per-service secret ref -> global env -> empty
  if (svc.cfAccessSecretRef && svc.cfAccessSecretRef.name) {
    const ns = svc.cfAccessSecretRef.namespace || process.env.POD_NAMESPACE || 'data-platform';
    const name = svc.cfAccessSecretRef.name;
    const clientIdKey = svc.cfAccessSecretRef.clientIdKey || 'client_id';
    const clientSecretKey = svc.cfAccessSecretRef.clientSecretKey || 'client_secret';
    const data = await readSecret(ns, name);
    const id = data[clientIdKey];
    const secret = data[clientSecretKey];
    if (id && secret) {
      return {
        'Cf-Access-Client-Id': bufferMaybe(id),
        'Cf-Access-Client-Secret': bufferMaybe(secret),
      };
    }
  }
  if (CF_ACCESS_CLIENT_ID && CF_ACCESS_CLIENT_SECRET) {
    return {
      'Cf-Access-Client-Id': CF_ACCESS_CLIENT_ID,
      'Cf-Access-Client-Secret': CF_ACCESS_CLIENT_SECRET,
    };
  }
  return {};
}

function bufferMaybe(value) {
  // value may already be decoded; try base64 decode if it looks like base64
  try {
    // if it contains non-base64 chars, Buffer will throw
    const decoded = Buffer.from(value, 'base64').toString('utf8');
    // heuristic: if decoding results in mostly printable chars and not binary, use it
    if (/^[\x09\x0A\x0D\x20-\x7E]+$/.test(decoded)) return decoded;
  } catch {}
  return value;
}

async function readSecret(namespace, name) {
  const key = `${namespace}/${name}`;
  const now = Date.now();
  const cached = secretCache.get(key);
  const ttlMs = Number(process.env.SECRET_CACHE_TTL_MS || 300000); // 5 min
  if (cached && (now - cached.ts) < ttlMs) return cached.data;
  const api = ensureKube();
  if (!api) throw new Error('Kubernetes client not configured');
  const resp = await api.readNamespacedSecret(name, namespace);
  const data = {};
  const raw = resp.body.data || {};
  Object.keys(raw).forEach(k => { data[k] = raw[k]; });
  secretCache.set(key, { ts: now, data });
  return data;
}

// Merge arrays of services by id, prefer existing fields, but allow discovery to add cfAccessSecretRef/internalUrl/healthPath/category/icon
function mergeServices(base, discovered) {
  const byId = new Map(normalizeServices(base).map(s => [s.id, { ...s }]));
  for (const raw of discovered) {
    const d = normalizeService(raw);
    if (!byId.has(d.id)) {
      byId.set(d.id, d);
      continue;
    }
    const cur = byId.get(d.id);
    const merged = { ...cur };
    for (const k of ['cfAccessSecretRef','internalUrl','healthPath','useCloudflareAccess']) {
      if (Object.prototype.hasOwnProperty.call(d, k) && d[k] !== undefined) {
        merged[k] = d[k];
      }
    }
    for (const k of ['category','categoryLabel','icon','description','name','url']) {
      if ((merged[k] == null || merged[k] === '') && d[k]) merged[k] = d[k];
    }
    byId.set(d.id, normalizeService(merged));
  }
  return Array.from(byId.values());
}

// Discover services from ingress annotations
// Annotation keys (prefix portal.254carbon.com/):
//  - service-id, service-name, service-category, service-description, service-icon, service-health-path
//  - cf-access-secret-name, cf-access-secret-namespace, cf-access-client-id-key, cf-access-client-secret-key
//  - use-cloudflare-access: "true"|"false"
async function discoverFromIngress(namespaces) {
  ensureKube();
  if (!netApi) throw new Error('Kubernetes networking client not available');
  const prefix = 'portal.254carbon.com/';
  const results = [];
  for (const ns of namespaces) {
    try {
      const resp = await netApi.listNamespacedIngress(ns);
      const items = resp.body.items || [];
      for (const ing of items) {
        const ann = (ing.metadata && ing.metadata.annotations) || {};
        const rules = (ing.spec && ing.spec.rules) || [];
        for (const rule of rules) {
          const host = rule.host;
          if (!host) continue;
          const http = rule.http;
          const paths = ((http && http.paths) || []).filter(p => p && p.backend && p.backend.service && p.backend.service.name);
          if (paths.length === 0) continue;

          const preferredPath = paths.find(p => !p.path || p.path === '/' || p.path === '/*') || paths[0];
          const backendService = preferredPath.backend.service;
          const backendName = backendService.name;
          const backendPortField = backendService.port || {};

          let backendPort = null;
          if (typeof backendPortField.number === 'number') {
            backendPort = backendPortField.number;
          } else if (typeof backendPortField.number === 'string' && backendPortField.number) {
            const parsed = Number(backendPortField.number);
            if (!Number.isNaN(parsed)) backendPort = parsed;
          }
          if (backendPort == null && typeof backendPortField.name === 'string' && backendPortField.name) {
            backendPort = await resolveServicePort(ing.metadata.namespace, backendName, backendPortField.name);
          }

          const ingressPath = normalizeIngressPath(preferredPath.path);
          const id = ann[prefix + 'service-id'] || (host.split('.')[0] || ing.metadata.name);
          const name = ann[prefix + 'service-name'] || capitalize(id);
          const category = ann[prefix + 'service-category'] || undefined;
          const description = ann[prefix + 'service-description'] || undefined;
          const icon = ann[prefix + 'service-icon'] || undefined;
          const healthPath = ann[prefix + 'service-health-path'] || '/';
          const useAccess = parseBool(ann[prefix + 'use-cloudflare-access'], true);
          const secretName = ann[prefix + 'cf-access-secret-name'] || undefined;
          const secretNs = ann[prefix + 'cf-access-secret-namespace'] || ing.metadata.namespace;
          const clientIdKey = ann[prefix + 'cf-access-client-id-key'] || 'client_id';
          const clientSecretKey = ann[prefix + 'cf-access-client-secret-key'] || 'client_secret';

          const url = `https://${host}/`;
          let internalUrl = undefined;
          if (backendName) {
            const portSeg = backendPort ? `:${backendPort}` : '';
            const basePath = ingressPath || '/';
            internalUrl = `http://${backendName}.${ing.metadata.namespace}.svc.cluster.local${portSeg}${basePath}`;
            if (!internalUrl.endsWith('/')) internalUrl = `${internalUrl}/`;
          }

          const entry = { id, name, url, category, description, icon, healthPath, useCloudflareAccess: useAccess };
          if (internalUrl) entry.internalUrl = internalUrl;
          if (secretName) {
            entry.cfAccessSecretRef = { name: secretName, namespace: secretNs, clientIdKey, clientSecretKey };
          }
          results.push(normalizeService(entry));
        }
      }
    } catch (e) {
      console.warn(`Ingress discovery failed in ns=${ns}:`, e.message);
    }
  }
  // Deduplicate by id preferring entries with cfAccessSecretRef
  const byId = new Map();
  for (const r of results) {
    const cur = byId.get(r.id);
    if (!cur) { byId.set(r.id, r); continue; }
    const prefer = (r.cfAccessSecretRef && !cur.cfAccessSecretRef) ? r : cur;
    byId.set(r.id, prefer);
  }
  return normalizeServices(Array.from(byId.values()));
}

async function resolveServicePort(namespace, serviceName, portName) {
  const cacheKey = `${namespace}/${serviceName}:${portName}`;
  if (servicePortCache.has(cacheKey)) return servicePortCache.get(cacheKey);
  const api = ensureKube();
  if (!api) return null;
  try {
    const resp = await api.readNamespacedService(serviceName, namespace);
    const ports = (resp.body && resp.body.spec && resp.body.spec.ports) || [];
    const match = ports.find(p => p.name === portName);
    const resolved = match && (match.port || (typeof match.targetPort === 'number' ? match.targetPort : null));
    servicePortCache.set(cacheKey, resolved || null);
    return resolved || null;
  } catch (e) {
    console.warn(`Failed to resolve service port ${namespace}/${serviceName}:${portName}:`, e.message);
    servicePortCache.set(cacheKey, null);
    return null;
  }
}

function normalizeIngressPath(p) {
  if (!p) return '/';
  const withLeading = p.startsWith('/') ? p : `/${p}`;
  const cleaned = withLeading === '/*' ? '/' : withLeading.replace(/\/\*$/, '/');
  return cleaned || '/';
}

function parseBool(v, dflt=false) {
  if (v == null) return dflt;
  const s = String(v).toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'y' || s === 'on';
}

function capitalize(s) {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}
