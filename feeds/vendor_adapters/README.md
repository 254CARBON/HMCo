# Vendor Adapters

This directory contains pluggable vendor adapter implementations for market data feeds.

## Adapter Interface

All adapters should implement the `VendorAdapter` interface defined in `base_adapter.py`.

## Supported Vendors

### Curve Data Providers
- **ICE** (Intercontinental Exchange)
- **CME** (Chicago Mercantile Exchange)
- **Proprietary feeds** (custom implementations)

### AIS/Marine Data Providers
- **MarineTraffic**
- **VesselFinder**
- **Custom AIS feeds**

## Usage

```python
from feeds.vendor_adapters import get_adapter

# Get adapter instance
adapter = get_adapter('ice', config={
    'api_key': 'your_key',
    'endpoint': 'https://api.ice.com/v1'
})

# Fetch curve snapshot
snapshot = adapter.fetch_curve_snapshot('PJM_WH', '2025-01-15')
```

## Adding New Adapters

1. Create a new file `{vendor_name}_adapter.py`
2. Implement the `VendorAdapter` interface
3. Register in `__init__.py`
4. Add tests in `tests/`
5. Update this README

## Configuration

Adapter credentials should be stored in ExternalSecrets and accessed via environment variables:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: vendor-api-keys
spec:
  secretStoreRef:
    name: vault-backend
  target:
    name: vendor-credentials
  data:
    - secretKey: ICE_API_KEY
      remoteRef:
        key: vendor/ice
        property: api_key
```

## Security

- **Never commit API keys** to the repository
- Use ExternalSecrets for all sensitive credentials
- Rotate keys quarterly
- Log all API calls for audit purposes
