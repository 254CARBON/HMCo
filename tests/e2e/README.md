# E2E Tests with Playwright

This directory contains Playwright-based end-to-end tests for the HMCo portal.

## Setup

Install dependencies:

```bash
npm install
npx playwright install chromium
```

## Running Tests

### Run all tests

```bash
npm test
```

### Run tests with E2E bypass (for CI/testing)

The E2E bypass allows tests to skip Cloudflare Access authentication in development/test environments:

```bash
# Set E2E_BYPASS=1 in the portal environment
cd ../../portal
E2E_BYPASS=1 npm run dev

# In another terminal, run the tests
cd tests/e2e
npm test
```

### Run tests in headed mode

```bash
npm run test:headed
```

### Run tests in debug mode

```bash
npm run test:debug
```

## E2E Bypass Feature

The E2E bypass feature allows automated tests to skip Cloudflare Access authentication by:

1. Setting the `E2E_BYPASS=1` environment variable on the portal server
2. Including the `x-e2e-bypass` header in HTTP requests

**Security Note**: This bypass is only active when explicitly enabled via the `E2E_BYPASS` environment variable and should NEVER be enabled in production.

## Test Structure

- `smoke.spec.ts` - Basic smoke tests that verify:
  - Redirect to Cloudflare Access when authentication is required
  - Bypass functionality with E2E test header in dev mode
  - Public paths are accessible without authentication
  - Basic page loading and content verification

## CI Integration

These tests run automatically in GitHub Actions with the E2E bypass enabled. See `.github/workflows/supplychain-and-e2e.yml` for the CI configuration.
