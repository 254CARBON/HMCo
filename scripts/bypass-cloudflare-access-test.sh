#!/bin/bash

# Test if services work when bypassing Cloudflare Access
# This helps diagnose if the redirect loop is from Access configuration

echo "===== Testing Service Access ====="
echo ""

# Test 1: Direct access through tunnel
echo "Test 1: Checking tunnel is working..."
curl -s -o /dev/null -w "Portal via tunnel: %{http_code}\n" \
  -H "Host: portal.254carbon.com" \
  http://ingress-nginx-controller.ingress-nginx:80 \
  --connect-to portal.254carbon.com:80:ingress-nginx-controller.ingress-nginx:80 2>&1 || \
  echo "Cannot test from outside cluster"

# Test 2: Check redirect chain
echo ""
echo "Test 2: Checking redirect chain..."
echo "Following redirects from https://portal.254carbon.com..."
curl -sL -D - https://portal.254carbon.com -o /dev/null 2>&1 | grep -E "HTTP|Location" | head -20

echo ""
echo "Test 3: Checking Cloudflare Access flow..."
FIRST_RESPONSE=$(curl -sI https://portal.254carbon.com 2>&1)
echo "$FIRST_RESPONSE" | head -5

if echo "$FIRST_RESPONSE" | grep -q "cloudflareaccess.com"; then
  echo "✓ Cloudflare Access is intercepting (normal)"
  echo ""
  echo "The redirect to cloudflareaccess.com is EXPECTED."
  echo "After you login, Access should redirect back to portal."
  echo ""
  echo "If you're stuck in a loop:"
  echo "1. Clear ALL browser data (not just cache)"
  echo "2. Close ALL browser windows"
  echo "3. Reopen browser in Incognito"
  echo "4. Access: https://portal.254carbon.com"
fi

echo ""
echo "===== Browser Cache Fix ====="
echo ""
echo "CRITICAL: You MUST clear browser data completely"
echo ""
echo "Chrome/Edge:"
echo "  1. Close ALL browser windows"
echo "  2. Reopen"
echo "  3. Press: Ctrl + Shift + Del"
echo "  4. Select: 'All time'"
echo "  5. Check: Cookies and Cached images"
echo "  6. Click: 'Clear data'"
echo "  7. Close browser again"
echo "  8. Open Incognito: Ctrl + Shift + N"
echo "  9. Go to: https://portal.254carbon.com"
echo ""
echo "Firefox:"
echo "  1. Press: Ctrl + Shift + Del"
echo "  2. Time range: Everything"
echo "  3. Check: Cookies and Cache"
echo "  4. Click: Clear Now"
echo "  5. Restart Firefox"
echo "  6. Open Private Window: Ctrl + Shift + P"
echo ""




