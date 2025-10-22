#!/bin/bash
# Update Cloudflare Tunnel Routes via API
# Adds RAPIDS and DolphinScheduler routes to existing tunnel configuration

set -e

# Cloudflare credentials
ACCOUNT_ID="0c93c74d5269a228e91d4bf91c547f56"
API_TOKEN="xZbVon568Jv5lUE8Ar-kzfQetT_PlknJAqype711"
TUNNEL_ID="291bc289-e3c3-4446-a9ad-8e327660ecd5"

echo "=================================="
echo "Updating Cloudflare Tunnel Routes"
echo "=================================="
echo

# Tunnel configuration with all routes
TUNNEL_CONFIG=$(cat <<'EOF'
{
  "config": {
    "ingress": [
      {
        "hostname": "portal.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "portal.254carbon.com"
        }
      },
      {
        "hostname": "www.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "www.254carbon.com"
        }
      },
      {
        "hostname": "datahub.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "datahub.254carbon.com"
        }
      },
      {
        "hostname": "grafana.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "grafana.254carbon.com"
        }
      },
      {
        "hostname": "superset.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "superset.254carbon.com"
        }
      },
      {
        "hostname": "trino.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "trino.254carbon.com"
        }
      },
      {
        "hostname": "vault.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "vault.254carbon.com"
        }
      },
      {
        "hostname": "minio.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "minio.254carbon.com"
        }
      },
      {
        "hostname": "dolphin.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "dolphin.254carbon.com"
        }
      },
      {
        "hostname": "dolphinscheduler.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "dolphinscheduler.254carbon.com"
        }
      },
      {
        "hostname": "harbor.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "harbor.254carbon.com"
        }
      },
      {
        "hostname": "lakefs.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "lakefs.254carbon.com"
        }
      },
      {
        "hostname": "rapids.254carbon.com",
        "service": "http://ingress-nginx-controller.ingress-nginx:80",
        "originRequest": {
          "httpHostHeader": "rapids.254carbon.com"
        }
      },
      {
        "service": "http_status:404"
      }
    ]
  }
}
EOF
)

echo "Updating tunnel configuration..."
response=$(curl -s -X PUT \
  "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/configurations" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  --data "$TUNNEL_CONFIG")

success=$(echo $response | jq -r '.success')

if [ "$success" == "true" ]; then
  echo "✓ Tunnel routes updated successfully"
  echo
  echo "Updated routes:"
  echo $response | jq -r '.result.config.ingress[] | select(.hostname) | "  - \(.hostname)"'
else
  echo "✗ Failed to update tunnel routes"
  echo "Error:"
  echo $response | jq -r '.errors[]'
fi

echo
echo "=================================="
echo "Tunnel Routes Update Complete"
echo "=================================="
echo
echo "Changes will be live in 1-2 minutes"
echo "Test access: https://rapids.254carbon.com"
echo

