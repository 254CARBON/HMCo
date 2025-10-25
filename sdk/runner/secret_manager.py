"""
Secret management integration with HashiCorp Vault.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class SecretManagerError(Exception):
    """Exception raised when secret management fails."""
    pass


class SecretManager:
    """Manages secrets using HashiCorp Vault."""

    def __init__(self, vault_address: str, vault_token: Optional[str] = None,
                 role_id: Optional[str] = None, secret_id: Optional[str] = None,
                 mount_path: str = "secret", timeout_seconds: int = 30):
        """Initialize Vault secret manager."""
        self.vault_address = vault_address.rstrip('/')
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.role_id = role_id or os.getenv("VAULT_ROLE_ID")
        self.secret_id = secret_id or os.getenv("VAULT_SECRET_ID")
        self.mount_path = mount_path
        self.timeout_seconds = timeout_seconds
        self.session = None
        self._token_cache: Optional[str] = None
        self._token_expiry: Optional[float] = None

        self._setup_session()

    def _setup_session(self):
        """Setup HTTP session with retry configuration."""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set timeout
        self.session.timeout = self.timeout_seconds

    def _get_auth_token(self) -> str:
        """Get authentication token for Vault."""
        if self.vault_token:
            return self.vault_token

        if self.role_id and self.secret_id:
            return self._authenticate_with_approle()

        raise SecretManagerError("No Vault authentication configured")

    def _authenticate_with_approle(self) -> str:
        """Authenticate using AppRole authentication."""
        try:
            url = f"{self.vault_address}/v1/auth/approle/login"

            payload = {
                "role_id": self.role_id,
                "secret_id": self.secret_id
            }

            response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()

            data = response.json()
            auth = data["auth"]

            token = auth["client_token"]
            lease_duration = auth["lease_duration"]

            # Cache token
            self._token_cache = token
            self._token_expiry = time.time() + lease_duration - 60  # Refresh 1 minute early

            logger.info(f"Successfully authenticated with Vault using AppRole")
            return token

        except Exception as e:
            logger.error(f"AppRole authentication failed: {e}")
            raise SecretManagerError(f"Vault authentication failed: {e}")

    def get_secret(self, secret_path: str) -> Dict[str, Any]:
        """Retrieve secret from Vault."""
        try:
            token = self._get_auth_token()

            # Build Vault path
            full_path = f"{self.mount_path}/{secret_path}".strip('/')

            url = f"{self.vault_address}/v1/{full_path}"

            headers = {
                "X-Vault-Token": token
            }

            response = self.session.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()

            data = response.json()
            secret_data = data.get("data", {})

            logger.info(f"Successfully retrieved secret: {secret_path}")
            return secret_data

        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_path}: {e}")
            raise SecretManagerError(f"Secret retrieval failed: {e}")

    def get_secret_value(self, secret_path: str, key: str) -> Any:
        """Retrieve specific value from secret."""
        secret_data = self.get_secret(secret_path)
        return secret_data.get(key)

    def list_secrets(self, path: str) -> List[str]:
        """List secrets under a path."""
        try:
            token = self._get_auth_token()

            # Build Vault path
            full_path = f"{self.mount_path}/{path}".strip('/')

            url = f"{self.vault_address}/v1/{full_path}"

            headers = {
                "X-Vault-Token": token
            }

            response = self.session.get(url, headers=headers, params={"list": "true"}, timeout=self.timeout_seconds)
            response.raise_for_status()

            data = response.json()
            keys = data.get("data", {}).get("keys", [])

            logger.info(f"Listed {len(keys)} secrets under: {path}")
            return keys

        except Exception as e:
            logger.error(f"Failed to list secrets under {path}: {e}")
            raise SecretManagerError(f"Secret listing failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Check Vault health status."""
        try:
            url = f"{self.vault_address}/v1/sys/health"

            response = self.session.get(url, timeout=10)
            data = response.json()

            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "initialized": data.get("initialized", False),
                "sealed": data.get("sealed", True),
                "standby": data.get("standby", False),
                "version": data.get("version", "unknown")
            }

        except Exception as e:
            logger.error(f"Vault health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def is_enabled(self) -> bool:
        """Check if secret management is enabled and available."""
        if not self.vault_token and not (self.role_id and self.secret_id):
            return False

        try:
            health = self.health_check()
            return health["status"] == "healthy" and not health["sealed"]
        except Exception:
            return False
