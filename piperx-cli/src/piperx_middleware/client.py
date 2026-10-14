import os
from pathlib import Path
from urllib.parse import urlparse

import httpx


class RobotClient:
    """A transport timeout is not evidence that an action failed; reuse request_id."""
    def __init__(self, url: str, token_file: Path):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Expected an HTTP(S) origin without credentials, query, or fragment.")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote access requires HTTPS or an SSH tunnel to localhost.")
        token = token_file.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise ValueError("Invalid token file.")
        self.http = httpx.Client(base_url=url.rstrip("/"), headers={"Authorization": "Bearer " + token},
                                 timeout=httpx.Timeout(10, connect=3), trust_env=False, follow_redirects=False)

    @classmethod
    def from_env(cls, *, url=None, token_file=None, root=None):
        if token_file is None:
            if os.environ.get("PIPERX_TOKEN_FILE"):
                token_file = Path(os.environ["PIPERX_TOKEN_FILE"])
            else:
                from .cli import default_root
                token_file = Path(root) / "model.token" if root is not None else default_root() / "model.token"
        return cls(url or os.environ.get("PIPERX_URL", "http://127.0.0.1:8765"), Path(token_file))

    def call(self, method, path, body=None):
        try:
            response = self.http.request(method, path, json=body)
        except httpx.TransportError as exc:
            return {"error": {"code": "transport_unknown", "message": f"{type(exc).__name__}: outcome is unknown. Query or retry with the SAME request_id; never create a replacement motion automatically."}}
        try:
            result = response.json()
        except ValueError:
            return {"error": {"code": "invalid_response", "message": f"HTTP {response.status_code}; outcome unknown."}}
        if not isinstance(result, dict) or response.is_redirect:
            return {"error": {"code": "invalid_response", "message": f"HTTP {response.status_code}; expected object, outcome unknown."}}
        if response.is_error and "error" not in result:
            return {"error": {"code": "http_error", "message": f"HTTP {response.status_code}"}}
        return result

    def close(self):
        self.http.close()
