"""Small artifact and OpenAI-compatible transport helpers (no SDK required)."""

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def image_block(path):
    path = Path(path)
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def content(payload):
    payload = dict(payload)
    images = payload.pop("_images", [])
    return [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            *[block for item in images for block in (
                {"type": "text", "text": item.get("caption", "")}, image_block(item["path"]))]]


class Client:
    def __init__(self, config):
        settings = {}
        env_file = Path(config.get("env_file", Path(__file__).resolve().parents[1] / ".env"))
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                key, separator, value = line.strip().removeprefix("export ").partition("=")
                if separator and key in {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_BASE_URL"}:
                    settings[key] = value.strip().strip("\"'")
        self.model = config["model"]
        self.effort = config.get("reasoning_effort")
        self.timeout = config.get("api_timeout", 300)
        self.base = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("CODEX_BASE_URL")
                     or settings.get("OPENAI_BASE_URL") or settings.get("CODEX_BASE_URL")
                     or "https://api.openai.com/v1").rstrip("/")
        self.key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
                    or settings.get("OPENAI_API_KEY") or settings.get("CODEX_API_KEY"))
        if not self.key:
            raise ValueError("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL)")
        self.calls = 0
        self.last_trace_directory = None

    def complete(self, messages, *, schema=None, tools=None, artifact=None):
        from .vlm_trace import VLMTrace
        if artifact is None:
            raise ValueError("A VLM artifact path is required to record every request")
        payload = {"model": self.model, "messages": messages, "stream": False}
        if self.effort:
            payload["reasoning_effort"] = self.effort
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "demonstration_hierarchy", "strict": True, "schema": schema}}
        if tools:
            payload.update(tools=tools, tool_choice="required", parallel_tool_calls=False)
        endpoint = self.base if self.base.endswith("/chat/completions") else self.base + "/chat/completions"
        body = json.dumps(payload).encode()
        trace = VLMTrace(artifact, payload, body)
        self.last_trace_directory = str(trace.directory.resolve())
        request = Request(endpoint, data=body, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        self.calls += 1
        raw = None
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
            result = json.loads(raw)
            trace.response(raw, result)
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("refusal") or choice["finish_reason"] not in {"stop", "tool_calls"}:
                raise RuntimeError("Model refused or returned an incomplete response")
            return message
        except Exception as error:
            if isinstance(error, HTTPError):
                raw = error.read()
            trace.error(error, raw)
            raise
