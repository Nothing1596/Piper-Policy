"""Archive exact VLM bodies, readable inputs and the actual transmitted images."""

import base64
import copy
import hashlib
import html
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import save


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


class VLMTrace:
    def __init__(self, artifact, payload, body):
        self.artifact = Path(artifact)
        self.directory = self.artifact.with_suffix("")
        initial = self.directory
        attempt = 2
        while self.directory.exists():
            self.directory = initial.with_name(f"{initial.name}_retry{attempt:03d}")
            attempt += 1
        self.started = time.monotonic()
        self.metadata = {"status": "pending", "model": payload["model"],
                         "started_at": datetime.now(timezone.utc).isoformat()}
        write_bytes(self.directory / "request.json", body)
        readable, images = copy.deepcopy(payload), []
        store = self.directory.parent / "image_store"
        for mi, message in enumerate(readable["messages"]):
            blocks = message.get("content")
            if not isinstance(blocks, list):
                continue
            caption = ""
            for bi, block in enumerate(blocks):
                if block.get("type") == "text":
                    caption = block.get("text", "")
                if block.get("type") != "image_url":
                    continue
                url = block["image_url"]["url"]
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("VLM image archiving requires an embedded image data URL")
                header, encoded = url.split(",", 1)
                mime = header[5:].split(";", 1)[0]
                suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(mime)
                if suffix is None:
                    raise ValueError(f"Unsupported VLM image MIME type: {mime}")
                raw = base64.b64decode(encoded, validate=True)
                digest = hashlib.sha256(raw).hexdigest()
                pooled = store / (digest + suffix)
                if not pooled.exists():
                    write_bytes(pooled, raw)
                local = self.directory / "input_images" / f"{len(images)+1:04d}{suffix}"
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    try:
                        os.link(pooled, local)
                    except OSError:
                        shutil.copyfile(pooled, local)
                relative = str(local.relative_to(self.directory))
                block["image_url"]["url"] = relative
                images.append({"image": relative, "sha256": digest, "mime_type": mime,
                    "message_index": mi, "content_index": bi, "role": message["role"], "caption": caption,
                    "detail": block["image_url"].get("detail")})
        self.metadata["input_image_count"] = len(images)
        save(self.directory / "request_readable.json", readable)
        save(self.directory / "images.json", images)
        save(self.directory / "metadata.json", self.metadata)
        index_path = self.artifact.parent / "calls_index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else []
        index.append({"directory": str(self.directory.resolve()), "model": payload["model"],
                      "started_at": self.metadata["started_at"]})
        save(index_path, index)
        gallery = "".join(f'<figure><figcaption>{i+1}. {html.escape(item["caption"])}</figcaption>'
            f'<img loading="lazy" src="{item["image"]}" style="max-width:100%"></figure>' for i,item in enumerate(images))
        page = ('<!doctype html><meta charset="utf-8"><title>VLM call</title>'
            '<style>body{max-width:1100px;margin:24px auto;font-family:sans-serif}pre{white-space:pre-wrap}figure{margin:18px 0}</style>'
            '<h1>VLM call</h1><p><a href="request_readable.json">Input JSON</a> · '
            '<a href="request.json">Exact input body</a> · <a href="response.json">Response</a> · '
            '<a href="output.json">Model output</a></p>'
            '<details><summary>Full input text, history and tool schemas</summary><pre>'
            + html.escape(json.dumps(readable, ensure_ascii=False, indent=2)) + '</pre></details>' + gallery)
        (self.directory / "index.html").write_text(page)

    def response(self, raw, result):
        write_bytes(self.directory / "response.json", raw)
        save(self.artifact, result)  # Preserve the previous response-only path.
        choices = result.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        save(self.directory / "output.json", message)
        text = message.get("content") or json.dumps(message.get("tool_calls", message), ensure_ascii=False, indent=2)
        (self.directory / "output.txt").write_text(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False))
        self.metadata.update(status="response_received", elapsed_s=time.monotonic()-self.started,
                             usage=result.get("usage"))
        save(self.directory / "metadata.json", self.metadata)

    def error(self, error, raw=None):
        if raw is not None:
            write_bytes(self.directory / "error_response.txt", raw)
        failure = {"type": type(error).__name__, "message": str(error)}
        save(self.directory / "error.json", failure)
        self.metadata.update(status="error", elapsed_s=time.monotonic()-self.started, error=failure)
        save(self.directory / "metadata.json", self.metadata)
