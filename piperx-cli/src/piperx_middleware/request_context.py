"""Per-request identity, propagated through HTTP/MCP worker threads."""
from contextvars import ContextVar
control_session_id = ContextVar('piper_control_session_id', default=None)
